"""
08_bertopic.py — BERTopic Topic Modeling

Fits a BERTopic model on LLM-generated summaries from 11_summarize. Bursts
with null summaries (too short, monologue, or error) are excluded from modeling
but their burst_ids are still recorded with a null topic in burst_topics.parquet.

Embeddings are saved to disk on first run and reused on subsequent runs with
different HDBSCAN parameters, skipping the expensive embedding step.

Reads:
  outputs/11_summarize/summaries.parquet
  outputs/07_segment/bursts.parquet       (for timestamps)

Writes:
  outputs/08_bertopic/burst_topics.parquet     burst_id → topic assignment
  outputs/08_bertopic/topic_model/             serialized BERTopic model
  outputs/08_bertopic/embeddings.npy           pre-computed embeddings
  outputs/08_bertopic/topic_info.json          topic labels + representative words
  outputs/08_bertopic/topic_over_time.json     topic frequency by month
  outputs/08_bertopic/topic_viz.html           interactive intertopic distance map
  outputs/08_bertopic/topic_sizes.png          top 25 topics by burst count
  outputs/08_bertopic/topic_words.png          top words grid for top 12 topics
  outputs/08_bertopic/topics_over_time.png     top 10 topics frequency over time
  outputs/08_bertopic/bertopic_report.json     run summary

Run from project root:
    python message_miner/08_bertopic.py [--min-cluster-size N] [--force]
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Corporate network SSL + CDN bypass.
# huggingface_hub uses httpx internally. Patch both Client and AsyncClient so
# every instance skips SSL verification. Also disable HF's XET/resolve-cache
# CDN acceleration which requires auth and often 403s on corporate networks.
import os as _os
_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")          # disable resolve-cache CDN
_os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "0") # allow token passthrough

import httpx as _httpx

_orig_client_init = _httpx.Client.__init__
def _no_ssl_client_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_client_init(self, *args, **kwargs)
_httpx.Client.__init__ = _no_ssl_client_init

_orig_async_init = _httpx.AsyncClient.__init__
def _no_ssl_async_init(self, *args, **kwargs):
    kwargs["verify"] = False
    _orig_async_init(self, *args, **kwargs)
_httpx.AsyncClient.__init__ = _no_ssl_async_init

warnings.filterwarnings("ignore", message=".*SSL.*")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

SUMMARIES_PATH = Path("outputs/11_summarize/summaries.parquet")
BURSTS_PATH    = Path("outputs/07_segment/bursts.parquet")
OUTPUT_DIR     = Path("outputs/08_bertopic")
EMBEDDINGS_PATH  = OUTPUT_DIR / "embeddings_summary.npy"
BURST_IDS_PATH   = OUTPUT_DIR / "embedded_burst_ids_summary.json"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_or_compute_embeddings(
    docs: list[str], burst_ids: list[str], model
) -> np.ndarray:
    """
    Load embeddings from disk if they match the current doc set.
    Otherwise compute fresh and save.
    """
    if EMBEDDINGS_PATH.exists() and BURST_IDS_PATH.exists():
        saved_ids = json.loads(BURST_IDS_PATH.read_text(encoding="utf-8"))
        if saved_ids == burst_ids:
            print("  Loading saved embeddings from disk...")
            return np.load(str(EMBEDDINGS_PATH))
        else:
            print("  Saved embeddings don't match current burst set — re-embedding...")

    print(f"  Embedding {len(docs):,} bursts with {EMBEDDING_MODEL_NAME}...")
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=256)
    np.save(str(EMBEDDINGS_PATH), embeddings)
    BURST_IDS_PATH.write_text(json.dumps(burst_ids), encoding="utf-8")
    print(f"  Embeddings saved to {EMBEDDINGS_PATH}")
    return embeddings


def _get_topic_label(topic_id: int, topic_info_df: pd.DataFrame) -> str:
    if topic_id == -1:
        return "outlier"
    row = topic_info_df[topic_info_df["Topic"] == topic_id]
    if row.empty:
        return f"topic_{topic_id}"
    return str(row.iloc[0]["Name"])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fit BERTopic model on burst text")
    parser.add_argument(
        "--min-cluster-size", type=int, default=15,
        help="HDBSCAN min_cluster_size — controls topic granularity (default: 15)"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "burst_topics.parquet").exists() and not args.force:
        answer = input("outputs/08_bertopic/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 08_bertopic ===\n")
    print(f"Input:            summaries (outputs/11_summarize/summaries.parquet)")
    print(f"min_cluster_size: {args.min_cluster_size}\n")

    for path in (SUMMARIES_PATH, BURSTS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    # ── Load and filter ───────────────────────────────────────────────────────
    summaries_df = pd.read_parquet(SUMMARIES_PATH)
    bursts_df    = pd.read_parquet(BURSTS_PATH)[["burst_id", "chat_id", "start_time"]]
    print(f"Loaded {len(summaries_df):,} summaries, {len(bursts_df):,} bursts.\n")

    modelable = summaries_df[summaries_df["summary"].notna()].copy()
    excluded  = len(summaries_df) - len(modelable)
    print(f"Bursts with summaries (eligible for modeling): {len(modelable):,}")
    print(f"Excluded (too short / monologue / error):      {excluded:,}\n")

    # Join timestamps for topics-over-time
    modelable = modelable.merge(bursts_df[["burst_id", "start_time"]], on="burst_id", how="left")

    docs      = modelable["summary"].tolist()
    burst_ids = modelable["burst_id"].tolist()

    # ── Imports (deferred — heavy ML libs) ───────────────────────────────────
    print("Loading models...")
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    # ── Embedding ─────────────────────────────────────────────────────────────
    print("\nEmbedding step:")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = _load_or_compute_embeddings(docs, burst_ids, embedding_model)

    # ── BERTopic configuration ────────────────────────────────────────────────
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(stop_words="english")
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
    representation_model = {
        "KeyBERT": KeyBERTInspired(),
        "MMR": MaximalMarginalRelevance(diversity=0.3),
    }

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        verbose=True,
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    print("\nFitting BERTopic...")
    topics, probs = topic_model.fit_transform(docs, embeddings)

    topic_info_df = topic_model.get_topic_info()
    n_topics = len(topic_info_df[topic_info_df["Topic"] != -1])
    n_outliers = int((np.array(topics) == -1).sum())
    print(f"\nTopics discovered: {n_topics}")
    print(f"Outliers:          {n_outliers} ({100 * n_outliers / len(docs):.1f}%)\n")

    # ── burst_topics.parquet ──────────────────────────────────────────────────
    topic_label_map = {
        int(row["Topic"]): str(row["Name"])
        for _, row in topic_info_df.iterrows()
    }
    topic_label_map[-1] = "outlier"

    # probs may be 1-D (one prob per doc) or 2-D depending on BERTopic version
    if hasattr(probs, "ndim") and probs.ndim == 2:
        doc_probs = probs[np.arange(len(topics)), np.maximum(np.array(topics), 0)]
    else:
        doc_probs = np.array(probs)

    burst_topics_df = pd.DataFrame({
        "burst_id": burst_ids,
        "topic_id": [int(t) for t in topics],
        "topic_label": [topic_label_map.get(int(t), f"topic_{t}") for t in topics],
        "topic_probability": doc_probs.astype("float64"),
    })
    burst_topics_df["topic_id"] = burst_topics_df["topic_id"].astype("int32")
    burst_topics_df.to_parquet(OUTPUT_DIR / "burst_topics.parquet", index=False)

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "topic_model.pkl"
    topic_model.save(str(model_path), serialization="pickle", save_ctfidf=True)
    print(f"Model saved to {model_path}")

    # ── topic_info.json ───────────────────────────────────────────────────────
    topic_info_list = []
    for _, row in topic_info_df.iterrows():
        tid = int(row["Topic"])
        if tid == -1:
            continue
        words_with_scores = topic_model.get_topic(tid)
        words = [w for w, _ in words_with_scores[:10]] if words_with_scores else []
        topic_info_list.append({
            "topic_id": tid,
            "label": str(row["Name"]),
            "count": int(row["Count"]),
            "representative_words": words,
        })

    (OUTPUT_DIR / "topic_info.json").write_text(
        json.dumps(topic_info_list, indent=2),
        encoding="utf-8",
    )

    # ── Topics over time ──────────────────────────────────────────────────────
    print("Computing topics over time...")
    timestamps = pd.to_datetime(modelable["start_time"], utc=True).dt.to_pydatetime().tolist()
    try:
        tot = topic_model.topics_over_time(docs, timestamps, nr_bins=60)
        tot_records = tot.to_dict(orient="records")
        # Timestamps aren't JSON-serializable natively
        for r in tot_records:
            if hasattr(r.get("Timestamp"), "isoformat"):
                r["Timestamp"] = r["Timestamp"].isoformat()
        (OUTPUT_DIR / "topic_over_time.json").write_text(
            json.dumps(tot_records, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception as e:
        print(f"  [warn] topics_over_time failed: {e}")

    # ── Interactive visualization ─────────────────────────────────────────────
    print("Generating topic visualization...")
    try:
        viz = topic_model.visualize_topics()
        viz.write_html(str(OUTPUT_DIR / "topic_viz.html"))
        print("  Saved topic_viz.html")
    except Exception as e:
        print(f"  [warn] Visualization failed: {e}")

    # ── Static matplotlib visualizations ─────────────────────────────────────
    print("Generating static visualizations...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1. Topic sizes — horizontal bar chart of top 25 topics
    try:
        top25 = sorted(topic_info_list, key=lambda x: -x["count"])[:25]
        labels = [t["label"].split("_", 1)[1] if "_" in t["label"] else t["label"] for t in top25]
        counts = [t["count"] for t in top25]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(range(len(top25)), counts, color=COLORS[0], edgecolor="white")
        ax.set_yticks(range(len(top25)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Bursts assigned")
        ax.set_title(f"Top 25 Topics by Size  (of {n_topics} total, {n_outliers:,} outliers excluded)")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 3, bar.get_y() + bar.get_height() / 2,
                    str(cnt), va="center", fontsize=8)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "topic_sizes.png", dpi=150)
        plt.close(fig)
        print("  Saved topic_sizes.png")
    except Exception as e:
        print(f"  [warn] topic_sizes.png failed: {e}")

    # 2. Top words grid — 4×3 grid of bar charts for top 12 topics
    try:
        top12 = sorted(topic_info_list, key=lambda x: -x["count"])[:12]
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        fig.suptitle("Top Words per Topic (c-TF-IDF)", fontsize=13, y=1.01)
        for ax, topic in zip(axes.flat, top12):
            words_scores = topic_model.get_topic(topic["topic_id"])
            if not words_scores:
                ax.axis("off")
                continue
            words = [w for w, _ in words_scores[:8]]
            scores = [s for _, s in words_scores[:8]]
            color = COLORS[top12.index(topic) % len(COLORS)]
            ax.barh(range(len(words)), scores, color=color, edgecolor="white")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=8)
            ax.invert_yaxis()
            short_label = topic["label"].split("_", 1)[1] if "_" in topic["label"] else topic["label"]
            ax.set_title(f"{short_label[:30]}  ({topic['count']})", fontsize=8, pad=3)
            ax.tick_params(axis="x", labelsize=7)
        for ax in axes.flat[len(top12):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "topic_words.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved topic_words.png")
    except Exception as e:
        print(f"  [warn] topic_words.png failed: {e}")

    # 3. Topics over time — line chart for top 10 topics
    try:
        tot_path = OUTPUT_DIR / "topic_over_time.json"
        if tot_path.exists():
            tot_records = json.loads(tot_path.read_text(encoding="utf-8"))
            tot_df = pd.DataFrame(tot_records)
            tot_df["Timestamp"] = pd.to_datetime(tot_df["Timestamp"])
            top10_ids = {t["topic_id"] for t in sorted(topic_info_list, key=lambda x: -x["count"])[:10]}
            tot_df = tot_df[tot_df["Topic"].isin(top10_ids)].copy()

            fig, ax = plt.subplots(figsize=(13, 6))
            for i, (tid, grp) in enumerate(tot_df.groupby("Topic")):
                grp = grp.sort_values("Timestamp")
                label_row = next((t for t in topic_info_list if t["topic_id"] == tid), None)
                raw_label = label_row["label"] if label_row else f"topic_{tid}"
                short = raw_label.split("_", 1)[1] if "_" in raw_label else raw_label
                ax.plot(grp["Timestamp"], grp["Frequency"],
                        label=short[:35], linewidth=1.5,
                        color=COLORS[i % len(COLORS)], marker="o", markersize=3)
            ax.set_xlabel("Month")
            ax.set_ylabel("Burst frequency")
            ax.set_title("Top 10 Topics Over Time")
            ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "topics_over_time.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print("  Saved topics_over_time.png")
    except Exception as e:
        print(f"  [warn] topics_over_time.png failed: {e}")

    # ── bertopic_report.json ──────────────────────────────────────────────────
    top_20 = [
        {
            "topic_id": t["topic_id"],
            "label": t["label"],
            "count": t["count"],
            "top_words": t["representative_words"][:5],
        }
        for t in sorted(topic_info_list, key=lambda x: -x["count"])[:20]
    ]

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "min_cluster_size": args.min_cluster_size,
        "total_bursts": len(summaries_df),
        "bursts_modeled": len(modelable),
        "bursts_excluded": excluded,
        "topics_discovered": n_topics,
        "outlier_count": n_outliers,
        "outlier_pct": round(100 * n_outliers / len(docs), 2),
        "top_20_topics": top_20,
    }
    (OUTPUT_DIR / "bertopic_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n--- BERTopic summary ---")
    print(f"  Bursts modeled:    {len(modelable):,}")
    print(f"  Topics discovered: {n_topics}")
    print(f"  Outliers:          {n_outliers:,} ({report['outlier_pct']}%)")
    print(f"\n  Top 10 topics:")
    for t in top_20[:10]:
        print(f"    [{t['topic_id']:>3}] {t['label']:<40} ({t['count']} bursts)")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  burst_topics.parquet   topic assignment per burst")
    print("  topic_model.pkl        serialized model")
    print("  embeddings_summary.npy reusable embeddings (computed on summaries)")
    print("  topic_info.json        topic labels and words")
    print("  topic_over_time.json   temporal topic distribution")
    print("  topic_viz.html         interactive distance map")
    print("  topic_sizes.png        top 25 topics by size")
    print("  topic_words.png        top words per topic (grid)")
    print("  topics_over_time.png   top 10 topics over time")
    print("  bertopic_report.json   run summary")
    print(f"\nNext: python message_miner/09_emoji.py  (or 10_goemotions.py — independent)")


if __name__ == "__main__":
    main()
