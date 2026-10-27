"""
10_goemotions.py — GoEmotions Emotion Classification

Classifies each message across 28 emotion categories using
SamLowe/roberta-base-go_emotions. Independent of 07/08 — reads directly
from messages_enriched.parquet.

Filters to messages with word_count >= 3, not deleted, message_type == "message".
Runs on CPU by default; auto-uses GPU if available.

Expect ~60-90 minutes on CPU for 100k messages. The model (~500MB) is cached
by HuggingFace after the first download.

Reads:
  outputs/05_enrich/messages_enriched.parquet

Writes:
  outputs/10_goemotions/messages_emotions.parquet   per-message emotion labels
  outputs/10_goemotions/emotion_summary.parquet     aggregate emotion stats
  outputs/10_goemotions/figures/                    visualizations
  outputs/10_goemotions/emotion_report.json         run summary

Run from project root:
    python message_miner/10_goemotions.py [--force] [--batch-size N]
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
OUTPUT_DIR    = Path("outputs/10_goemotions")
FIGURES_DIR   = OUTPUT_DIR / "figures"

MODEL_NAME       = "SamLowe/roberta-base-go_emotions"
MIN_WORD_COUNT   = 3
DETECT_THRESHOLD = 0.5
DEFAULT_BATCH    = 128


# ── Figures ───────────────────────────────────────────────────────────────────

def _make_figures(
    emotions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    messages_df: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(exist_ok=True)
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # 1. Emotion distribution — all 28 emotions by detection frequency
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        ordered = summary_df.sort_values("times_detected", ascending=True)
        ax.barh(range(len(ordered)), ordered["times_detected"],
                color=COLORS[0], edgecolor="white")
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered["emotion"], fontsize=9)
        ax.set_xlabel("Messages detected (score > 0.5)")
        ax.set_title("GoEmotions Distribution — All 28 Categories")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "emotion_distribution.png", dpi=150)
        plt.close(fig)
        print("  Saved emotion_distribution.png")
    except Exception as e:
        print(f"  [warn] emotion_distribution.png: {e}")

    # 2. Emotion by sender — top 10 senders, top 5 emotions stacked
    try:
        # Join chat_type from messages
        chat_type_map = messages_df.set_index("message_id")["chat_id"].to_dict()
        em = emotions_df.copy()

        top_senders = (
            em.groupby("sender_name")["message_id"].count()
            .nlargest(10).index.tolist()
        )
        top5_emotions = (
            summary_df[summary_df["emotion"] != "neutral"]
            .nlargest(5, "times_top_emotion")["emotion"].tolist()
        )
        pivot_data = []
        for sender in top_senders:
            sender_em = em[em["sender_name"] == sender]
            row = {"sender": sender}
            for emo in top5_emotions:
                row[emo] = int((sender_em["top_emotion"] == emo).sum())
            row["other"] = int(
                (~sender_em["top_emotion"].isin(top5_emotions + ["neutral"])).sum()
            )
            pivot_data.append(row)
        pivot_df = pd.DataFrame(pivot_data).set_index("sender")
        cols = top5_emotions + ["other"]

        fig, ax = plt.subplots(figsize=(13, 6))
        bottom = np.zeros(len(pivot_df))
        for i, col in enumerate(cols):
            vals = pivot_df[col].values
            ax.bar(range(len(pivot_df)), vals, bottom=bottom,
                   label=col, color=COLORS[i % len(COLORS)])
            bottom += vals
        ax.set_xticks(range(len(pivot_df)))
        ax.set_xticklabels(pivot_df.index, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Messages")
        ax.set_title("Top 10 Senders — Emotion Breakdown")
        ax.legend(bbox_to_anchor=(1, 1), fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "emotion_by_sender.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved emotion_by_sender.png")
    except Exception as e:
        print(f"  [warn] emotion_by_sender.png: {e}")

    # 3. Top 5 non-neutral emotions over time (monthly share)
    try:
        ts_map = messages_df.set_index("message_id")["timestamp"]
        em = emotions_df.copy()
        em["timestamp"] = em["message_id"].map(ts_map)
        em["month"] = pd.to_datetime(em["timestamp"]).dt.to_period("M")

        top5 = (
            summary_df[summary_df["emotion"] != "neutral"]
            .nlargest(5, "times_top_emotion")["emotion"].tolist()
        )

        fig, ax = plt.subplots(figsize=(13, 6))
        for i, emo in enumerate(top5):
            monthly = (
                em.groupby("month").apply(
                    lambda g: (g["top_emotion"] == emo).sum() / len(g),
                    include_groups=False,
                )
                .sort_index()
            )
            ax.plot(monthly.index.astype(str), monthly.values * 100,
                    label=emo, linewidth=1.5, marker="o", markersize=3,
                    color=COLORS[i % len(COLORS)])
        ax.set_xlabel("Month")
        ax.set_ylabel("% of messages")
        ax.set_title("Top 5 Non-Neutral Emotions Over Time (monthly share)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "emotion_over_time.png", dpi=150)
        plt.close(fig)
        print("  Saved emotion_over_time.png")
    except Exception as e:
        print(f"  [warn] emotion_over_time.png: {e}")

    # 4. Emotion by chat type
    try:
        chat_type_map = messages_df.set_index("message_id")["chat_id"].to_dict()
        chats_df_path = Path("outputs/02_extract/chats.parquet")
        if chats_df_path.exists():
            chats_df = pd.read_parquet(chats_df_path)
            chat_type_by_id = chats_df.set_index("chat_id")["chat_type"].astype(str).to_dict()
            em = emotions_df.copy()
            em["chat_type"] = em["chat_id"].map(chat_type_by_id).fillna("unknown")

            top5 = (
                summary_df[summary_df["emotion"] != "neutral"]
                .nlargest(5, "times_top_emotion")["emotion"].tolist()
            )
            chat_types = em["chat_type"].unique().tolist()

            pivot = {}
            for ct in chat_types:
                ct_em = em[em["chat_type"] == ct]
                pivot[ct] = {
                    emo: round(100 * (ct_em["top_emotion"] == emo).sum() / max(len(ct_em), 1), 2)
                    for emo in top5
                }
            pivot_df = pd.DataFrame(pivot).T

            x = np.arange(len(pivot_df))
            width = 0.15
            fig, ax = plt.subplots(figsize=(11, 6))
            for i, emo in enumerate(top5):
                ax.bar(x + i * width, pivot_df[emo], width,
                       label=emo, color=COLORS[i % len(COLORS)])
            ax.set_xticks(x + width * 2)
            ax.set_xticklabels(pivot_df.index, fontsize=9)
            ax.set_ylabel("% of messages")
            ax.set_title("Top 5 Emotions by Chat Type")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "emotion_by_chat_type.png", dpi=150)
            plt.close(fig)
            print("  Saved emotion_by_chat_type.png")
    except Exception as e:
        print(f"  [warn] emotion_by_chat_type.png: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="GoEmotions emotion classification")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help=f"Batch size for inference (default: {DEFAULT_BATCH})")
    args = parser.parse_args()

    if (OUTPUT_DIR / "emotion_report.json").exists() and not args.force:
        answer = input("outputs/10_goemotions/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 10_goemotions ===\n")

    if not MESSAGES_PATH.exists():
        print(f"ERROR: {MESSAGES_PATH} not found.")
        sys.exit(1)

    # ── Load and filter ───────────────────────────────────────────────────────
    print("Loading messages...")
    df = pd.read_parquet(MESSAGES_PATH)
    print(f"  {len(df):,} total messages.\n")

    filtered = df[
        (df["message_type"] == "message") &
        (df["is_deleted"] == False) &
        (df["text"].notna()) &
        (df["word_count"] >= MIN_WORD_COUNT)
    ].copy()
    n_filtered = len(filtered)
    print(f"  Eligible for classification (word_count >= {MIN_WORD_COUNT}): {n_filtered:,}\n")

    texts   = filtered["text"].tolist()
    msg_ids = filtered["message_id"].tolist()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading GoEmotions model...")
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"  Device: {'GPU' if device == 0 else 'CPU'}")

    classifier = pipeline(
        task="text-classification",
        model=MODEL_NAME,
        top_k=None,
        device=device,
    )
    print(f"  Model loaded: {MODEL_NAME}\n")

    # ── Classify ──────────────────────────────────────────────────────────────
    print(f"Classifying {n_filtered:,} messages (batch_size={args.batch_size})...")
    print("  This may take 60-90 minutes on CPU.\n")

    from tqdm import tqdm

    all_results = []
    for i in tqdm(range(0, len(texts), args.batch_size), desc="GoEmotions", unit="batch"):
        batch = texts[i: i + args.batch_size]
        # Truncate to 512 chars to stay within model token limit
        batch = [t[:512] for t in batch]
        try:
            batch_results = classifier(batch)
            all_results.extend(batch_results)
        except Exception as e:
            print(f"\n  [warn] Batch {i // args.batch_size} failed: {e} — filling with neutral")
            for _ in batch:
                all_results.append([{"label": "neutral", "score": 1.0}])

    print(f"\n  Classification complete. Building output tables...")

    # ── Build messages_emotions ───────────────────────────────────────────────
    meta_map = filtered.set_index("message_id")[
        ["chat_id", "sender_name", "is_self", "timestamp"]
    ].to_dict(orient="index")

    emotion_rows = []
    for msg_id, scores in zip(msg_ids, all_results):
        sorted_scores = sorted(scores, key=lambda x: -x["score"])
        top = sorted_scores[0]
        top3 = [{"label": s["label"], "score": round(s["score"], 4)} for s in sorted_scores[:3]]
        detected = [s["label"] for s in sorted_scores if s["score"] >= DETECT_THRESHOLD]
        meta = meta_map.get(msg_id, {})

        emotion_rows.append({
            "message_id":       msg_id,
            "chat_id":          meta.get("chat_id"),
            "sender_name":      meta.get("sender_name"),
            "is_self":          bool(meta.get("is_self", False)),
            "timestamp":        meta.get("timestamp"),
            "top_emotion":      top["label"],
            "top_emotion_score": round(float(top["score"]), 4),
            "top_3_emotions":   json.dumps(top3),
            "is_neutral":       top["label"] == "neutral",
            "detected_emotions": json.dumps(detected),
            "detected_count":   len(detected),
        })

    emotions_df = pd.DataFrame(emotion_rows)
    emotions_df["top_emotion_score"] = emotions_df["top_emotion_score"].astype("float32")
    emotions_df["detected_count"]    = emotions_df["detected_count"].astype("int32")
    emotions_df.to_parquet(OUTPUT_DIR / "messages_emotions.parquet", index=False)

    # ── emotion_summary.parquet ───────────────────────────────────────────────
    total = len(emotions_df)
    all_emotions = sorted({s["label"] for batch in all_results for s in batch})

    summary_rows = []
    for emo in all_emotions:
        times_top  = int((emotions_df["top_emotion"] == emo).sum())
        times_det  = int(emotions_df["detected_emotions"].apply(
            lambda x: emo in json.loads(x)
        ).sum())
        self_det   = int(emotions_df[emotions_df["is_self"]]["detected_emotions"].apply(
            lambda x: emo in json.loads(x)
        ).sum())
        other_det  = int(emotions_df[~emotions_df["is_self"]]["detected_emotions"].apply(
            lambda x: emo in json.loads(x)
        ).sum())
        n_self  = int(emotions_df["is_self"].sum())
        n_other = total - n_self

        summary_rows.append({
            "emotion":         emo,
            "times_top_emotion": times_top,
            "times_detected":  times_det,
            "pct_of_messages": round(100 * times_det / total, 2) if total else 0,
            "self_pct":        round(100 * self_det / n_self, 2) if n_self else 0,
            "other_pct":       round(100 * other_det / n_other, 2) if n_other else 0,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("times_detected", ascending=False)
    summary_df["times_top_emotion"] = summary_df["times_top_emotion"].astype("int32")
    summary_df["times_detected"]    = summary_df["times_detected"].astype("int32")
    summary_df["pct_of_messages"]   = summary_df["pct_of_messages"].astype("float32")
    summary_df["self_pct"]          = summary_df["self_pct"].astype("float32")
    summary_df["other_pct"]         = summary_df["other_pct"].astype("float32")
    summary_df.to_parquet(OUTPUT_DIR / "emotion_summary.parquet", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures...")
    _make_figures(emotions_df, summary_df, filtered)
    print()

    # ── Report ────────────────────────────────────────────────────────────────
    neutral_pct = round(100 * emotions_df["is_neutral"].sum() / total, 2)
    top10 = summary_df.head(10)[["emotion", "times_top_emotion", "times_detected", "pct_of_messages"]].to_dict(orient="records")

    report = {
        "run_at":              datetime.now(timezone.utc).isoformat(),
        "model":               MODEL_NAME,
        "total_messages":      len(df),
        "messages_classified": total,
        "messages_excluded":   len(df) - total,
        "neutral_count":       int(emotions_df["is_neutral"].sum()),
        "neutral_pct":         neutral_pct,
        "detect_threshold":    DETECT_THRESHOLD,
        "top_10_emotions":     top10,
    }
    (OUTPUT_DIR / "emotion_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("--- GoEmotions summary ---")
    print(f"  Messages classified: {total:,}")
    print(f"  Neutral:             {report['neutral_count']:,} ({neutral_pct}%)")
    print(f"\n  Top 10 detected emotions:")
    for row in top10:
        print(f"    {row['emotion']:<20} {row['pct_of_messages']:>5.1f}%  ({row['times_detected']:,} msgs)")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_emotions.parquet  per-message emotion labels")
    print("  emotion_summary.parquet    aggregate emotion stats")
    print("  figures/                   4 charts")
    print("  emotion_report.json        run summary")


if __name__ == "__main__":
    main()
