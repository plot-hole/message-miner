"""
validation.py — Boundary Detection Quality Framework

Measures the quality of conversation boundaries produced by 07b_segment_llm.py
without requiring manual review of every conversation.

Reads:
  outputs/07_segment/bursts.parquet
  outputs/05_enrich/messages_enriched.parquet
  outputs/07_segment/message_burst_map.parquet

Writes:
  outputs/07_segment/validation_report.json

Run from project root:
    python message_miner/validation.py [--golden golden_set.json] [--sample 50] [--no-judge]

Validation checks:
  1. Statistical signature analysis  — per-conversation quality signals
  2. Population outlier flagging     — under/over-split, interleaved, monologues
  3. LLM-as-judge evaluation         — strong model reviews sampled conversations
  4. Golden set evaluation           — precision/recall/F1 vs. human-labeled boundaries
"""

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from message_miner.helpers.async_llm import async_call_api

BURSTS_PATH = Path("outputs/07_segment/bursts.parquet")
BURST_MAP_PATH = Path("outputs/07_segment/message_burst_map.parquet")
MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
OUTPUT_PATH = Path("outputs/07_segment/validation_report.json")

JUDGE_MODEL = "gpt-5.4"
EMBEDDING_MODEL = "text-embedding-3-small"
TEMPERATURE = 0


# ── Embedding helpers ─────────────────────────────────────────────────────────

async def _get_embeddings_batch(client, texts: list[str]) -> list[list[float]] | None:
    """Fetch embeddings for a batch of texts. Returns None on failure."""
    try:
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"  [warn] Embedding API failed: {e}. Falling back to vocabulary coherence.")
        return None


def _vocabulary_coherence(texts: list[str]) -> float:
    """
    Fallback coherence measure when embeddings are unavailable.
    Average Jaccard similarity between all pairs of message token sets.
    """
    token_sets = [set(t.lower().split()) for t in texts if t.strip()]
    if len(token_sets) < 2:
        return 1.0
    scores = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            union = len(a | b)
            if union > 0:
                scores.append(len(a & b) / union)
    return mean(scores) if scores else 0.0


def _cosine_distance(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    norm = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(1.0 - np.dot(va, vb) / norm) if norm > 0 else 1.0


# ── Statistical signature analysis ───────────────────────────────────────────

def compute_conversation_signatures(
    burst: dict,
    embeddings: list[list[float]] | None = None,
) -> dict:
    """
    Compute statistical quality signals for a single detected conversation.

    burst dict expects: msg_count, word_count, participant_count, duration_seconds,
    messages (list of {"sender": str, "text": str}).

    embeddings: pre-computed per-message vectors; falls back to vocabulary overlap.
    """
    messages = burst.get("messages", [])
    msg_count = burst.get("msg_count", len(messages))
    unique_participants = burst.get("participant_count", 1)
    duration_minutes = (burst.get("duration_seconds") or 0) / 60

    # Participant consistency — messages per participant (higher = more focused)
    messages_per_participant = msg_count / unique_participants if unique_participants > 0 else 0

    # Turn-taking ratio — % of consecutive pairs from different senders
    senders = [m.get("sender") for m in messages]
    turns = sum(
        1 for i in range(1, len(senders))
        if senders[i] and senders[i - 1] and senders[i] != senders[i - 1]
    )
    turn_ratio = turns / (len(senders) - 1) if len(senders) > 1 else 0.0

    # Embedding coherence (or vocabulary fallback)
    if embeddings and len(embeddings) >= 2:
        centroid = list(np.mean(embeddings, axis=0))
        distances = [_cosine_distance(e, centroid) for e in embeddings]
        coherence = float(1.0 - np.mean(distances))
    else:
        texts = [m.get("text", "") for m in messages if m.get("text")]
        coherence = _vocabulary_coherence(texts) if texts else 1.0

    return {
        "participant_consistency": round(messages_per_participant, 2),
        "turn_ratio": round(turn_ratio, 3),
        "embedding_coherence": round(coherence, 3),
        "duration_minutes": round(duration_minutes, 1),
        "message_count": msg_count,
        "word_count": burst.get("word_count", 0),
        "unique_participants": unique_participants,
    }


def analyze_boundary_quality(all_signatures: list[dict]) -> tuple[dict, dict]:
    """
    Compute population-level distributions and flag outliers.

    Returns (distributions, outliers) where outliers maps category → list of indices.
    """
    outliers: dict[str, list[int]] = {
        "possibly_under_split": [],
        "possibly_over_split": [],
        "possibly_interleaved": [],
        "monologues": [],
    }

    for i, sig in enumerate(all_signatures):
        if sig["message_count"] > 100 and sig["embedding_coherence"] < 0.5:
            outliers["possibly_under_split"].append(i)
        if sig["message_count"] <= 3 and sig["duration_minutes"] < 1:
            outliers["possibly_over_split"].append(i)
        if sig["unique_participants"] > 4 and sig["turn_ratio"] < 0.3:
            outliers["possibly_interleaved"].append(i)
        if sig["turn_ratio"] < 0.1 and sig["message_count"] > 5:
            outliers["monologues"].append(i)

    def _p95(values):
        s = sorted(values)
        return round(s[int(len(s) * 0.95)], 3) if s else None

    msg_counts = [s["message_count"] for s in all_signatures]
    coherences = [s["embedding_coherence"] for s in all_signatures]
    turn_ratios = [s["turn_ratio"] for s in all_signatures]
    durations = [s["duration_minutes"] for s in all_signatures]

    distributions = {
        "message_count": {
            "median": round(median(msg_counts), 1) if msg_counts else None,
            "mean": round(mean(msg_counts), 1) if msg_counts else None,
            "p95": _p95(msg_counts),
            "max": max(msg_counts) if msg_counts else None,
        },
        "embedding_coherence": {
            "median": round(median(coherences), 3) if coherences else None,
            "mean": round(mean(coherences), 3) if coherences else None,
        },
        "turn_ratio": {
            "median": round(median(turn_ratios), 3) if turn_ratios else None,
            "mean": round(mean(turn_ratios), 3) if turn_ratios else None,
        },
        "duration_minutes": {
            "median": round(median(durations), 1) if durations else None,
            "mean": round(mean(durations), 1) if durations else None,
            "p95": _p95(durations),
        },
    }

    return distributions, outliers


# ── LLM-as-judge evaluation ───────────────────────────────────────────────────

async def _judge_one(client, conv: dict) -> dict:
    messages_text = "\n".join(
        f"[{i}] {(m.get('sender') or 'Unknown').split()[0]}: {str(m.get('text', '')).strip()[:300]}"
        for i, m in enumerate(conv.get("messages", []))
    )
    prompt = f"""Here is a set of messages that were identified as a single conversation.

{messages_text}

Evaluate this as a bounded conversation unit:

1. COHERENCE (1-5): Do these messages belong together as a single conversation?
   5 = perfectly coherent, 1 = clearly multiple unrelated conversations merged.

2. BOUNDARIES (1-5): Are the start and end points correct?
   5 = starts and ends at natural points, 1 = starts or ends mid-exchange.

3. SPLITS: Message index(es) where you would split this. Empty list if none needed.

4. MERGES: Does this appear cut off — should it continue with messages not included? (true/false)

Return JSON: {{"coherence": N, "boundaries": N, "splits": [], "should_merge": true_or_false, "notes": "brief explanation"}}"""

    try:
        content, _ = await async_call_api(
            client,
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        result = json.loads(content)
        result["burst_id"] = conv.get("burst_id")
        return result
    except Exception as e:
        return {
            "burst_id": conv.get("burst_id"),
            "coherence": None,
            "boundaries": None,
            "splits": [],
            "should_merge": None,
            "notes": f"Error: {e}",
        }


async def llm_judge_evaluation(
    conversations: list[dict],
    api_key: str,
    sample_size: int = 50,
) -> dict:
    """
    Sample N conversations and ask a strong model to score them.

    Targets: coherence > 4.0, boundaries > 4.0, split_rate < 15%, merge_rate < 15%.
    """
    from openai import AsyncOpenAI

    sample = random.sample(conversations, min(sample_size, len(conversations)))
    print(f"  Running LLM judge on {len(sample)} conversations...")

    client = AsyncOpenAI(api_key=api_key, timeout=120.0)
    semaphore = asyncio.Semaphore(5)

    async def _rate_limited(conv):
        async with semaphore:
            return await _judge_one(client, conv)

    results = await asyncio.gather(*[_rate_limited(c) for c in sample])
    await client.close()

    scored = [r for r in results if r.get("coherence") is not None]
    if not scored:
        return {"error": "All judge calls failed", "detailed_results": list(results)}

    avg_coherence = mean(r["coherence"] for r in scored)
    avg_boundaries = mean(r["boundaries"] for r in scored)
    split_rate = sum(1 for r in scored if r.get("splits")) / len(scored)
    merge_rate = sum(1 for r in scored if r.get("should_merge")) / len(scored)

    return {
        "sample_size": len(scored),
        "avg_coherence": round(avg_coherence, 2),
        "avg_boundaries": round(avg_boundaries, 2),
        "split_rate": round(split_rate, 3),
        "merge_rate": round(merge_rate, 3),
        "meets_targets": {
            "coherence": avg_coherence >= 4.0,
            "boundaries": avg_boundaries >= 4.0,
            "split_rate": split_rate < 0.15,
            "merge_rate": merge_rate < 0.15,
        },
        "detailed_results": list(results),
    }


# ── Golden set evaluation ─────────────────────────────────────────────────────

def evaluate_against_golden_set(
    detected_boundaries: list[int],
    golden_boundaries: list[int],
    tolerance: int = 2,
) -> dict:
    """
    Compare detected boundaries against human-labeled ground truth.

    tolerance: slack in message positions — detected index 45 matches golden 43–47.
    Returns precision, recall, F1. Target: F1 > 0.8.
    """
    true_positives = 0
    matched_golden: set[int] = set()

    for detected in detected_boundaries:
        for golden in golden_boundaries:
            if abs(detected - golden) <= tolerance and golden not in matched_golden:
                true_positives += 1
                matched_golden.add(golden)
                break

    precision = true_positives / len(detected_boundaries) if detected_boundaries else 0.0
    recall = true_positives / len(golden_boundaries) if golden_boundaries else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positives": true_positives,
        "false_positives": len(detected_boundaries) - true_positives,
        "false_negatives": len(golden_boundaries) - true_positives,
        "meets_target": f1 >= 0.8,
    }


# ── Data loading ──────────────────────────────────────────────────────────────

def load_conversations(
    bursts_df: pd.DataFrame,
    messages_df: pd.DataFrame,
    burst_map_df: pd.DataFrame,
) -> list[dict]:
    """Reconstruct conversation dicts from parquet outputs."""
    msg_lookup = messages_df.set_index("message_id")[["sender_name", "text", "timestamp"]].to_dict("index")

    burst_messages: dict[str, list[dict]] = {}
    for _, row in burst_map_df.iterrows():
        bid = row["burst_id"]
        msg = msg_lookup.get(row["message_id"], {})
        burst_messages.setdefault(bid, []).append({
            "message_id": row["message_id"],
            "sender": msg.get("sender_name"),
            "text": msg.get("text"),
            "timestamp": msg.get("timestamp"),
        })

    conversations = []
    for _, burst_row in bursts_df.iterrows():
        bid = burst_row["burst_id"]
        conversations.append({
            "burst_id": bid,
            "chat_id": burst_row.get("chat_id"),
            "msg_count": int(burst_row.get("msg_count", 0)),
            "word_count": int(burst_row.get("word_count", 0)),
            "participant_count": int(burst_row.get("participant_count", 1)),
            "duration_seconds": burst_row.get("duration_seconds"),
            "messages": burst_messages.get(bid, []),
        })
    return conversations


def extract_boundary_indices(conversations: list[dict]) -> list[int]:
    """Return boundary positions (first message of each conversation) relative to channel order."""
    boundaries, pos = [], 0
    for conv in conversations:
        boundaries.append(pos)
        pos += conv["msg_count"]
    return boundaries


# ── Quality report ────────────────────────────────────────────────────────────

async def generate_quality_report(
    conversations: list[dict],
    api_key: str,
    golden_set: dict | None = None,
    judge_sample_size: int = 50,
    run_judge: bool = True,
) -> dict:
    """Run all validation checks and return a report dict."""
    from openai import AsyncOpenAI

    report: dict = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_conversations": len(conversations),
    }

    msg_counts = [c["msg_count"] for c in conversations]
    durations = [c["duration_seconds"] for c in conversations if c["duration_seconds"] is not None]
    report["avg_message_count"] = round(mean(msg_counts), 1) if msg_counts else 0
    report["median_message_count"] = round(median(msg_counts), 1) if msg_counts else 0
    report["avg_duration_minutes"] = round(mean(durations) / 60, 1) if durations else 0

    # ── Statistical signatures ────────────────────────────────────────────────
    print(f"  Computing statistical signatures for {len(conversations)} conversations...")

    # Try to get embeddings; fall back to vocabulary coherence
    embeddings_map: dict[str, list[list[float]]] = {}
    try:
        emb_client = AsyncOpenAI(api_key=api_key, timeout=120.0)
        emb_sample = random.sample(conversations, min(500, len(conversations)))
        semaphore = asyncio.Semaphore(5)

        async def _embed_conv(conv):
            texts = [m.get("text", "") for m in conv["messages"] if m.get("text")]
            if len(texts) < 2:
                return conv["burst_id"], None
            async with semaphore:
                vecs = await _get_embeddings_batch(emb_client, texts[:20])
            return conv["burst_id"], vecs

        embed_results = await asyncio.gather(*[_embed_conv(c) for c in emb_sample])
        await emb_client.close()
        for bid, vecs in embed_results:
            if vecs:
                embeddings_map[bid] = vecs
    except Exception as e:
        print(f"  [warn] Could not compute embeddings: {e}. Using vocabulary fallback.")

    signatures = [
        compute_conversation_signatures(conv, embeddings=embeddings_map.get(conv["burst_id"]))
        for conv in conversations
    ]

    distributions, outliers = analyze_boundary_quality(signatures)
    report["distributions"] = distributions
    report["outliers"] = {k: len(v) for k, v in outliers.items()}
    report["avg_embedding_coherence"] = round(mean(s["embedding_coherence"] for s in signatures), 3) if signatures else None
    report["avg_turn_ratio"] = round(mean(s["turn_ratio"] for s in signatures), 3) if signatures else None

    # ── LLM judge ────────────────────────────────────────────────────────────
    if run_judge:
        judge_results = await llm_judge_evaluation(conversations, api_key, judge_sample_size)
        report["judge"] = {k: v for k, v in judge_results.items() if k != "detailed_results"}
        report["judge_detailed"] = judge_results.get("detailed_results", [])
    else:
        report["judge"] = None

    # ── Golden set ────────────────────────────────────────────────────────────
    if golden_set:
        report["golden_set_results"] = {}
        for channel_id, golden in golden_set.items():
            channel_convs = [c for c in conversations if c["chat_id"] == channel_id]
            if not channel_convs:
                print(f"  [warn] No conversations for golden set channel {channel_id}")
                continue
            detected = extract_boundary_indices(channel_convs)
            gs_result = evaluate_against_golden_set(detected, golden["boundaries"])
            report["golden_set_results"][channel_id] = gs_result
            print(f"  Golden [{channel_id}]: F1={gs_result['f1']:.3f}  P={gs_result['precision']:.3f}  R={gs_result['recall']:.3f}")

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = []
    if outliers["possibly_under_split"]:
        recommendations.append(
            f"{len(outliers['possibly_under_split'])} long low-coherence conversations — possible under-splitting."
        )
    if outliers["possibly_over_split"]:
        recommendations.append(
            f"{len(outliers['possibly_over_split'])} very short conversations (<3 msgs, <1 min) — possible over-splitting."
        )
    if run_judge and report.get("judge"):
        j = report["judge"]
        if j.get("split_rate", 0) >= 0.15:
            recommendations.append(f"Judge split_rate={j['split_rate']:.1%} (target <15%) — under-splitting.")
        if j.get("merge_rate", 0) >= 0.15:
            recommendations.append(f"Judge merge_rate={j['merge_rate']:.1%} (target <15%) — over-splitting.")
        if j.get("avg_coherence", 5) < 4.0:
            recommendations.append(f"Judge avg_coherence={j['avg_coherence']:.2f} (target ≥4.0) — topic mixing.")
    report["recommendations"] = recommendations

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary detection quality framework")
    parser.add_argument("--golden", type=str, default=None,
                        help="Path to golden set JSON for precision/recall evaluation")
    parser.add_argument("--sample", type=int, default=50,
                        help="Conversations to sample for LLM judge (default: 50)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge evaluation (saves API cost)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    for path in (BURSTS_PATH, BURST_MAP_PATH, MESSAGES_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run 07b_segment_llm.py first.")
            sys.exit(1)

    print("=== message-miner: validation ===\n")
    bursts_df = pd.read_parquet(BURSTS_PATH)
    burst_map_df = pd.read_parquet(BURST_MAP_PATH)
    messages_df = pd.read_parquet(MESSAGES_PATH)
    print(f"  {len(bursts_df):,} bursts  |  {len(messages_df):,} messages\n")

    conversations = load_conversations(bursts_df, messages_df, burst_map_df)

    golden_set = None
    if args.golden:
        golden_path = Path(args.golden)
        if not golden_path.exists():
            print(f"ERROR: Golden set file not found: {args.golden}")
            sys.exit(1)
        golden_set = json.loads(golden_path.read_text(encoding="utf-8"))
        print(f"  Golden set: {len(golden_set)} channels\n")

    report = asyncio.run(
        generate_quality_report(
            conversations,
            api_key=api_key,
            golden_set=golden_set,
            judge_sample_size=args.sample,
            run_judge=not args.no_judge,
        )
    )

    # Write report — judge details go to a separate file to keep the main report readable
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    detailed = report.pop("judge_detailed", [])
    if detailed:
        detail_path = OUTPUT_PATH.parent / "validation_judge_details.json"
        detail_path.write_text(json.dumps(detailed, indent=2, default=str), encoding="utf-8")

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n--- Validation Summary ---")
    print(f"  Total conversations:  {report['total_conversations']:,}")
    print(f"  Avg message count:    {report['avg_message_count']}")
    print(f"  Avg duration:         {report['avg_duration_minutes']} min")
    print(f"\n  Outliers:")
    for k, v in report["outliers"].items():
        print(f"    {k:<30} {v}")

    if report.get("judge"):
        j = report["judge"]
        print(f"\n  LLM Judge (n={j.get('sample_size')}):")
        print(f"    Coherence:     {j.get('avg_coherence'):.2f}  (target ≥4.0)")
        print(f"    Boundaries:    {j.get('avg_boundaries'):.2f}  (target ≥4.0)")
        print(f"    Split rate:    {j.get('split_rate'):.1%}  (target <15%)")
        print(f"    Merge rate:    {j.get('merge_rate'):.1%}  (target <15%)")

    if report.get("golden_set_results"):
        print(f"\n  Golden set:")
        for cid, gs in report["golden_set_results"].items():
            print(f"    {cid}: F1={gs['f1']:.3f}  P={gs['precision']:.3f}  R={gs['recall']:.3f}")

    if report.get("recommendations"):
        print(f"\n  Recommendations:")
        for rec in report["recommendations"]:
            print(f"    • {rec}")

    print(f"\nReport written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
