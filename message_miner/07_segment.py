"""
07_segment.py — Burst Segmentation

Splits each chat into discrete conversation bursts by time-gap thresholding.
A burst is a contiguous sequence of messages where no consecutive pair is
separated by more than GAP_THRESHOLD_SECONDS. When a gap exceeds the threshold,
a new burst begins.

This creates the natural conversation unit that BERTopic and downstream analysis
operate on — equivalent to what Subtext gets for free from ChatGPT's conversation
structure.

Input filter (before segmentation):
  message_type == "message"   (excludes system events)
  is_deleted == False
  is_empty == False

Reads:
  outputs/05_enrich/messages_enriched.parquet
  outputs/02_extract/chats.parquet

Writes:
  outputs/07_segment/bursts.parquet            one row per burst
  outputs/07_segment/message_burst_map.parquet message_id → burst_id
  outputs/07_segment/segmentation_report.json  summary stats

Run from project root:
    python message_miner/07_segment.py [--threshold SECONDS] [--force]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
CHATS_PATH = Path("outputs/02_extract/chats.parquet")
OUTPUT_DIR = Path("outputs/07_segment")

GAP_THRESHOLD_SECONDS = 1800  # 30 minutes default

# Markdown code fence pattern
_CODE_FENCE_RE = re.compile(r"```|~~~")


# ── Text helpers ──────────────────────────────────────────────────────────────

def _first_name(display_name: str | None) -> str:
    if not display_name:
        return "Unknown"
    return display_name.split()[0]


def _build_text_concatenated(group: pd.DataFrame) -> str | None:
    """Join non-null message texts as 'FirstName: text', one per line."""
    lines = []
    for _, row in group.iterrows():
        if pd.isna(row["text"]) or not str(row["text"]).strip():
            continue
        name = _first_name(row.get("sender_name"))
        lines.append(f"{name}: {row['text'].strip()}")
    return "\n".join(lines) if lines else None


# ── Burst-level aggregation ───────────────────────────────────────────────────

def _aggregate_burst(group: pd.DataFrame, chat_type_map: dict) -> dict:
    """Compute all burst-level columns from a group of messages sharing a burst_id."""
    burst_id = group["burst_id"].iloc[0]
    chat_id = group["chat_id"].iloc[0]
    burst_index = int(group["_burst_index"].iloc[0])

    timestamps = group["timestamp"].dropna()
    start_time = timestamps.min() if not timestamps.empty else pd.NaT
    end_time = timestamps.max() if not timestamps.empty else pd.NaT
    duration_seconds = (
        (end_time - start_time).total_seconds()
        if pd.notna(start_time) and pd.notna(end_time)
        else None
    )

    senders = group["sender_name"].dropna().unique().tolist()
    sender_ids = group["sender_id"].dropna().unique().tolist()

    self_msgs = int(group["is_self"].sum())
    msg_count = len(group)

    word_count_total = group["word_count"].dropna().sum()

    text_concat = _build_text_concatenated(group)

    has_code = bool(
        group["text"]
        .dropna()
        .apply(lambda t: bool(_CODE_FENCE_RE.search(t)))
        .any()
    )
    has_attachments = bool(group["has_attachments"].any())

    return {
        "burst_id": burst_id,
        "chat_id": chat_id,
        "chat_type": chat_type_map.get(chat_id),
        "burst_index": burst_index,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "msg_count": msg_count,
        "participant_count": len(sender_ids),
        "participants": ",".join(senders) if senders else None,
        "self_msg_count": self_msgs,
        "self_msg_ratio": float(self_msgs / msg_count) if msg_count else None,
        "text_concatenated": text_concat,
        "word_count": int(word_count_total) if word_count_total else 0,
        "has_code": has_code,
        "has_attachments": has_attachments,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _build_report(
    bursts_df: pd.DataFrame,
    threshold: int,
    n_input_messages: int,
    n_filtered_messages: int,
) -> dict:
    def _dist(s: pd.Series) -> dict:
        if s.empty:
            return {}
        desc = s.describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        return {k: round(float(v), 2) for k, v in desc.items()}

    bursts_per_chat = bursts_df.groupby("chat_id").size()
    single_msg = int((bursts_df["msg_count"] == 1).sum())
    total_bursts = len(bursts_df)

    by_type = bursts_df["chat_type"].value_counts().to_dict()

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "gap_threshold_seconds": threshold,
        "input_messages": n_input_messages,
        "filtered_messages": n_filtered_messages,
        "total_bursts": total_bursts,
        "single_message_bursts": single_msg,
        "single_message_burst_pct": round(100 * single_msg / total_bursts, 2) if total_bursts else 0,
        "by_chat_type": {str(k): int(v) for k, v in by_type.items()},
        "burst_length_distribution": _dist(bursts_df["msg_count"].astype(float)),
        "burst_duration_distribution": _dist(bursts_df["duration_seconds"].dropna()),
        "bursts_per_chat_distribution": _dist(bursts_per_chat.astype(float)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Segment chats into conversation bursts")
    parser.add_argument(
        "--threshold", type=int, default=GAP_THRESHOLD_SECONDS,
        help=f"Gap threshold in seconds to start a new burst (default: {GAP_THRESHOLD_SECONDS})"
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "bursts.parquet").exists() and not args.force:
        answer = input("outputs/07_segment/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 07_segment ===\n")
    print(f"Gap threshold: {args.threshold}s ({args.threshold / 60:.0f} min)\n")

    for path in (MESSAGES_PATH, CHATS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(MESSAGES_PATH)
    chats_df = pd.read_parquet(CHATS_PATH)
    n_input = len(df)
    print(f"Loaded {n_input:,} messages.\n")

    # Ensure timestamp is tz-aware
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ── Filter ────────────────────────────────────────────────────────────────
    mask = (
        (df["message_type"] == "message") &
        (df["is_deleted"] == False) &
        (df["is_empty"] == False)
    )
    df = df[mask].copy()
    n_filtered = len(df)
    print(f"After filtering (message, not deleted, not empty): {n_filtered:,} messages.\n")

    # ── Sort ──────────────────────────────────────────────────────────────────
    df = df.sort_values(["chat_id", "timestamp"]).reset_index(drop=True)

    # ── Burst assignment ──────────────────────────────────────────────────────
    is_new_chat = df["chat_id"] != df["chat_id"].shift(1)
    is_gap = df["time_delta_seconds"] > args.threshold
    is_burst_start = is_new_chat | is_gap

    df["_burst_start"] = is_burst_start.astype(int)
    df["_burst_index"] = df.groupby("chat_id", sort=False)["_burst_start"].cumsum() - 1
    df["burst_id"] = df["chat_id"] + "__burst_" + df["_burst_index"].astype(str)

    total_bursts = df["burst_id"].nunique()
    print(f"Identified {total_bursts:,} bursts across {df['chat_id'].nunique():,} chats.\n")

    # ── message_burst_map ─────────────────────────────────────────────────────
    burst_map = df[["message_id", "burst_id"]].copy()
    burst_map.to_parquet(OUTPUT_DIR / "message_burst_map.parquet", index=False)

    # ── Build bursts.parquet ──────────────────────────────────────────────────
    chat_type_map = chats_df.set_index("chat_id")["chat_type"].astype(str).to_dict()

    print("Aggregating burst records...")
    burst_records = []
    for burst_id, group in df.groupby("burst_id", sort=False):
        burst_records.append(_aggregate_burst(group, chat_type_map))

    bursts_df = pd.DataFrame(burst_records)

    # Apply dtypes
    bursts_df["burst_index"] = bursts_df["burst_index"].astype("int32")
    bursts_df["msg_count"] = bursts_df["msg_count"].astype("int32")
    bursts_df["participant_count"] = bursts_df["participant_count"].astype("int32")
    bursts_df["self_msg_count"] = bursts_df["self_msg_count"].astype("int32")
    bursts_df["word_count"] = bursts_df["word_count"].astype("int32")
    bursts_df["self_msg_ratio"] = bursts_df["self_msg_ratio"].astype("float32")
    bursts_df["chat_type"] = bursts_df["chat_type"].astype("category")
    for col in ("start_time", "end_time"):
        bursts_df[col] = pd.to_datetime(bursts_df[col], utc=True, errors="coerce")

    bursts_df.to_parquet(OUTPUT_DIR / "bursts.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    report = _build_report(bursts_df, args.threshold, n_input, n_filtered)
    (OUTPUT_DIR / "segmentation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    bl = report["burst_length_distribution"]
    bd = report["burst_duration_distribution"]

    print(f"\n--- Segmentation summary ---")
    print(f"  Input messages:       {n_input:,}")
    print(f"  Filtered messages:    {n_filtered:,}")
    print(f"  Total bursts:         {total_bursts:,}")
    print(f"  Single-msg bursts:    {report['single_message_bursts']:,}  ({report['single_message_burst_pct']}%)")

    print(f"\n  Burst length (messages):")
    print(f"    median  {bl.get('50%', 0):.0f}")
    print(f"    p95     {bl.get('95%', 0):.0f}")
    print(f"    max     {bl.get('max', 0):.0f}")

    print(f"\n  Burst duration (seconds):")
    print(f"    median  {bd.get('50%', 0):.0f}s  ({bd.get('50%', 0)/60:.1f} min)")
    print(f"    p95     {bd.get('95%', 0):.0f}s  ({bd.get('95%', 0)/60:.1f} min)")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  bursts.parquet            one row per burst")
    print("  message_burst_map.parquet message_id to burst_id")
    print("  segmentation_report.json  summary stats")
    print(f"\nNext: python message_miner/08_bertopic.py")


if __name__ == "__main__":
    main()
