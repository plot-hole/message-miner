"""
27_segment_mail.py — Email Thread Segmentation

Takes enriched email data and produces clean conversation threads suitable
for summarization and classification downstream.

Reads:
  outputs/23_enrich_mail/messages_mail_enriched.parquet
  outputs/22_edge_cases_mail/messages_mail_flagged.parquet
  outputs/20_extract_mail/extraction_report.json

Writes:
  outputs/27_segment_mail/
    email_threads.parquet       -- one row per thread
    message_thread_map.parquet  -- maps each message to its thread
    segmentation_report.json    -- stats

Run from project root:
    python mail-miner/27_segment_mail.py [--force]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ENRICHED_PATH = Path("outputs/23_enrich_mail/messages_mail_enriched.parquet")
FLAGGED_PATH = Path("outputs/22_edge_cases_mail/messages_mail_flagged.parquet")
EXTRACTION_REPORT_PATH = Path("outputs/20_extract_mail/extraction_report.json")
OUTPUT_DIR = Path("outputs/27_segment_mail")

_GAP_THRESHOLD_S = 72 * 3600   # 3 days
_SIZE_CAP = 100                 # max messages before hard split
_CHUNK_SIZE = 50
_CHUNK_OVERLAP = 5
_MSG_WORD_LIMIT = 500
_THREAD_WORD_LIMIT = 3000
_THREAD_HALF = 1500


# ── Text helpers ──────────────────────────────────────────────────────────────

def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " [truncated]"


def _build_thread_text(messages: pd.DataFrame, subject: str | None) -> str:
    parts = []
    first = True
    for _, row in messages.iterrows():
        name = (row.get("sender_name") or row.get("sender_email") or "Unknown").strip()
        dt = row.get("received_datetime")
        if pd.notna(dt) and hasattr(dt, "strftime"):
            ts = dt.strftime("%Y-%m-%d %H:%M")
        else:
            ts = "unknown time"

        header = f"From: {name} ({ts})"
        if first and subject:
            header += f"\nSubject: {subject}"
            first = False

        body = (row.get("body_text") or "").strip()
        if not body:
            body = "[Attachment only — no body text]"
        else:
            body = _truncate_words(body, _MSG_WORD_LIMIT)

        parts.append(f"{header}\n\n{body}")

    full_text = "\n\n---\n\n".join(parts)

    # Truncate total to 3000 words
    words = full_text.split()
    if len(words) > _THREAD_WORD_LIMIT:
        first_half = " ".join(words[:_THREAD_HALF])
        last_half = " ".join(words[-_THREAD_HALF:])
        full_text = first_half + "\n\n[... middle truncated ...]\n\n" + last_half

    return full_text


# ── Thread segmentation ───────────────────────────────────────────────────────

def _time_gap_split(group: pd.DataFrame) -> list[pd.DataFrame]:
    """Split a sorted message group on gaps > 72h. Returns list of sub-DataFrames."""
    if len(group) < 2:
        return [group]

    segments = []
    start = 0
    times = group["received_datetime"].values  # numpy datetime64

    for i in range(1, len(group)):
        gap_s = (times[i] - times[i - 1]) / np.timedelta64(1, "s")
        if gap_s > _GAP_THRESHOLD_S:
            segments.append(group.iloc[start:i])
            start = i
    segments.append(group.iloc[start:])
    return segments


def _size_cap_split(segment: pd.DataFrame) -> list[pd.DataFrame]:
    """If segment exceeds 100 messages, chunk into 50 with 5 overlap."""
    if len(segment) <= _SIZE_CAP:
        return [segment]

    chunks = []
    start = 0
    while start < len(segment):
        end = min(start + _CHUNK_SIZE, len(segment))
        chunks.append(segment.iloc[start:end])
        if end == len(segment):
            break
        start = end - _CHUNK_OVERLAP
    return chunks


def _most_recent_name(sub: pd.DataFrame, email: str) -> str:
    """Return the most recently seen display name for an email address."""
    rows = sub[sub["sender_email"].fillna("").str.lower() == email.lower()]
    rows = rows.sort_values("received_datetime", ascending=False)
    for _, r in rows.iterrows():
        name = (r.get("sender_name") or "").strip()
        if name:
            return name
    return email


def _build_thread_row(
    thread_id: str,
    segment: pd.DataFrame,
    original_conv_id: str,
    was_split: bool,
    user_email: str,
) -> dict:
    segment = segment.sort_values("received_datetime").reset_index(drop=True)
    first = segment.iloc[0]
    last = segment.iloc[-1]

    sender_emails = segment["sender_email"].fillna("").str.lower().unique().tolist()
    sender_emails = [e for e in sender_emails if e]

    # Deduplicate by email, use most-recent display name
    name_map = {e: _most_recent_name(segment, e) for e in sender_emails}
    participants_str = ", ".join(name_map.get(e, e) for e in sender_emails)
    participant_emails_json = json.dumps(sender_emails)

    start_time = first["received_datetime"]
    end_time = last["received_datetime"]
    if pd.notna(start_time) and pd.notna(end_time):
        duration_s = (end_time - start_time).total_seconds()
    else:
        duration_s = float("nan")

    ue = user_email.lower()
    is_self_col = segment["is_self"].fillna(False)
    has_self = bool(is_self_col.any())
    self_msg_count = int(is_self_col.sum())
    is_self_initiated = bool(first.get("is_self", False))

    subject = first.get("subject_clean") or first.get("subject") or None

    # Skip logic
    skip = False
    skip_reason = None
    if len(segment) == 1 and self_msg_count == 0:
        skip = True
        skip_reason = "single_inbound"

    # Thread text (only for non-skipped)
    if not skip:
        thread_text = _build_thread_text(segment, subject)
    else:
        thread_text = ""

    word_count = len(thread_text.split()) if thread_text else 0

    return {
        "thread_id": thread_id,
        "subject": subject,
        "msg_count": len(segment),
        "participants": participants_str,
        "participant_count": len(sender_emails),
        "participant_emails": participant_emails_json,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_s,
        "has_self": has_self,
        "self_msg_count": self_msg_count,
        "is_self_initiated": is_self_initiated,
        "was_split": was_split,
        "original_conversation_id": original_conv_id,
        "skip": skip,
        "skip_reason": skip_reason,
        "thread_text": thread_text,
        "word_count": word_count,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Segment email threads")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not args.force:
        answer = input("outputs/27_segment_mail/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 27_segment_mail ===\n")

    for path in (ENRICHED_PATH, FLAGGED_PATH, EXTRACTION_REPORT_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    extraction_report = json.loads(EXTRACTION_REPORT_PATH.read_text(encoding="utf-8"))
    user_email = extraction_report.get("user_email", "")
    user_name = extraction_report.get("user_name", "")
    print(f"User: {user_name} <{user_email}>\n")

    print("Loading enriched data...")
    enriched = pd.read_parquet(ENRICHED_PATH)

    print("Loading flagged data...")
    flagged = pd.read_parquet(FLAGGED_PATH)

    flag_cols = [c for c in flagged.columns if c not in enriched.columns or c == "message_id"]
    df = enriched.merge(flagged[flag_cols], on="message_id", how="left")

    total_loaded = len(df)
    df = df[df["is_human_mail"].fillna(False)].copy()
    print(f"Human mail: {len(df):,} of {total_loaded:,} messages\n")

    # Ensure tz-aware datetimes
    for col in ("received_datetime", "sent_datetime"):
        if col in df.columns and df[col].dt.tz is None:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    raw_conv_ids = df["conversation_id"].nunique()
    print(f"Processing {raw_conv_ids:,} conversation groups...")

    thread_rows = []
    map_rows = []   # {message_id, thread_id, position_in_thread}

    threads_split_time = 0
    threads_split_size = 0

    for conv_id, group in df.sort_values("received_datetime").groupby("conversation_id", sort=False):
        group = group.sort_values("received_datetime").reset_index(drop=True)

        # Step 1: time-gap split
        time_segments = _time_gap_split(group)
        was_time_split = len(time_segments) > 1
        if was_time_split:
            threads_split_time += 1

        # Step 2: size-cap split
        final_segments = []
        size_split_occurred = False
        for seg in time_segments:
            size_segs = _size_cap_split(seg)
            if len(size_segs) > 1:
                size_split_occurred = True
            final_segments.extend(size_segs)

        if size_split_occurred:
            threads_split_size += 1

        total_segs = len(final_segments)
        was_split = total_segs > 1

        for seg_idx, segment in enumerate(final_segments, 1):
            if total_segs == 1:
                thread_id = str(conv_id)
            else:
                thread_id = f"{conv_id}_{seg_idx}"

            row = _build_thread_row(
                thread_id=thread_id,
                segment=segment,
                original_conv_id=str(conv_id),
                was_split=was_split,
                user_email=user_email,
            )
            thread_rows.append(row)

            # Build message->thread map
            for pos, (_, msg_row) in enumerate(segment.iterrows(), 1):
                map_rows.append({
                    "message_id": msg_row.get("message_id"),
                    "thread_id": thread_id,
                    "position_in_thread": pos,
                })

    threads_df = pd.DataFrame(thread_rows)

    # Enforce dtypes
    threads_df["msg_count"] = threads_df["msg_count"].astype("int32")
    threads_df["participant_count"] = threads_df["participant_count"].astype("int32")
    threads_df["self_msg_count"] = threads_df["self_msg_count"].astype("int32")
    threads_df["word_count"] = threads_df["word_count"].astype("int32")
    threads_df["duration_seconds"] = threads_df["duration_seconds"].astype("float64")
    threads_df["has_self"] = threads_df["has_self"].astype(bool)
    threads_df["is_self_initiated"] = threads_df["is_self_initiated"].astype(bool)
    threads_df["was_split"] = threads_df["was_split"].astype(bool)
    threads_df["skip"] = threads_df["skip"].astype(bool)

    map_df = pd.DataFrame(map_rows)
    map_df["position_in_thread"] = map_df["position_in_thread"].astype("int32")

    # ── Stats ─────────────────────────────────────────────────────────────────
    total_threads = len(threads_df)
    skip_single = int((threads_df["skip_reason"] == "single_inbound").sum())
    skip_nopart = int((threads_df["skip_reason"] == "no_participation").sum())
    active_threads = int((~threads_df["skip"]).sum())

    mc = threads_df["msg_count"]
    dur_h = threads_df["duration_seconds"].dropna() / 3600

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_messages": len(df),
        "raw_conversation_ids": raw_conv_ids,
        "threads_after_splitting": total_threads,
        "threads_split_by_time_gap": threads_split_time,
        "threads_split_by_size": threads_split_size,
        "total_threads": total_threads,
        "skipped_threads": {
            "single_inbound": skip_single,
            "no_participation": skip_nopart,
            "total": skip_single + skip_nopart,
        },
        "active_threads": active_threads,
        "thread_stats": {
            "avg_msg_count": round(float(mc.mean()), 1),
            "median_msg_count": int(mc.median()),
            "max_msg_count": int(mc.max()),
            "avg_participants": round(float(threads_df["participant_count"].mean()), 1),
            "avg_duration_hours": round(float(dur_h.mean()), 1) if not dur_h.empty else 0,
            "threads_with_1_message": int((mc == 1).sum()),
            "threads_with_2_5_messages": int(((mc >= 2) & (mc <= 5)).sum()),
            "threads_with_6_10_messages": int(((mc >= 6) & (mc <= 10)).sum()),
            "threads_with_11_plus_messages": int((mc >= 11).sum()),
        },
    }

    # ── Write ─────────────────────────────────────────────────────────────────
    threads_df.to_parquet(OUTPUT_DIR / "email_threads.parquet", index=False)
    map_df.to_parquet(OUTPUT_DIR / "message_thread_map.parquet", index=False)
    (OUTPUT_DIR / "segmentation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Split into {total_threads:,} threads, {active_threads:,} active for summarization.\n")

    print("--- Segmentation summary ---")
    print(f"  Input messages:          {len(df):,}")
    print(f"  Raw conversation IDs:    {raw_conv_ids:,}")
    print(f"  Threads after splitting: {total_threads:,}")
    print(f"    Split by time gap:     {threads_split_time:,} conversations")
    print(f"    Split by size:         {threads_split_size:,} conversations")
    print(f"  Skipped (single inbound):{skip_single:,}")
    print(f"  Skipped (no participation):{skip_nopart:,}")
    print(f"  Active threads:          {active_threads:,}")
    print(f"  Avg msg/thread:          {report['thread_stats']['avg_msg_count']}")
    print(f"  Max msg/thread:          {report['thread_stats']['max_msg_count']}")
    print(f"  Avg duration:            {report['thread_stats']['avg_duration_hours']}h")
    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  email_threads.parquet")
    print("  message_thread_map.parquet")
    print("  segmentation_report.json")
    print(f"\nNext: python mail-miner/28_summarize_mail.py")


if __name__ == "__main__":
    main()
