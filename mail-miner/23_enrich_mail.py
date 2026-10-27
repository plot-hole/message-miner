"""
23_enrich_mail.py — Email Feature Enrichment

Adds computed features to the flagged mail table.

New columns:
  is_self               Sender email matches authenticated user
  is_sent               Sent Items folder or is_self
  is_reply              Subject starts with Re:
  is_forward            Subject starts with Fw: or Fwd:
  subject_clean         Subject with Re:/Fw:/Fwd: prefixes stripped
  word_count            Word count of body_text
  char_count            Character count of body_text
  recipient_count       to_count + cc_count
  thread_position       Position within conversation_id thread (1-based)
  thread_length         Total messages in this conversation thread
  is_thread_starter     thread_position == 1
  response_time_seconds Seconds from last inbound message to this sent reply
  day_of_week           0=Monday ... 6=Sunday (from received_datetime)
  hour_of_day           0-23 UTC (from received_datetime)
  is_business_hours     Monday-Friday, hour 8-17 UTC

Reads:
  outputs/22_edge_cases_mail/messages_mail_flagged.parquet
  outputs/20_extract_mail/extraction_report.json    (for user email)

Writes:
  outputs/23_enrich_mail/messages_mail_enriched.parquet
  outputs/23_enrich_mail/enrichment_report.json

Run from project root:
    python mail-miner/23_enrich_mail.py [--force]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

INPUT_PATH = Path("outputs/22_edge_cases_mail/messages_mail_flagged.parquet")
EXTRACTION_REPORT_PATH = Path("outputs/20_extract_mail/extraction_report.json")
OUTPUT_DIR = Path("outputs/23_enrich_mail")

_RE_PREFIX = re.compile(r"^(re|fw|fwd)\s*:\s*", re.IGNORECASE)

_MAX_RESPONSE_SECONDS = 7 * 24 * 3600  # 7 days cap


def _strip_subject_prefixes(subject: str | None) -> str | None:
    """Remove all Re:/Fw:/Fwd: prefixes from a subject string."""
    if not subject:
        return subject
    prev = None
    result = subject.strip()
    while result != prev:
        prev = result
        result = _RE_PREFIX.sub("", result).strip()
    return result if result else subject


def _add_basic_features(df: pd.DataFrame, user_email: str) -> pd.DataFrame:
    user_email_lower = user_email.lower()

    df["is_self"] = df["sender_email"].fillna("").str.lower() == user_email_lower
    df["is_sent"] = df["is_self"] | (df["folder_name"].astype(str) == "Sent Items")
    df["is_reply"] = df["subject"].fillna("").str.match(r"^\s*re\s*:", case=False)
    df["is_forward"] = df["subject"].fillna("").str.match(r"^\s*f(w|wd)\s*:", case=False)
    df["subject_clean"] = df["subject"].apply(_strip_subject_prefixes)
    df["word_count"] = (
        df["body_text"].str.split().str.len().astype("Int32")
    )
    df["char_count"] = df["body_text"].str.len().astype("Int32")
    df["recipient_count"] = (
        (df["to_count"].fillna(0) + df["cc_count"].fillna(0)).astype("int32")
    )
    df["day_of_week"] = df["received_datetime"].dt.dayofweek.astype("Int8")
    df["hour_of_day"] = df["received_datetime"].dt.hour.astype("Int8")
    df["is_business_hours"] = (
        (df["day_of_week"].between(0, 4)) &  # Mon-Fri
        (df["hour_of_day"].between(8, 17))
    )
    return df


def _add_thread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add thread_position, thread_length, is_thread_starter."""
    df = df.sort_values(["conversation_id", "received_datetime"]).reset_index(drop=True)

    df["thread_position"] = (
        df.groupby("conversation_id", sort=False).cumcount() + 1
    ).astype("int32")

    thread_lengths = df.groupby("conversation_id", sort=False)["conversation_id"].transform("count")
    df["thread_length"] = thread_lengths.astype("int32")
    df["is_thread_starter"] = df["thread_position"] == 1

    return df


def _add_response_times(df: pd.DataFrame, user_email: str) -> pd.DataFrame:
    """
    For each sent reply, find the most recent inbound message in the same thread
    and compute the response time in seconds.

    Cap at _MAX_RESPONSE_SECONDS. Null for non-replies and when no prior inbound
    message exists.
    """
    user_email_lower = user_email.lower()

    # Work with timezone-aware datetimes
    if df["received_datetime"].dt.tz is None:
        df["received_datetime"] = pd.to_datetime(df["received_datetime"], utc=True, errors="coerce")
    if df["sent_datetime"].dt.tz is None:
        df["sent_datetime"] = pd.to_datetime(df["sent_datetime"], utc=True, errors="coerce")

    df["response_time_seconds"] = float("nan")

    # Only compute for self-sent replies
    is_sent_reply = df["is_self"] & df["is_reply"]
    if not is_sent_reply.any():
        return df

    # Build lookup: conversation_id → sorted list of (received_datetime, sender_email)
    # for inbound messages only
    inbound = df[~df["is_self"] & df["received_datetime"].notna()].copy()
    inbound_grouped = inbound.sort_values("received_datetime").groupby(
        "conversation_id", sort=False
    )

    response_times = []
    for idx, row in df[is_sent_reply].iterrows():
        conv_id = row["conversation_id"]
        sent_dt = row["sent_datetime"]
        if pd.isna(sent_dt) or conv_id not in inbound_grouped.groups:
            response_times.append((idx, float("nan")))
            continue

        conv_inbound = inbound_grouped.get_group(conv_id)
        # Most recent inbound message before this reply
        prior = conv_inbound[conv_inbound["received_datetime"] < sent_dt]
        if prior.empty:
            response_times.append((idx, float("nan")))
            continue

        last_inbound_dt = prior["received_datetime"].max()
        delta = (sent_dt - last_inbound_dt).total_seconds()

        if delta < 0:
            response_times.append((idx, float("nan")))
        elif delta > _MAX_RESPONSE_SECONDS:
            response_times.append((idx, _MAX_RESPONSE_SECONDS))
        else:
            response_times.append((idx, delta))

    for idx, rt in response_times:
        df.at[idx, "response_time_seconds"] = rt

    df["response_time_seconds"] = pd.to_numeric(df["response_time_seconds"], errors="coerce")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich emails with computed features")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "messages_mail_enriched.parquet").exists() and not args.force:
        answer = input("outputs/23_enrich_mail/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 23_enrich_mail ===\n")

    for path in (INPUT_PATH, EXTRACTION_REPORT_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    extraction_report = json.loads(EXTRACTION_REPORT_PATH.read_text(encoding="utf-8"))
    user_email = extraction_report.get("user_email", "")
    user_name = extraction_report.get("user_name", "")

    print(f"Loaded {len(df):,} messages.")
    print(f"User: {user_name} <{user_email}>\n")

    # Ensure tz-aware datetimes
    for col in ("received_datetime", "sent_datetime"):
        if col in df.columns and df[col].dt.tz is None:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    print("Adding basic features (is_self, is_reply, word_count, ...)...")
    df = _add_basic_features(df, user_email)

    print("Building thread features (thread_position, thread_length)...")
    df = _add_thread_features(df)

    print("Computing response times...")
    df = _add_response_times(df, user_email)

    # ── Write ─────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_DIR / "messages_mail_enriched.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    total = len(df)
    sent_count = int(df["is_self"].sum())
    received_count = total - sent_count
    replies = int(df["is_reply"].sum())
    forwards = int(df["is_forward"].sum())
    unique_threads = int(df["conversation_id"].nunique())
    avg_thread_len = round(float(df["thread_length"].mean()), 1) if total else 0
    max_thread_len = int(df["thread_length"].max()) if total else 0
    avg_wc = round(float(df["word_count"].dropna().mean()), 1) if total else 0
    rt = df["response_time_seconds"].dropna()
    median_rt = round(float(rt.median()), 0) if not rt.empty else None
    external_count = int(df["is_external"].sum()) if "is_external" in df.columns else 0
    external_pct = round(100 * external_count / total, 1) if total else 0

    sent_df = df[df["is_self"]]
    avg_recipients_sent = round(
        float(sent_df["recipient_count"].mean()), 1
    ) if not sent_df.empty else 0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_messages": total,
        "sent_by_self": sent_count,
        "received": received_count,
        "replies": replies,
        "forwards": forwards,
        "unique_threads": unique_threads,
        "avg_thread_length": avg_thread_len,
        "max_thread_length": max_thread_len,
        "avg_word_count": avg_wc,
        "median_response_time_s": median_rt,
        "avg_recipients_per_sent": avg_recipients_sent,
        "external_pct": external_pct,
    }
    (OUTPUT_DIR / "enrichment_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\n--- Enrichment summary ---")
    print(f"  Total messages:        {total:,}")
    print(f"  Sent by you:           {sent_count:,}")
    print(f"  Received:              {received_count:,}")
    print(f"  Replies:               {replies:,}")
    print(f"  Forwards:              {forwards:,}")
    print(f"  Unique threads:        {unique_threads:,}")
    print(f"  Avg thread length:     {avg_thread_len}")
    print(f"  Max thread length:     {max_thread_len:,}")
    print(f"  Avg word count:        {avg_wc}")
    if median_rt is not None:
        print(f"  Median response time:  {median_rt:,.0f}s  ({median_rt/60:.0f} min)")
    print(f"  External mail:         {external_pct}%")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_mail_enriched.parquet")
    print("  enrichment_report.json")
    print(f"\nNext: python mail-miner/24_quality_report_mail.py")


if __name__ == "__main__":
    main()
