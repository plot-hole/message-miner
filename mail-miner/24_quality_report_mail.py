"""
24_quality_report_mail.py — Email Data Quality Validation

Final checkpoint after the email pipeline. Runs integrity checks and
produces a pass/warn/fail verdict.

Checks:
  row_count             Total messages matches extraction_report
  date_range            No messages before 2020 or after today
  null_subjects         Messages with null/empty subject
  null_senders          Messages with null sender_email
  duplicate_message_ids Duplicate message_id values
  empty_bodies          Messages where is_empty_body == True
  self_send_consistency is_self count ≈ Sent Items folder count
  thread_integrity      Replies have a prior message in their thread
  response_time_sanity  Median response time between 1 min and 48 hours
  encoding_artifacts    Messages with mojibake / encoding errors

Reads:
  outputs/23_enrich_mail/messages_mail_enriched.parquet
  outputs/20_extract_mail/extraction_report.json

Writes:
  outputs/24_quality_mail/quality_report_mail.json
  outputs/24_quality_mail/flagged_threads.json

Run from project root:
    python mail-miner/24_quality_report_mail.py [--force]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/23_enrich_mail/messages_mail_enriched.parquet")
EXTRACTION_REPORT_PATH = Path("outputs/20_extract_mail/extraction_report.json")
OUTPUT_DIR = Path("outputs/24_quality_mail")

_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}

# Encoding artifact patterns: common mojibake from UTF-8 bytes read as cp1252.
# U+00C3 (Ã) followed by a latin supplement char signals a mangled 2-byte UTF-8 sequence.
# U+00E2 U+0080 (â€) prefix signals a mangled 3-byte UTF-8 sequence (smart quotes, ellipsis, etc.).
_ENCODING_RE = re.compile(
    "\u00c3[\u0080-\u00bf]"   # Ã + continuation byte — e.g. Ã© instead of é
    "|\u00e2\u0080[\u0090-\u00be]"  # â€ + byte — e.g. â€™ instead of '
    "|\u00c2[\u00a0-\u00bf]"  # Â + byte — e.g. Â  (non-breaking space artifact)
)


def _worst(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


# ── Checks ────────────────────────────────────────────────────────────────────

def check_row_count(df: pd.DataFrame, extraction_report: dict) -> dict:
    expected = extraction_report.get("total_messages", 0)
    actual = len(df)
    diff_pct = abs(actual - expected) / expected * 100 if expected else 0

    if actual == expected:
        return {"status": "pass", "expected": expected, "actual": actual,
                "note": f"{actual:,} messages (matches extraction)"}
    elif diff_pct <= 1:
        return {"status": "warn", "expected": expected, "actual": actual,
                "note": f"{actual:,} vs {expected:,} expected ({diff_pct:.2f}% difference)"}
    else:
        return {"status": "fail", "expected": expected, "actual": actual,
                "note": f"{actual:,} vs {expected:,} expected ({diff_pct:.2f}% difference — exceeds 1%)"}


def check_date_range(df: pd.DataFrame) -> dict:
    today = pd.Timestamp.now(tz="UTC")
    cutoff_past = pd.Timestamp("2020-01-01", tz="UTC")

    dates = df["received_datetime"].dropna()
    total = len(dates)
    if total == 0:
        return {"status": "warn", "note": "No received_datetime values to check."}

    before_2020 = int((dates < cutoff_past).sum())
    after_today = int((dates > today).sum())
    outside = before_2020 + after_today
    pct = 100 * outside / total

    date_first = str(dates.min().date())
    date_last = str(dates.max().date())

    if outside == 0:
        return {"status": "pass",
                "note": f"{date_first} to {date_last}",
                "date_first": date_first, "date_last": date_last}
    elif pct < 0.1:
        return {"status": "warn",
                "note": f"{outside} messages outside valid range ({pct:.3f}%)",
                "date_first": date_first, "date_last": date_last}
    else:
        return {"status": "fail",
                "note": f"{outside} messages outside valid range ({pct:.3f}% — exceeds 0.1%)",
                "date_first": date_first, "date_last": date_last}


def check_null_subjects(df: pd.DataFrame) -> dict:
    total = len(df)
    null_count = int(df["subject"].isna().sum() + (df["subject"].fillna("").str.strip() == "").sum())
    pct = 100 * null_count / total if total else 0

    if pct < 1:
        return {"status": "pass", "null_count": null_count,
                "note": f"{null_count} messages ({pct:.1f}%)"}
    elif pct <= 5:
        return {"status": "warn", "null_count": null_count,
                "note": f"{null_count} messages ({pct:.1f}%) — likely calendar or notification artifacts"}
    else:
        return {"status": "fail", "null_count": null_count,
                "note": f"{null_count} messages ({pct:.1f}%) — exceeds 5% threshold"}


def check_null_senders(df: pd.DataFrame) -> dict:
    null_count = int(df["sender_email"].isna().sum())
    if null_count == 0:
        return {"status": "pass", "null_count": 0, "note": "No null senders"}
    else:
        return {"status": "warn", "null_count": null_count,
                "note": f"{null_count} messages have no sender_email"}


def check_duplicate_ids(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated(subset=["message_id"]).sum())
    if dup_count == 0:
        return {"status": "pass", "duplicate_count": 0, "note": "No duplicate message IDs"}
    elif dup_count <= 10:
        return {"status": "warn", "duplicate_count": dup_count,
                "note": f"{dup_count} duplicate message_id values"}
    else:
        return {"status": "fail", "duplicate_count": dup_count,
                "note": f"{dup_count} duplicate message_id values — exceeds 10"}


def check_empty_bodies(df: pd.DataFrame) -> dict:
    total = len(df)
    empty_count = int(df["is_empty_body"].sum()) if "is_empty_body" in df.columns else 0
    pct = 100 * empty_count / total if total else 0

    if pct < 5:
        return {"status": "pass", "empty_count": empty_count,
                "note": f"{empty_count} ({pct:.1f}%)"}
    elif pct <= 15:
        return {"status": "warn", "empty_count": empty_count,
                "note": f"{empty_count} ({pct:.1f}%)"}
    else:
        return {"status": "fail", "empty_count": empty_count,
                "note": f"{empty_count} ({pct:.1f}%) — exceeds 15% threshold"}


def check_self_send_consistency(df: pd.DataFrame) -> dict:
    self_count = int(df["is_self"].sum())
    sent_folder_count = int((df["folder_name"].astype(str) == "Sent Items").sum())

    if sent_folder_count == 0:
        return {"status": "warn",
                "note": "No Sent Items messages found — cannot validate self-send consistency",
                "is_self_count": self_count, "sent_folder_count": sent_folder_count}

    diff_pct = abs(self_count - sent_folder_count) / sent_folder_count * 100

    if diff_pct <= 5:
        return {"status": "pass",
                "note": f"{self_count:,} sent vs {sent_folder_count:,} in Sent Items ({diff_pct:.1f}%)",
                "is_self_count": self_count, "sent_folder_count": sent_folder_count}
    elif diff_pct <= 25:
        return {"status": "warn",
                "note": f"{self_count:,} sent vs {sent_folder_count:,} in Sent Items ({diff_pct:.1f}%) — likely sent from mobile or delegate",
                "is_self_count": self_count, "sent_folder_count": sent_folder_count}
    else:
        return {"status": "fail",
                "note": f"{self_count:,} sent vs {sent_folder_count:,} in Sent Items ({diff_pct:.1f}% — exceeds 25%)",
                "is_self_count": self_count, "sent_folder_count": sent_folder_count}


def check_thread_integrity(df: pd.DataFrame) -> dict:
    """Replies should have a prior message in their conversation thread."""
    replies = df[df["is_reply"].fillna(False)]
    if len(replies) == 0:
        return {"status": "pass", "note": "No replies to check."}

    # A reply has thread_integrity if its thread_position > 1
    valid_replies = int((replies["thread_position"] > 1).sum())
    pct = 100 * valid_replies / len(replies)

    if pct >= 95:
        return {"status": "pass",
                "note": f"{pct:.1f}% of replies have prior messages",
                "valid_pct": round(pct, 1)}
    elif pct >= 80:
        return {"status": "warn",
                "note": f"{pct:.1f}% of replies have prior messages (80-95% range)",
                "valid_pct": round(pct, 1)}
    else:
        return {"status": "fail",
                "note": f"Only {pct:.1f}% of replies have prior messages (below 80%)",
                "valid_pct": round(pct, 1)}


def check_response_time_sanity(df: pd.DataFrame) -> dict:
    rt = df["response_time_seconds"].dropna()
    if rt.empty:
        return {"status": "warn", "note": "No response_time_seconds values to check."}

    median_rt = float(rt.median())
    one_min = 60.0
    forty_eight_hours = 48 * 3600.0
    seven_days = 7 * 24 * 3600.0

    n_negative = int((rt < 0).sum())
    if n_negative > 0:
        return {"status": "fail",
                "note": f"{n_negative} negative response times (impossible — sorting issue?)",
                "median_seconds": round(median_rt, 0)}

    if median_rt < one_min or median_rt > forty_eight_hours:
        if median_rt > seven_days:
            return {"status": "fail",
                    "note": f"Median response time {median_rt/3600:.1f}h — exceeds 7 days",
                    "median_seconds": round(median_rt, 0)}
        return {"status": "warn",
                "note": f"Median response time {median_rt/60:.0f} min — outside 1min–48h range",
                "median_seconds": round(median_rt, 0)}

    median_min = round(median_rt / 60, 0)
    return {"status": "pass",
            "note": f"Median response time {median_min:.0f} minutes",
            "median_seconds": round(median_rt, 0)}


def check_encoding_artifacts(df: pd.DataFrame) -> dict:
    total = len(df)
    bodies = df["body_text"].fillna("")
    artifact_count = int(bodies.str.contains(_ENCODING_RE, regex=True).sum())
    pct = 100 * artifact_count / total if total else 0

    if pct < 0.5:
        return {"status": "pass", "artifact_count": artifact_count,
                "note": f"{artifact_count} messages ({pct:.2f}%)"}
    elif pct <= 2:
        return {"status": "warn", "artifact_count": artifact_count,
                "note": f"{artifact_count} messages ({pct:.2f}%) — encoding artifacts present"}
    else:
        return {"status": "fail", "artifact_count": artifact_count,
                "note": f"{artifact_count} messages ({pct:.2f}%) — exceeds 2% threshold"}


# ── Flagged threads ───────────────────────────────────────────────────────────

def _build_flagged_threads(df: pd.DataFrame) -> dict:
    """Find threads with broken reply chains, oversized threads, or time order issues."""

    # Broken reply chains: replies with thread_position == 1
    broken_replies = df[df["is_reply"].fillna(False) & (df["thread_position"] == 1)]
    broken_by_thread = (
        broken_replies.groupby("conversation_id", sort=False)
        .agg(subject=("subject_clean", "first"), message_count=("message_id", "count"))
        .reset_index()
    )
    broken_chains = [
        {
            "conversation_id": row["conversation_id"],
            "subject": row["subject"],
            "message_count": int(row["message_count"]),
            "issue": f"{row['message_count']} reply/replies with no prior message",
        }
        for _, row in broken_by_thread.iterrows()
    ]

    # Oversized threads: conversation_id with > 50 messages
    thread_sizes = df.groupby("conversation_id", sort=False)["message_id"].count()
    oversized = thread_sizes[thread_sizes > 50]
    oversized_threads = []
    for conv_id, count in oversized.items():
        subj = df[df["conversation_id"] == conv_id]["subject_clean"].dropna().iloc[0] \
            if not df[df["conversation_id"] == conv_id]["subject_clean"].dropna().empty else None
        oversized_threads.append({
            "conversation_id": conv_id,
            "subject": subj,
            "message_count": int(count),
        })

    return {
        "broken_reply_chains": broken_chains,
        "oversized_threads": oversized_threads,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run email data quality checks")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "quality_report_mail.json").exists() and not args.force:
        answer = input("outputs/24_quality_mail/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 24_quality_report_mail ===\n")

    for path in (MESSAGES_PATH, EXTRACTION_REPORT_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(MESSAGES_PATH)
    extraction_report = json.loads(EXTRACTION_REPORT_PATH.read_text(encoding="utf-8"))

    print(f"Loaded {len(df):,} messages.\n")

    # Ensure tz-aware datetime
    if df["received_datetime"].dt.tz is None:
        df["received_datetime"] = pd.to_datetime(df["received_datetime"], utc=True, errors="coerce")

    # ── Run checks ────────────────────────────────────────────────────────────
    print("Email Data Quality Report")
    print("=========================")

    def _run(label: str, result: dict) -> dict:
        icon = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}[result["status"]]
        print(f"  {icon} {label}: {result.get('note', '')}")
        return result

    checks = {
        "row_count":               _run("Row count",               check_row_count(df, extraction_report)),
        "date_range":              _run("Date range",              check_date_range(df)),
        "null_subjects":           _run("Null subjects",           check_null_subjects(df)),
        "null_senders":            _run("Null senders",            check_null_senders(df)),
        "duplicate_message_ids":   _run("Duplicate message IDs",   check_duplicate_ids(df)),
        "empty_bodies":            _run("Empty bodies",            check_empty_bodies(df)),
        "self_send_consistency":   _run("Self-send consistency",   check_self_send_consistency(df)),
        "thread_integrity":        _run("Thread integrity",        check_thread_integrity(df)),
        "response_time_sanity":    _run("Response time sanity",    check_response_time_sanity(df)),
        "encoding_artifacts":      _run("Encoding artifacts",      check_encoding_artifacts(df)),
    }

    overall = "pass"
    for result in checks.values():
        overall = _worst(overall, result["status"])

    warn_count = sum(1 for r in checks.values() if r["status"] == "warn")
    fail_count = sum(1 for r in checks.values() if r["status"] == "fail")

    print()
    if overall == "pass":
        print(f"  Overall: PASS ({warn_count} warnings)" if warn_count else "  Overall: PASS")
    elif overall == "warn":
        print(f"  Overall: WARN ({warn_count} warnings, {fail_count} failures)")
    else:
        print(f"  Overall: FAIL ({fail_count} failures, {warn_count} warnings)")

    # ── Flagged threads ───────────────────────────────────────────────────────
    flagged_threads = _build_flagged_threads(df)

    # ── Write outputs ─────────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "checks": checks,
        "summary": {
            "total_messages": len(df),
            "warnings": warn_count,
            "failures": fail_count,
        },
    }
    (OUTPUT_DIR / "quality_report_mail.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "flagged_threads.json").write_text(
        json.dumps(flagged_threads, indent=2, default=str),
        encoding="utf-8",
    )

    broken = len(flagged_threads["broken_reply_chains"])
    oversized = len(flagged_threads["oversized_threads"])
    print(f"\nFlagged threads: {broken} broken reply chains, {oversized} oversized threads")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  quality_report_mail.json")
    print("  flagged_threads.json")

    if overall == "pass":
        print("\nEmail data looks good. Ready for analysis.")
    elif overall == "warn":
        print("\nWarnings found. Review quality_report_mail.json before proceeding.")
    else:
        print("\nFailures found. Investigate before proceeding.")


if __name__ == "__main__":
    main()
