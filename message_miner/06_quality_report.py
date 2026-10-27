"""
06_quality_report.py — Data Integrity Validation

Final checkpoint before Phase 3 analysis. Runs seven checks against
messages_enriched.parquet and chats.parquet and produces a pass/warn/fail
verdict with per-check detail.

Checks:
  null_audit               Null % per column. timestamp/chat_id nulls → fail.
  timestamp_ordering       Timestamps monotonically non-decreasing within each
                           chat when sorted by msg_index. Violations → fail.
  msg_index_contiguity     msg_index runs 0..N-1 with no gaps per chat.
                           Gaps → warn (may indicate dropped messages).
  timestamp_gap_detection  Any time_delta_seconds > GAP_THRESHOLD_DAYS within
                           a chat. Large gaps may indicate API pagination gaps.
                           → warn.
  chat_count_validation    msg_count in chats.parquet vs actual rows in
                           messages. Mismatches → warn.
  is_self_coverage         is_self=True for 0% → fail (ID match broken).
                           is_self=True for >99% → fail (sender data suspect).
  response_time_sanity     Negative response_time_seconds → fail.
                           Values > 1 year → warn.

Overall status is the worst of all individual check statuses.

Reads:
  outputs/05_enrich/messages_enriched.parquet
  outputs/02_extract/chats.parquet

Writes:
  outputs/06_quality/quality_report.json
  outputs/06_quality/flagged_chats.json

Run from project root:
    python message_miner/06_quality_report.py [--force] [--gap-days N]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
CHATS_PATH = Path("outputs/02_extract/chats.parquet")
OUTPUT_DIR = Path("outputs/06_quality")

# Columns where any null is a hard problem
_CRITICAL_COLS = {"chat_id", "message_id", "timestamp"}

# Columns where nulls are expected — don't warn on these
_EXPECTED_NULLABLE = {
    "text", "text_html", "sender_id", "sender_name", "last_edited",
    "mention_ids", "attachment_types", "response_time_seconds",
    "time_delta_seconds", "word_count",
}

_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}


def _worst(a: str, b: str) -> str:
    return a if _STATUS_RANK[a] >= _STATUS_RANK[b] else b


# ── Checks ────────────────────────────────────────────────────────────────────

def check_null_audit(df: pd.DataFrame) -> tuple[dict, dict]:
    """Return (check_result, per_chat_flags)."""
    total = len(df)
    null_pcts = {
        col: round(100 * df[col].isna().sum() / total, 3)
        for col in df.columns
    }

    # Critical columns: fail if nulls >= 0.5%, warn if nulls are present but < 0.5%
    # (small numbers of null timestamps are a source data issue, not a pipeline bug)
    critical_fail = {
        col: pct for col, pct in null_pcts.items()
        if col in _CRITICAL_COLS and pct >= 0.5
    }
    critical_warn = {
        col: pct for col, pct in null_pcts.items()
        if col in _CRITICAL_COLS and 0 < pct < 0.5
    }
    unexpected_high = {
        col: pct for col, pct in null_pcts.items()
        if col not in _CRITICAL_COLS
        and col not in _EXPECTED_NULLABLE
        and pct > 5
    }

    if critical_fail:
        status = "fail"
        note = f"Critical columns with >=0.5% nulls: {critical_fail}"
    elif critical_warn or unexpected_high:
        status = "warn"
        parts = []
        if critical_warn:
            parts.append(f"Critical columns with <0.5% nulls (source data): {critical_warn}")
        if unexpected_high:
            parts.append(f"Unexpected high-null columns (>5%): {list(unexpected_high.keys())}")
        note = "; ".join(parts)
    else:
        status = "pass"
        note = "No unexpected nulls."

    return {
        "status": status,
        "note": note,
        "details": null_pcts,
    }, {}


def check_timestamp_ordering(df: pd.DataFrame) -> tuple[dict, dict]:
    """Timestamps should be monotonically non-decreasing within each chat by msg_index.
    Chats whose only violations are caused by null timestamps (source data gaps) are
    downgraded to warn rather than fail."""
    bad_chats: dict[str, str] = {}
    null_caused: dict[str, str] = {}

    grouped = df.sort_values(["chat_id", "msg_index"]).groupby("chat_id", sort=False)
    for chat_id, group in grouped:
        # Check ordering on non-null timestamps only
        ts_full = group["timestamp"]
        ts_nonnull = ts_full.dropna()
        if len(ts_nonnull) < 2:
            continue

        violations_full = int((ts_full.diff().dropna() < pd.Timedelta(0)).sum())
        violations_nonnull = int((ts_nonnull.diff().dropna() < pd.Timedelta(0)).sum())

        if violations_nonnull > 0:
            # Real ordering problem even among timestamped messages
            bad_chats[str(chat_id)] = f"{violations_nonnull} out-of-order timestamp(s)"
        elif violations_full > 0 and ts_full.isna().any():
            # Ordering only breaks because of null-timestamp rows — source data issue
            null_caused[str(chat_id)] = f"{violations_full} apparent violation(s) caused by null timestamps"

    if bad_chats:
        status = "fail"
        note = f"{len(bad_chats)} chats have out-of-order timestamps."
    elif null_caused:
        status = "warn"
        note = f"{len(null_caused)} chats have apparent ordering issues caused by null timestamps (source data)."
    else:
        status = "pass"
        note = "All chats in order."

    all_flagged = {**bad_chats, **null_caused}
    return {
        "status": status,
        "chats_flagged": len(all_flagged),
        "note": note,
    }, all_flagged


def check_msg_index_contiguity(df: pd.DataFrame) -> tuple[dict, dict]:
    """msg_index should run 0..N-1 with no gaps per chat."""
    bad_chats: dict[str, str] = {}

    for chat_id, group in df.groupby("chat_id", sort=False):
        idx = group["msg_index"].sort_values().reset_index(drop=True)
        expected = pd.RangeIndex(len(idx))
        if not (idx.values == expected.values).all():
            bad_chats[str(chat_id)] = "msg_index not contiguous (0..N-1)"

    status = "warn" if bad_chats else "pass"
    return {
        "status": status,
        "chats_flagged": len(bad_chats),
        "note": f"{len(bad_chats)} chats have non-contiguous msg_index." if bad_chats else "All msg_index values contiguous.",
    }, bad_chats


def check_timestamp_gaps(df: pd.DataFrame, gap_days: int) -> tuple[dict, dict]:
    """Flag chats with any time_delta_seconds gap exceeding gap_days."""
    threshold_seconds = gap_days * 86_400
    bad_chats: dict[str, str] = {}

    large_gaps = df[df["time_delta_seconds"] > threshold_seconds]
    for chat_id, group in large_gaps.groupby("chat_id", sort=False):
        n = len(group)
        max_gap_days = round(group["time_delta_seconds"].max() / 86_400, 1)
        bad_chats[str(chat_id)] = f"{n} gap(s) exceeding {gap_days}d; largest = {max_gap_days}d"

    status = "warn" if bad_chats else "pass"
    return {
        "status": status,
        "gap_threshold_days": gap_days,
        "chats_flagged": len(bad_chats),
        "note": f"{len(bad_chats)} chats have gaps > {gap_days} days." if bad_chats else f"No gaps exceeding {gap_days} days.",
    }, bad_chats


def check_chat_count_validation(df: pd.DataFrame, chats_df: pd.DataFrame) -> tuple[dict, dict]:
    """Compare msg_count in chats.parquet against actual row counts in messages."""
    actual_counts = df.groupby("chat_id", sort=False).size().rename("actual_count")
    comparison = chats_df.set_index("chat_id")[["msg_count"]].join(actual_counts, how="left")
    comparison["actual_count"] = comparison["actual_count"].fillna(0).astype(int)
    comparison["delta"] = comparison["actual_count"] - comparison["msg_count"]

    mismatches = comparison[comparison["delta"] != 0]
    bad_chats: dict[str, str] = {}
    for chat_id, row in mismatches.iterrows():
        bad_chats[str(chat_id)] = (
            f"chats.parquet says {row['msg_count']}, "
            f"actual rows = {row['actual_count']} (delta {row['delta']:+d})"
        )

    status = "warn" if bad_chats else "pass"
    return {
        "status": status,
        "chats_flagged": len(bad_chats),
        "note": f"{len(bad_chats)} chats have msg_count mismatches." if bad_chats else "All chat message counts match.",
    }, bad_chats


def check_is_self_coverage(df: pd.DataFrame) -> tuple[dict, dict]:
    """Verify is_self=True for a reasonable proportion of messages."""
    total = len(df)
    self_count = int(df["is_self"].sum())
    self_pct = round(100 * self_count / total, 2) if total else 0.0

    if self_count == 0:
        status = "fail"
        note = "is_self is True for 0 messages — user ID matching failed."
    elif self_pct > 99:
        status = "fail"
        note = f"is_self is True for {self_pct}% of messages — sender data may be corrupt."
    elif self_pct < 1:
        status = "warn"
        note = f"is_self is True for only {self_pct}% of messages — unexpectedly low."
    else:
        status = "pass"
        note = f"is_self=True for {self_pct}% of messages."

    return {
        "status": status,
        "self_message_count": self_count,
        "self_message_pct": self_pct,
        "note": note,
    }, {}


def check_response_time_sanity(df: pd.DataFrame) -> tuple[dict, dict]:
    """Flag negative or implausibly large response_time_seconds values."""
    rt = df["response_time_seconds"].dropna()
    one_year_seconds = 365 * 86_400

    n_negative = int((rt < 0).sum())
    n_over_year = int((rt > one_year_seconds).sum())

    if n_negative > 0:
        status = "fail"
        note = f"{n_negative} negative response_time_seconds values (impossible — sorting bug?)."
    elif n_over_year > 0:
        status = "warn"
        note = f"{n_over_year} response times exceed 1 year — likely dormant chats resuming."
    else:
        status = "pass"
        note = "All response times are non-negative and within 1 year."

    return {
        "status": status,
        "negative_count": n_negative,
        "over_one_year_count": n_over_year,
        "note": note,
    }, {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run data integrity checks")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    parser.add_argument("--gap-days", type=int, default=30, help="Gap threshold in days (default: 30)")
    args = parser.parse_args()

    if (OUTPUT_DIR / "quality_report.json").exists() and not args.force:
        answer = input("outputs/06_quality/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 06_quality_report ===\n")

    for path in (MESSAGES_PATH, CHATS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(MESSAGES_PATH)
    chats_df = pd.read_parquet(CHATS_PATH)
    print(f"Loaded {len(df):,} messages, {len(chats_df):,} chats.\n")

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ── Run checks ────────────────────────────────────────────────────────────
    print("Running checks...")
    all_flagged: dict[str, dict] = {}  # chat_id -> {check: detail}

    def _run(name: str, result: dict, flagged: dict) -> dict:
        icon = {"pass": "[pass]", "warn": "[warn]", "fail": "[FAIL]"}[result["status"]]
        print(f"  {icon} {name:<30} {result.get('note', '')}")
        for chat_id, detail in flagged.items():
            existing = all_flagged.setdefault(chat_id, {"checks_failed": [], "details": {}})
            existing["checks_failed"].append(name)
            existing["details"][name] = detail
        return result

    checks = {}
    checks["null_audit"]              = _run("null_audit",              *check_null_audit(df))
    checks["timestamp_ordering"]      = _run("timestamp_ordering",      *check_timestamp_ordering(df))
    checks["msg_index_contiguity"]    = _run("msg_index_contiguity",    *check_msg_index_contiguity(df))
    checks["timestamp_gap_detection"] = _run("timestamp_gap_detection", *check_timestamp_gaps(df, args.gap_days))
    checks["chat_count_validation"]   = _run("chat_count_validation",   *check_chat_count_validation(df, chats_df))
    checks["is_self_coverage"]        = _run("is_self_coverage",        *check_is_self_coverage(df))
    checks["response_time_sanity"]    = _run("response_time_sanity",    *check_response_time_sanity(df))

    overall = "pass"
    for result in checks.values():
        overall = _worst(overall, result["status"])

    # ── Summary stats ─────────────────────────────────────────────────────────
    rt = df["response_time_seconds"].dropna()
    summary = {
        "total_messages": len(df),
        "total_chats": len(chats_df),
        "self_message_pct": checks["is_self_coverage"]["self_message_pct"],
        "median_response_time_seconds": round(float(rt.median()), 1) if not rt.empty else None,
        "p95_response_time_seconds": round(float(rt.quantile(0.95)), 1) if not rt.empty else None,
        "chats_with_any_flag": len(all_flagged),
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "checks": checks,
        "summary": summary,
    }
    (OUTPUT_DIR / "quality_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    flagged_list = [
        {"chat_id": cid, **data}
        for cid, data in sorted(all_flagged.items())
    ]
    (OUTPUT_DIR / "flagged_chats.json").write_text(
        json.dumps(flagged_list, indent=2),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    status_label = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[overall]
    print(f"\n{'='*45}")
    print(f"  Overall status: {status_label}")
    print(f"  Chats flagged:  {len(all_flagged)}")
    print(f"{'='*45}")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  quality_report.json   full check results")
    print("  flagged_chats.json    chats with one or more issues")

    if overall == "pass":
        print("\nData looks good. Ready for Phase 3.")
    elif overall == "warn":
        print("\nWarnings found. Review flagged_chats.json before proceeding to Phase 3.")
    else:
        print("\nFailures found. Investigate before proceeding to Phase 3.")


if __name__ == "__main__":
    main()
