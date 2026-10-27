"""
05_enrich.py — Message Enrichment

Adds four columns to the flagged message table. All existing columns are
preserved unchanged — this step only appends.

New columns:
  is_self                bool     True if sender_id matches the authenticated
                                  user's Azure AD ID from connection_report.json.
                                  Matched on stable user ID, not display name.

  response_time_seconds  float64  Seconds since the last message from a
                                  *different* sender in the same chat.
                                  Null for the first message in a chat and for
                                  consecutive messages from the same sender
                                  (those are continuations, not responses).

  time_delta_seconds     float64  Seconds since the previous message in the
                                  same chat, regardless of sender.
                                  Null for the first message in each chat.

  word_count             Int32    Whitespace-split word count of `text`.
                                  Null when text is null.

Reads:
  outputs/04_edge_cases/messages_flagged.parquet
  outputs/01_connect/connection_report.json      (for authenticated user's ID)

Writes:
  outputs/05_enrich/messages_enriched.parquet
  outputs/05_enrich/enrichment_report.json

Run from project root:
    python message_miner/05_enrich.py [--force]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/04_edge_cases/messages_flagged.parquet")
CONNECTION_REPORT_PATH = Path("outputs/01_connect/connection_report.json")
OUTPUT_DIR = Path("outputs/05_enrich")


def _load_self_id() -> str:
    """Read the authenticated user's Azure AD ID from connection_report.json."""
    if not CONNECTION_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"{CONNECTION_REPORT_PATH} not found. Run 01_connect.py first."
        )
    report = json.loads(CONNECTION_REPORT_PATH.read_text(encoding="utf-8"))
    user_id = report.get("user", {}).get("id")
    if not user_id:
        raise ValueError(
            f"Could not find user.id in {CONNECTION_REPORT_PATH}. "
            "Re-run 01_connect.py to regenerate."
        )
    return user_id


def _add_is_self(df: pd.DataFrame, self_id: str) -> pd.DataFrame:
    df["is_self"] = df["sender_id"] == self_id
    return df


def _add_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add response_time_seconds and time_delta_seconds.

    Requires df to be sorted by (chat_id, timestamp) before calling.
    Both columns are computed via groupby diff to stay within each chat.
    """
    # time_delta_seconds: elapsed since previous message in same chat
    df["time_delta_seconds"] = (
        df.groupby("chat_id", sort=False)["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # response_time_seconds: elapsed since last message from a *different* sender.
    # Null when the previous message in the chat is from the same sender (continuation).
    # Also null for the first message in a chat (time_delta_seconds is already null there).
    prev_sender = df.groupby("chat_id", sort=False)["sender_id"].shift(1)
    sender_changed = df["sender_id"] != prev_sender

    df["response_time_seconds"] = df["time_delta_seconds"].where(sender_changed)

    return df


def _add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    df["word_count"] = (
        df["text"]
        .str.split()
        .str.len()
        .astype("Int32")   # nullable integer — preserves null for empty/deleted
    )
    return df


def _build_report(df: pd.DataFrame, self_id: str, user_name: str) -> dict:
    self_msgs = df["is_self"].sum()
    total = len(df)

    rt = df["response_time_seconds"].dropna()
    td = df["time_delta_seconds"].dropna()
    wc = df["word_count"].dropna()

    def _stats(s: pd.Series) -> dict:
        if s.empty:
            return {}
        desc = s.describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        return {k: round(float(v), 2) for k, v in desc.items()}

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "authenticated_user": {"id": self_id, "name": user_name},
        "total_messages": total,
        "self_messages": int(self_msgs),
        "self_message_pct": round(100 * self_msgs / total, 2) if total else 0,
        "response_time_seconds": _stats(rt),
        "time_delta_seconds": _stats(td),
        "word_count": _stats(wc.astype(float)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich messages with is_self, timing, word count")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "messages_enriched.parquet").exists() and not args.force:
        answer = input("outputs/05_enrich/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 05_enrich ===\n")

    if not MESSAGES_PATH.exists():
        print(f"ERROR: {MESSAGES_PATH} not found. Run 04_edge_cases.py first.")
        sys.exit(1)

    # ── Load ──────────────────────────────────────────────────────────────────
    self_id = _load_self_id()
    connection_report = json.loads(CONNECTION_REPORT_PATH.read_text(encoding="utf-8"))
    user_name = connection_report.get("user", {}).get("name", "unknown")
    print(f"Authenticated user: {user_name} ({self_id})\n")

    df = pd.read_parquet(MESSAGES_PATH)
    print(f"Loaded {len(df):,} messages.\n")

    # Ensure timestamp is tz-aware for diff() to work correctly
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ── Sort: required for correct within-chat deltas ─────────────────────────
    df = df.sort_values(["chat_id", "timestamp"]).reset_index(drop=True)

    # ── Enrich ────────────────────────────────────────────────────────────────
    print("Adding is_self...")
    df = _add_is_self(df, self_id)

    print("Adding time_delta_seconds and response_time_seconds...")
    df = _add_time_columns(df)

    print("Adding word_count...")
    df = _add_word_count(df)

    # ── Write ─────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_DIR / "messages_enriched.parquet", index=False)

    report = _build_report(df, self_id, user_name)
    (OUTPUT_DIR / "enrichment_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n--- Enrichment summary ---")
    print(f"  Total messages:       {report['total_messages']:,}")
    print(f"  Your messages:        {report['self_messages']:,}  ({report['self_message_pct']}%)")

    rt = report["response_time_seconds"]
    if rt:
        print(f"\n  Response time (seconds):")
        print(f"    median   {rt.get('50%', 0):>10,.1f}")
        print(f"    p95      {rt.get('95%', 0):>10,.1f}")
        print(f"    max      {rt.get('max', 0):>10,.1f}")

    wc = report["word_count"]
    if wc:
        print(f"\n  Word count per message:")
        print(f"    median   {wc.get('50%', 0):>10.1f}")
        print(f"    p95      {wc.get('95%', 0):>10.1f}")
        print(f"    max      {wc.get('max', 0):>10.1f}")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_enriched.parquet")
    print("  enrichment_report.json")
    print(f"\nNext: python message_miner/06_quality_report.py")


if __name__ == "__main__":
    main()
