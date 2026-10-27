"""
04_edge_cases.py — Edge Case Flagging

Reads outputs/03_clean/messages_clean.parquet and adds boolean flag columns
for non-standard message content. Nothing is excluded — only flagged so
downstream steps can filter as needed.

Flags added:
  is_system_event    messageType == "systemEventMessage"
  is_bot_message     sender_id is null (from.user was absent — bot/app sender)
  is_adaptive_card   attachment_types contains a Teams card contentType
  is_file_share      attachment_types contains "reference"
  is_meeting_chat    chat is of type "meeting" (joined from chats.parquet)
  is_forwarded       attachment_types contains "messageReference"
  is_loop_component  attachment_types contains "fluidEmbedCard"
  is_deleted         already in schema; re-confirmed here
  is_emoji_only      cleaned text contains only emoji characters
  is_empty           cleaned text is null or whitespace-only

Outputs:
  outputs/04_edge_cases/messages_flagged.parquet
  outputs/04_edge_cases/edge_case_report.json

Run from project root:
    python message_miner/04_edge_cases.py [--force]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/03_clean/messages_clean.parquet")
CHATS_PATH = Path("outputs/02_extract/chats.parquet")
OUTPUT_DIR = Path("outputs/04_edge_cases")

# ── Emoji detection ───────────────────────────────────────────────────────────
# Covers the main Unicode emoji blocks plus ZWJ / variation selectors.
_EMOJI_ONLY_RE = re.compile(
    r"^["
    r"\U0001F300-\U0001FAFF"   # Misc symbols, emoticons, transport, supplemental
    r"\U0001F000-\U0001F02F"   # Mahjong tiles
    r"\U0001F0A0-\U0001F0FF"   # Playing cards
    r"\U00002600-\U000027BF"   # Misc symbols (stars, weather, etc.)
    r"\U00002700-\U000027BF"   # Dingbats
    r"\U0000FE00-\U0000FEFF"   # Variation selectors
    r"\U0001F900-\U0001F9FF"   # Supplemental symbols and pictographs
    r"\u200d"                   # Zero-width joiner (for compound emoji)
    r"\uFE0F"                   # Emoji variation selector-16
    r"\s"
    r"]+$",
    re.UNICODE,
)


def _is_emoji_only(text: str | None) -> bool:
    if not text or not text.strip():
        return False
    return bool(_EMOJI_ONLY_RE.match(text.strip()))


# ── Attachment type helpers ───────────────────────────────────────────────────

def _attachment_contains(series: pd.Series, substring: str) -> pd.Series:
    """True where attachment_types (comma-separated string) contains substring."""
    return series.fillna("").str.contains(substring, case=False, regex=False)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Flag edge case messages")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "messages_flagged.parquet").exists() and not args.force:
        answer = input("outputs/04_edge_cases/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 04_edge_cases ===\n")

    for path in (MESSAGES_PATH, CHATS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(MESSAGES_PATH)
    chats_df = pd.read_parquet(CHATS_PATH)
    print(f"Loaded {len(df):,} messages across {chats_df['chat_id'].nunique():,} chats.\n")

    # ── Build meeting chat set ────────────────────────────────────────────────
    meeting_chat_ids = set(
        chats_df.loc[chats_df["chat_type"] == "meeting", "chat_id"]
    )

    # ── Apply flags ───────────────────────────────────────────────────────────

    df["is_system_event"] = df["message_type"] == "systemEventMessage"

    # Bot / app messages: sender_id is null on non-deleted messages.
    # Deleted messages can also have null sender, so we don't restrict.
    df["is_bot_message"] = df["sender_id"].isna()

    df["is_adaptive_card"] = _attachment_contains(df["attachment_types"], "vnd.microsoft.card")

    df["is_file_share"] = _attachment_contains(df["attachment_types"], "reference")

    df["is_meeting_chat"] = df["chat_id"].isin(meeting_chat_ids)

    df["is_forwarded"] = _attachment_contains(df["attachment_types"], "messageReference")

    df["is_loop_component"] = _attachment_contains(df["attachment_types"], "fluidEmbedCard")

    # is_deleted already exists in schema — no change needed

    df["is_emoji_only"] = df["text"].apply(_is_emoji_only)

    df["is_empty"] = df["text"].isna() | (df["text"].str.strip() == "")

    # ── Write output ──────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_DIR / "messages_flagged.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    flag_cols = [
        "is_system_event",
        "is_bot_message",
        "is_adaptive_card",
        "is_file_share",
        "is_meeting_chat",
        "is_forwarded",
        "is_loop_component",
        "is_deleted",
        "is_emoji_only",
        "is_empty",
    ]

    flag_counts = {col: int(df[col].sum()) for col in flag_cols}

    # Co-occurrence: how many messages have more than one flag
    flag_df = df[flag_cols].astype(int)
    multi_flagged = int((flag_df.sum(axis=1) > 1).sum())
    clean_messages = int((flag_df.sum(axis=1) == 0).sum())

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_messages": len(df),
        "clean_messages": clean_messages,
        "multi_flagged_messages": multi_flagged,
        "flag_counts": flag_counts,
    }
    (OUTPUT_DIR / "edge_case_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("--- Edge case counts ---")
    for col, count in flag_counts.items():
        pct = 100 * count / len(df) if len(df) else 0
        print(f"  {col:<22} {count:>7,}  ({pct:.1f}%)")

    print(f"\n  Clean (no flags):      {clean_messages:>7,}  ({100*clean_messages/len(df):.1f}%)")
    print(f"  Multi-flagged:         {multi_flagged:>7,}  ({100*multi_flagged/len(df):.1f}%)")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_flagged.parquet  all messages with flag columns added")
    print("  edge_case_report.json     flag counts and summary")
    print(f"\nNext: python message_miner/04b_report.py")


if __name__ == "__main__":
    main()
