"""
03_clean.py — HTML Cleaning & System Event Separation

Reads outputs/02_extract/messages.parquet.
Strips HTML from message bodies to produce plain text.
Separates systemEventMessage rows into their own file.

HTML handling (BeautifulSoup, html.parser — no regex for HTML):
  <at>DisplayName</at>        -> @DisplayName
  <emoji alt="...">           -> alt text
  <attachment id="...">       -> stripped (presence already in has_attachments)
  <img ...>                   -> [image]
  <p>, <div>, <br>            -> newlines / whitespace
  <blockquote>, <pre>, <code> -> content preserved, blockquote indented

Outputs:
  outputs/03_clean/messages_clean.parquet   non-system messages with text populated
  outputs/03_clean/system_events.parquet    systemEventMessage rows
  outputs/03_clean/cleaning_report.json     cleaning summary

Run from project root:
    python message_miner/03_clean.py [--force]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

INPUT_PATH = Path("outputs/02_extract/messages.parquet")
OUTPUT_DIR = Path("outputs/03_clean")


# ── HTML → plain text ─────────────────────────────────────────────────────────

def _html_to_text(html: str | None) -> str | None:
    """
    Convert a Teams message HTML body to plain text.

    Preserves:
      - @mention names (from <at> tags)
      - Emoji alt text (from <emoji> tags)
      - Code block content
      - Paragraph / line break structure

    Strips:
      - <attachment> placeholder tags
      - All HTML attributes and inline styles
      - Redundant whitespace
    """
    if not html or not html.strip():
        return None

    soup = BeautifulSoup(html, "html.parser")

    # <at id="N">DisplayName</at>  →  @DisplayName
    for tag in soup.find_all("at"):
        tag.replace_with(f"@{tag.get_text()}")

    # <emoji id="..." alt="😊" title="Smiling face"> → 😊
    for tag in soup.find_all("emoji"):
        replacement = tag.get("alt") or tag.get("title") or ""
        tag.replace_with(replacement)

    # <attachment id="..."></attachment> — strip entirely
    for tag in soup.find_all("attachment"):
        tag.decompose()

    # <img ...> → [image]
    for tag in soup.find_all("img"):
        tag.replace_with("[image]")

    # <blockquote> → indent content with "> "
    for tag in soup.find_all("blockquote"):
        lines = tag.get_text(separator="\n").strip().splitlines()
        quoted = "\n".join(f"> {line}" for line in lines)
        tag.replace_with(f"\n{quoted}\n")

    # <br> → newline
    for tag in soup.find_all("br"):
        tag.replace_with("\n")

    # Block-level elements: append a newline after their content
    for tag in soup.find_all(["p", "div", "li"]):
        tag.append(soup.new_string("\n"))

    # <ul> / <ol> — add a trailing newline after the list
    for tag in soup.find_all(["ul", "ol"]):
        tag.append(soup.new_string("\n"))

    text = soup.get_text(separator="")

    # Normalise whitespace
    text = re.sub(r"[ \t]+", " ", text)       # collapse horizontal whitespace
    text = re.sub(r" \n", "\n", text)          # trailing space before newline
    text = re.sub(r"\n ", "\n", text)          # leading space after newline
    text = re.sub(r"\n{3,}", "\n\n", text)     # max two consecutive newlines
    text = text.strip()

    return text if text else None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean HTML bodies and split system events")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "messages_clean.parquet").exists() and not args.force:
        answer = input("outputs/03_clean/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 03_clean ===\n")

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run 02_extract.py first.")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    input_count = len(df)
    print(f"Loaded {input_count:,} messages.\n")

    # ── Duplicates ────────────────────────────────────────────────────────────
    n_dupes = df.duplicated(subset=["message_id"]).sum()
    if n_dupes:
        print(f"  Dropping {n_dupes} duplicate message_id rows.")
        df = df.drop_duplicates(subset=["message_id"], keep="first")

    # ── Separate system events ────────────────────────────────────────────────
    is_system = df["message_type"] == "systemEventMessage"
    system_df = df[is_system].copy()
    messages_df = df[~is_system].copy()

    print(f"  System events separated: {len(system_df):,}")
    print(f"  Messages remaining:      {len(messages_df):,}\n")

    # ── HTML → text ───────────────────────────────────────────────────────────
    print("Cleaning HTML bodies...")
    tqdm.pandas(desc="  Cleaning HTML", unit="msg")
    messages_df["text"] = messages_df["text_html"].progress_apply(_html_to_text)

    # ── Cleaning diagnostics ──────────────────────────────────────────────────
    null_html = messages_df["text_html"].isna().sum()
    deleted = messages_df["is_deleted"].sum()
    empty_after_clean = (
        messages_df["text"].isna() & ~messages_df["is_deleted"]
    ).sum()

    print(f"  Deleted messages (body expected empty): {deleted:,}")
    print(f"  Null HTML body:                         {null_html:,}")
    print(f"  Empty text after cleaning (non-deleted):{empty_after_clean:,}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    messages_df.to_parquet(OUTPUT_DIR / "messages_clean.parquet", index=False)
    system_df.to_parquet(OUTPUT_DIR / "system_events.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    msg_type_counts = messages_df["message_type"].value_counts().to_dict()

    # Text length distribution (non-null)
    text_lengths = messages_df["text"].dropna().str.len()
    length_stats: dict = {}
    if not text_lengths.empty:
        desc = text_lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        length_stats = {k: round(float(v), 1) for k, v in desc.items()}

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "input_message_count": input_count,
        "duplicate_rows_dropped": int(n_dupes),
        "system_events_separated": len(system_df),
        "output_message_count": len(messages_df),
        "deleted_messages": int(deleted),
        "null_html_body": int(null_html),
        "empty_text_after_cleaning": int(empty_after_clean),
        "by_message_type": {str(k): int(v) for k, v in msg_type_counts.items()},
        "text_length_distribution": length_stats,
    }
    (OUTPUT_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\n--- Cleaning summary ---")
    print(f"  Input messages:          {input_count:,}")
    print(f"  System events file:      {len(system_df):,}")
    print(f"  Clean messages output:   {len(messages_df):,}")
    if n_dupes:
        print(f"  Duplicates dropped:      {n_dupes:,}")

    if not text_lengths.empty:
        print(f"\n  Text length (chars):")
        print(f"    median  {length_stats.get('50%', 0):.0f}")
        print(f"    p95     {length_stats.get('95%', 0):.0f}")
        print(f"    max     {length_stats.get('max', 0):.0f}")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_clean.parquet  clean messages with text populated")
    print("  system_events.parquet   system event messages")
    print("  cleaning_report.json    cleaning summary")
    print(f"\nNext: python message_miner/04_edge_cases.py")


if __name__ == "__main__":
    main()
