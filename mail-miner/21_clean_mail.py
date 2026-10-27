"""
21_clean_mail.py — Email Cleaning

Reads messages_mail.parquet.
Converts HTML to plain text, removes signatures and quoted/forwarded content.

New columns added:
  body_clean          HTML stripped to plain text, whitespace normalized
  body_text           Final cleaned text: signature and quoted content removed
  has_quoted_content  True if quoted/forwarded content was detected and removed
  is_empty_body       True if body_text is empty after cleaning
  signature_removed   True if a signature was detected and removed

Reads:
  outputs/20_extract_mail/messages_mail.parquet

Writes:
  outputs/21_clean_mail/messages_mail_clean.parquet
  outputs/21_clean_mail/cleaning_report.json

Run from project root:
    python mail-miner/21_clean_mail.py [--force]
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

INPUT_PATH = Path("outputs/20_extract_mail/messages_mail.parquet")
OUTPUT_DIR = Path("outputs/21_clean_mail")

# ── HTML → plain text ─────────────────────────────────────────────────────────

def _html_to_text(html: str | None) -> str | None:
    """Strip HTML to plain text, preserving line break structure."""
    if not html or not html.strip():
        return None

    soup = BeautifulSoup(html, "html.parser")

    # Block-level tags → newlines
    for tag in soup.find_all(["br"]):
        tag.replace_with("\n")
    for tag in soup.find_all(["p", "div", "tr"]):
        tag.append(soup.new_string("\n"))
    for tag in soup.find_all(["li"]):
        tag.insert(0, soup.new_string("• "))
        tag.append(soup.new_string("\n"))

    text = soup.get_text(separator="")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n", "\n", text)
    text = re.sub(r"\n ", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return text if text else None


# ── Signature removal ─────────────────────────────────────────────────────────

# Patterns that mark the start of a signature block (matched against stripped lines)
_SIG_DELIMITER_RE = re.compile(
    r"^(--|---+|___+)$"
)
_SIG_MOBILE_RE = re.compile(
    r"^(sent from my (iphone|ipad|android|mobile)|"
    r"get outlook for (ios|android)|"
    r"sent from outlook)$",
    re.IGNORECASE,
)
# Closing salutations followed by a short name line
_SIG_CLOSING_RE = re.compile(
    r"^(best,?|thanks,?|thank you,?|regards,?|kind regards,?|"
    r"cheers,?|sincerely,?|warm regards,?|best regards,?|"
    r"many thanks,?|cordially,?|respectfully,?)$",
    re.IGNORECASE,
)


def _remove_signature(text: str) -> tuple[str, bool]:
    """
    Strip signature from plain text. Returns (cleaned_text, signature_removed).

    Scans from the bottom up, finds the first signature marker, and trims
    everything from that line downward.
    """
    lines = text.splitlines()
    cut_at = None

    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if (
            _SIG_DELIMITER_RE.match(stripped)
            or _SIG_MOBILE_RE.match(stripped)
            or _SIG_CLOSING_RE.match(stripped)
        ):
            cut_at = i
            # Keep scanning upward to find the topmost signature marker
            # (don't break — we want the first/highest occurrence in a run)
            break

    if cut_at is not None:
        result = "\n".join(lines[:cut_at]).strip()
        return result, True

    return text, False


# ── Quoted / forwarded content removal ───────────────────────────────────────

_QUOTE_PATTERNS = [
    # "On Mon, Jan 1, 2024, John wrote:" — standard reply header
    re.compile(r"^On .{5,80}wrote:$", re.IGNORECASE),
    # Outlook-style "From: Name / Sent: date / To: recipient"
    re.compile(r"^From:\s+\S", re.IGNORECASE),
    # Hard separator lines
    re.compile(r"^-{5,}(original message|-+forwarded message-+|-+)\s*$", re.IGNORECASE),
    # Outlook horizontal rule (underscores, 20+)
    re.compile(r"^_{20,}$"),
    # Lines starting with > (standard quoting)
    re.compile(r"^>"),
    # Forwarded message headers
    re.compile(r"^(begin forwarded message|------+\s*forwarded message)", re.IGNORECASE),
]


def _remove_quoted_content(text: str) -> tuple[str, bool]:
    """
    Remove quoted/forwarded content. Returns (cleaned_text, had_quoted_content).
    """
    lines = text.splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern in _QUOTE_PATTERNS:
            if pattern.match(stripped):
                result = "\n".join(lines[:i]).strip()
                return result, True

    return text, False


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean email HTML bodies")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "messages_mail_clean.parquet").exists() and not args.force:
        answer = input("outputs/21_clean_mail/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 21_clean_mail ===\n")

    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run 20_extract_mail.py first.")
        sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded {len(df):,} messages.")

    n_dupes = df.duplicated(subset=["message_id"]).sum()
    if n_dupes:
        print(f"  Dropping {n_dupes:,} duplicate message_id rows (Graph API pagination artifact).")
        df = df.drop_duplicates(subset=["message_id"], keep="first")
    print()

    avg_before = df["body_html"].dropna().str.len().mean()

    # ── HTML → plain text ────────────────────────────────────────────────────
    print("Converting HTML to plain text...")
    tqdm.pandas(desc="  HTML → text", unit="msg")
    df["body_clean"] = df["body_html"].progress_apply(_html_to_text)

    # ── Signature removal ─────────────────────────────────────────────────────
    print("Removing signatures...")
    body_text_list = []
    sig_removed_list = []
    for text in tqdm(df["body_clean"], desc="  Signatures", unit="msg"):
        if text:
            cleaned, removed = _remove_signature(text)
            body_text_list.append(cleaned)
            sig_removed_list.append(removed)
        else:
            body_text_list.append(text)
            sig_removed_list.append(False)

    df["body_text"] = body_text_list
    df["signature_removed"] = sig_removed_list

    # ── Quoted content removal ────────────────────────────────────────────────
    print("Removing quoted/forwarded content...")
    body_text2_list = []
    quoted_list = []
    for text in tqdm(df["body_text"], desc="  Quoted text", unit="msg"):
        if text:
            cleaned, had_quoted = _remove_quoted_content(text)
            body_text2_list.append(cleaned)
            quoted_list.append(had_quoted)
        else:
            body_text2_list.append(text)
            quoted_list.append(False)

    df["body_text"] = body_text2_list
    df["has_quoted_content"] = quoted_list

    # ── Empty body flag ───────────────────────────────────────────────────────
    df["is_empty_body"] = df["body_text"].isna() | (df["body_text"].str.strip() == "")

    # ── Write ─────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_DIR / "messages_mail_clean.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    avg_after = df["body_text"].dropna().str.len().mean()
    sigs_removed = int(df["signature_removed"].sum())
    quoted_removed = int(df["has_quoted_content"].sum())
    empty_after = int(df["is_empty_body"].sum())

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_messages": len(df),
        "signatures_removed": sigs_removed,
        "quoted_content_removed": quoted_removed,
        "empty_after_cleaning": empty_after,
        "avg_body_length_before": round(float(avg_before), 1) if avg_before and not pd.isna(avg_before) else 0,
        "avg_body_length_after": round(float(avg_after), 1) if avg_after and not pd.isna(avg_after) else 0,
    }
    (OUTPUT_DIR / "cleaning_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\n--- Cleaning summary ---")
    print(f"  Total messages:          {len(df):,}")
    print(f"  Signatures removed:      {sigs_removed:,}  ({100*sigs_removed/len(df):.1f}%)")
    print(f"  Quoted content removed:  {quoted_removed:,}  ({100*quoted_removed/len(df):.1f}%)")
    print(f"  Empty after cleaning:    {empty_after:,}  ({100*empty_after/len(df):.1f}%)")
    print(f"  Avg length before:       {report['avg_body_length_before']:,.0f} chars")
    print(f"  Avg length after:        {report['avg_body_length_after']:,.0f} chars")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_mail_clean.parquet")
    print("  cleaning_report.json")
    print(f"\nNext: python mail-miner/22_edge_cases_mail.py")


if __name__ == "__main__":
    main()
