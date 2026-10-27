"""
22_edge_cases_mail.py — Email Edge Case Flagging

Adds boolean flag columns to classify non-human / automated / noise emails.
Nothing is dropped — only flagged so downstream steps can filter as needed.

Flags added:
  is_auto_reply           Out-of-office / automatic reply
  is_calendar             Meeting accept/decline/tentative/cancel notifications
  is_newsletter           Marketing, digest, noreply senders, List-Unsubscribe header
  is_notification         Automated system notifications (Jira, GitHub, ServiceNow, etc.)
  is_read_receipt         Delivery / read receipt messages
  is_encrypted            Protected/encrypted message placeholder
  is_attachment_only      Empty body + has_attachments
  is_distribution_list    Large recipient count (>= 20 To+CC recipients)
  is_draft                Draft messages
  is_deleted              Emails in Deleted Items / Recoverable Items
  is_external             Sender domain differs from authenticated user's domain

Compound flag:
  is_human_mail           True if none of: is_auto_reply, is_calendar, is_newsletter,
                          is_notification, is_read_receipt, is_encrypted, is_draft, is_deleted

Reads:
  outputs/21_clean_mail/messages_mail_clean.parquet
  outputs/20_extract_mail/extraction_report.json    (for user domain)

Writes:
  outputs/22_edge_cases_mail/messages_mail_flagged.parquet
  outputs/22_edge_cases_mail/edge_case_report.json

Run from project root:
    python mail-miner/22_edge_cases_mail.py [--force] [--dist-list-threshold N]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

INPUT_PATH = Path("outputs/21_clean_mail/messages_mail_clean.parquet")
EXTRACTION_REPORT_PATH = Path("outputs/20_extract_mail/extraction_report.json")
OUTPUT_DIR = Path("outputs/22_edge_cases_mail")

_DEFAULT_DIST_LIST_THRESHOLD = 20

# ── Flag helpers ──────────────────────────────────────────────────────────────

# is_auto_reply: subject-based only. "noreply" in the sender is NOT an auto-reply
# signal — those are system notifications. Auto-replies come from real user addresses
# with specific subject prefixes set by mail clients / Exchange.
_AUTO_REPLY_SUBJECT_RE = re.compile(
    r"^(automatic reply|out of office|auto:|autoreply:|away:)",
    re.IGNORECASE,
)

_CALENDAR_SUBJECT_RE = re.compile(
    r"^(accepted:|tentative:|declined:|canceled:|cancelled:)",
    re.IGNORECASE,
)
_CALENDAR_BODY_RE = re.compile(
    r"When:.*Where:",
    re.IGNORECASE | re.DOTALL,
)

# is_newsletter: genuine marketing/digest content. Does NOT include generic
# "noreply" senders — those belong in is_notification.
_NEWSLETTER_SENDER_RE = re.compile(
    r"(newsletter@|newsletters@|digest@|marketing@|promotions@|"
    r"campaign@|announce@|announcements@|no-?reply@.*newsletter|"
    r"subscriptions?@|unsubscribe@)",
    re.IGNORECASE,
)

# is_notification: all automated system senders, including any noreply/donotreply
# address, plus common IT and devops tools.
_NOTIFICATION_SENDER_RE = re.compile(
    r"(no-?reply|noreply|do-?not-?reply|donotreply|mailer-daemon|postmaster|"
    # IT / devops
    r"@jira\.|jira@|@atlassian\.|servicenow|confluence|"
    r"github\.com|gitlab\.com|azure-noreply|"
    r"noreply@microsoft\.com|teams@microsoft\.com|no-reply@microsoft\.com|"
    r"MSSecurity-noreply)",
    re.IGNORECASE,
)
_NOTIFICATION_SUBJECT_RE = re.compile(
    r"(\[jira\]|\[confluence\]|build #|pr #|pull request|"
    r"alert:|warning:|deployment|pipeline failed|"
    r"\[github\]|\[gitlab\])",
    re.IGNORECASE,
)

_ENCRYPTED_BODY_RE = re.compile(
    r"(encrypted message|protected message|"
    r"you have received a (protected|encrypted)|"
    r"to read this message)",
    re.IGNORECASE,
)

_READ_RECEIPT_SUBJECT_RE = re.compile(
    r"^read:|delivery notification|delivery status|"
    r"undeliverable:|undelivered mail|failed to deliver",
    re.IGNORECASE,
)


def _flag_auto_reply(df: pd.DataFrame) -> pd.Series:
    # Subject-based only — "noreply" senders are notifications, not auto-replies.
    return df["subject"].fillna("").str.contains(_AUTO_REPLY_SUBJECT_RE, regex=True)


def _flag_calendar(df: pd.DataFrame) -> pd.Series:
    subject_match = df["subject"].fillna("").str.contains(_CALENDAR_SUBJECT_RE, regex=True)
    body_match = df["body_clean"].fillna("").str.contains(_CALENDAR_BODY_RE, regex=True)
    html_schema = df["body_html"].fillna("").str.contains(
        "itemtype=\"http://schema.org/Event\"", regex=False
    )
    return subject_match | body_match | html_schema


def _flag_newsletter(df: pd.DataFrame) -> pd.Series:
    return df["sender_email"].fillna("").str.contains(_NEWSLETTER_SENDER_RE, regex=True)


def _flag_notification(df: pd.DataFrame) -> pd.Series:
    sender_match = df["sender_email"].fillna("").str.contains(_NOTIFICATION_SENDER_RE, regex=True)
    subject_match = df["subject"].fillna("").str.contains(_NOTIFICATION_SUBJECT_RE, regex=True)
    return sender_match | subject_match


def _flag_read_receipt(df: pd.DataFrame) -> pd.Series:
    return df["subject"].fillna("").str.contains(_READ_RECEIPT_SUBJECT_RE, regex=True)


def _flag_encrypted(df: pd.DataFrame) -> pd.Series:
    return df["body_clean"].fillna("").str.contains(_ENCRYPTED_BODY_RE, regex=True)


def _user_domain(user_email: str) -> str:
    """Extract domain from user email address."""
    if "@" in user_email:
        return user_email.split("@", 1)[1].lower()
    return ""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Flag edge case emails")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    parser.add_argument(
        "--dist-list-threshold", type=int, default=_DEFAULT_DIST_LIST_THRESHOLD,
        help=f"Min To+CC recipient count to flag as distribution list (default: {_DEFAULT_DIST_LIST_THRESHOLD})"
    )
    args = parser.parse_args()

    if (OUTPUT_DIR / "messages_mail_flagged.parquet").exists() and not args.force:
        answer = input("outputs/22_edge_cases_mail/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 22_edge_cases_mail ===\n")

    for path in (INPUT_PATH, EXTRACTION_REPORT_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(INPUT_PATH)
    extraction_report = json.loads(EXTRACTION_REPORT_PATH.read_text(encoding="utf-8"))
    user_email = extraction_report.get("user_email", "")
    domain = _user_domain(user_email)

    print(f"Loaded {len(df):,} messages.\n")
    print(f"User domain: {domain}\n")

    # ── Apply flags ───────────────────────────────────────────────────────────
    print("Applying flags...")

    df["is_auto_reply"] = _flag_auto_reply(df)
    df["is_calendar"] = _flag_calendar(df)
    df["is_newsletter"] = _flag_newsletter(df)
    df["is_notification"] = _flag_notification(df)
    df["is_read_receipt"] = _flag_read_receipt(df)
    df["is_encrypted"] = _flag_encrypted(df)
    df["is_attachment_only"] = df["is_empty_body"].fillna(False) & df["has_attachments"].fillna(False)
    df["is_distribution_list"] = (
        (df["to_count"].fillna(0) + df["cc_count"].fillna(0)) >= args.dist_list_threshold
    )
    # is_draft already exists in schema from extraction
    if "is_draft" not in df.columns:
        df["is_draft"] = False
    df["is_deleted"] = df["folder_name"].astype(str).isin(["Deleted Items", "Recoverable Items"])

    if domain:
        df["is_external"] = ~df["sender_email"].fillna("").str.lower().str.endswith(f"@{domain}")
    else:
        df["is_external"] = False

    # ── Compound flag ─────────────────────────────────────────────────────────
    noise_flags = [
        "is_auto_reply", "is_calendar", "is_newsletter", "is_notification",
        "is_read_receipt", "is_encrypted", "is_draft",
    ]
    df["is_human_mail"] = ~df[noise_flags].any(axis=1)

    # ── Write ─────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_DIR / "messages_mail_flagged.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    flag_cols = [
        "is_auto_reply", "is_calendar", "is_newsletter", "is_notification",
        "is_read_receipt", "is_encrypted", "is_attachment_only",
        "is_distribution_list", "is_draft", "is_deleted", "is_external",
    ]
    flag_counts = {col: int(df[col].sum()) for col in flag_cols}
    human_count = int(df["is_human_mail"].sum())
    total = len(df)

    # Top notification senders
    notification_df = df[df["is_notification"] | df["is_newsletter"] | df["is_auto_reply"]]
    top_senders = (
        notification_df["sender_email"]
        .dropna()
        .value_counts()
        .head(10)
        .reset_index()
        .rename(columns={"sender_email": "sender", "count": "count"})
        .to_dict(orient="records")
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_messages": total,
        "flag_counts": flag_counts,
        "human_mail_count": human_count,
        "human_mail_pct": round(100 * human_count / total, 1) if total else 0,
        "dist_list_threshold": args.dist_list_threshold,
        "top_notification_senders": top_senders,
    }
    (OUTPUT_DIR / "edge_case_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("--- Edge case counts ---")
    for col, count in flag_counts.items():
        pct = 100 * count / total if total else 0
        print(f"  {col:<26} {count:>7,}  ({pct:.1f}%)")

    print(f"\n  Human mail (is_human_mail):  {human_count:>7,}  ({report['human_mail_pct']:.1f}%)")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_mail_flagged.parquet")
    print("  edge_case_report.json")
    print(f"\nNext: python mail-miner/23_enrich_mail.py")


if __name__ == "__main__":
    main()
