"""
20_extract_mail.py — Email Extraction

Pulls all emails from the Microsoft Graph API and writes them to Parquet.
Uses the same MSAL auth helper as the chat pipeline.

Resume support: if a previous run was interrupted, re-running the script
automatically resumes from the last completed page. Use --force to restart
from scratch.

Outputs:
  outputs/20_extract_mail/messages_mail.parquet     one row per email
  outputs/20_extract_mail/folders.parquet           folder ID → name mapping
  outputs/20_extract_mail/raw/                      raw JSON per page
  outputs/20_extract_mail/extraction_report.json    stats

Run from project root:
    python mail-miner/20_extract_mail.py [--force]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from message_miner.helpers.auth import get_access_token
from message_miner.helpers.graph import graph_get

OUTPUT_DIR = Path("outputs/20_extract_mail")
RAW_DIR = OUTPUT_DIR / "raw"
PROGRESS_PATH = OUTPUT_DIR / "progress.json"


# ── Resume helpers ────────────────────────────────────────────────────────────

def _save_progress(page_num: int, next_url: str | None, messages_fetched: int) -> None:
    """Persist pagination state after each completed page."""
    PROGRESS_PATH.write_text(
        json.dumps({
            "page_num": page_num,
            "next_url": next_url,
            "messages_fetched": messages_fetched,
        }),
        encoding="utf-8",
    )


def _load_progress() -> dict | None:
    if PROGRESS_PATH.exists():
        return json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    return None


def _load_raw_pages() -> list[dict]:
    """Read all previously saved raw page files, sorted by page number."""
    raw_files = sorted(RAW_DIR.glob("page_*.json"), key=lambda p: p.name)
    messages: list[dict] = []
    for f in raw_files:
        messages.extend(json.loads(f.read_text(encoding="utf-8")))
    return messages


def _graph_get_refreshing(token: str, url: str, params=None) -> tuple[dict, str]:
    """
    Call graph_get, refreshing the token on 401 and retrying once.
    Returns (response_data, current_token) so the caller can track the live token.
    """
    try:
        return graph_get(token, url, params), token
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            print("  [auth] Token expired — refreshing and retrying...")
            token = get_access_token()
            return graph_get(token, url, params), token
        raise

_SELECT = ",".join([
    "id", "subject", "bodyPreview", "body",
    "sender", "from", "toRecipients", "ccRecipients", "bccRecipients",
    "receivedDateTime", "sentDateTime", "hasAttachments", "importance",
    "conversationId", "conversationIndex", "isRead", "isDraft", "flag",
    "internetMessageHeaders", "categories", "parentFolderId",
    "inferenceClassification",
])


# ── Folder fetch ──────────────────────────────────────────────────────────────

def _fetch_folders(token: str) -> dict[str, str]:
    """Return a mapping of folder_id → folder_name by paging /me/mailFolders."""
    folder_map: dict[str, str] = {}
    data = graph_get(token, "/me/mailFolders", {"$top": "100"})
    for folder in data.get("value", []):
        folder_map[folder["id"]] = folder.get("displayName", folder["id"])
    while "@odata.nextLink" in data:
        data = graph_get(token, data["@odata.nextLink"])
        for folder in data.get("value", []):
            folder_map[folder["id"]] = folder.get("displayName", folder["id"])
    return folder_map


# ── Message parsing ───────────────────────────────────────────────────────────

def _extract_header(headers: list[dict], name: str) -> str | None:
    """Find a header by name (case-insensitive) and return its value."""
    if not headers:
        return None
    name_lower = name.lower()
    for h in headers:
        if h.get("name", "").lower() == name_lower:
            return h.get("value")
    return None


def _parse_recipients(recipients: list[dict]) -> str:
    """Serialize a list of {emailAddress: {name, address}} to JSON."""
    parsed = [
        {
            "name": r.get("emailAddress", {}).get("name"),
            "email": r.get("emailAddress", {}).get("address"),
        }
        for r in (recipients or [])
    ]
    return json.dumps(parsed)


def _parse_message(msg: dict, folder_map: dict[str, str]) -> dict:
    """Flatten one raw Graph API message object into a schema-aligned record."""
    from_addr = (msg.get("from") or {}).get("emailAddress") or {}
    to_recipients = msg.get("toRecipients") or []
    cc_recipients = msg.get("ccRecipients") or []
    bcc_recipients = msg.get("bccRecipients") or []
    body = msg.get("body") or {}
    flag = msg.get("flag") or {}
    headers = msg.get("internetMessageHeaders") or []
    categories = msg.get("categories") or []
    folder_id = msg.get("parentFolderId")

    return {
        "message_id": msg.get("id"),
        "conversation_id": msg.get("conversationId"),
        "subject": msg.get("subject"),
        "body_html": body.get("content"),
        "body_preview": msg.get("bodyPreview"),
        "sender_email": from_addr.get("address"),
        "sender_name": from_addr.get("name"),
        "to_recipients": _parse_recipients(to_recipients),
        "cc_recipients": _parse_recipients(cc_recipients),
        "bcc_recipients": _parse_recipients(bcc_recipients),
        "to_count": len(to_recipients),
        "cc_count": len(cc_recipients),
        "received_datetime": msg.get("receivedDateTime"),
        "sent_datetime": msg.get("sentDateTime"),
        "has_attachments": bool(msg.get("hasAttachments")),
        "importance": msg.get("importance") or "normal",
        "is_read": bool(msg.get("isRead")),
        "is_draft": bool(msg.get("isDraft")),
        "folder_id": folder_id,
        "folder_name": folder_map.get(folder_id) if folder_id else None,
        "inference_classification": msg.get("inferenceClassification"),
        "categories": json.dumps(categories),
        "flag_status": flag.get("flagStatus") or "notFlagged",
        "internet_message_id": _extract_header(headers, "Message-ID"),
        "in_reply_to": _extract_header(headers, "In-Reply-To"),
        "references": _extract_header(headers, "References"),
    }


# ── dtype coercion ────────────────────────────────────────────────────────────

def _apply_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("received_datetime", "sent_datetime"):
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ("has_attachments", "is_read", "is_draft"):
        df[col] = df[col].astype(bool)
    for col in ("to_count", "cc_count"):
        df[col] = df[col].astype("int32")
    for col in ("importance", "inference_classification", "flag_status", "folder_name"):
        df[col] = df[col].astype("category")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract emails from Microsoft Graph API")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Resume / overwrite logic ──────────────────────────────────────────────
    progress = _load_progress()
    resuming = progress is not None and not args.force

    if args.force:
        if PROGRESS_PATH.exists():
            PROGRESS_PATH.unlink()
        for f in RAW_DIR.glob("page_*.json"):
            f.unlink()
    elif resuming:
        print(
            f"Resuming from page {progress['page_num']} "
            f"({progress['messages_fetched']:,} messages already fetched).\n"
            f"Delete outputs/20_extract_mail/progress.json and re-run with --force to restart.\n"
        )
    elif (OUTPUT_DIR / "messages_mail.parquet").exists():
        answer = input("outputs/20_extract_mail/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    print("=== message-miner: 20_extract_mail ===\n")

    # ── Auth ──────────────────────────────────────────────────────────────────
    token = get_access_token()
    print("Authenticated.\n")

    # ── User identity ─────────────────────────────────────────────────────────
    me = graph_get(token, "/me")
    user_email = me.get("mail") or me.get("userPrincipalName") or ""
    user_name = me.get("displayName") or ""
    print(f"User: {user_name} <{user_email}>\n")

    # ── Folder map ────────────────────────────────────────────────────────────
    print("Fetching mail folders...")
    folder_map = _fetch_folders(token)
    print(f"  {len(folder_map)} folders found.\n")

    # ── Paginate messages ─────────────────────────────────────────────────────
    print("Fetching emails...")

    if resuming:
        all_messages = _load_raw_pages()
        page_num = progress["page_num"]
        next_url: str | None = progress["next_url"]
        print(f"  Loaded {len(all_messages):,} messages from {page_num} previously saved pages.")
    else:
        all_messages = []
        page_num = 0
        next_url = "/me/messages"

    params: dict | None = None if resuming else {
        "$top": "200",
        "$orderby": "receivedDateTime desc",
        "$select": _SELECT,
    }

    while next_url is not None:
        data, token = _graph_get_refreshing(token, next_url, params)
        params = None  # params are baked into nextLink on subsequent pages

        page_items = data.get("value", [])
        all_messages.extend(page_items)
        page_num += 1

        # Save raw page
        raw_path = RAW_DIR / f"page_{page_num:04d}.json"
        raw_path.write_text(
            json.dumps(page_items, indent=2, default=str),
            encoding="utf-8",
        )

        next_url = data.get("@odata.nextLink")
        total = len(all_messages)

        # Persist progress after every page so a crash can be resumed
        _save_progress(page_num, next_url, total)

        if next_url:
            print(f"  [{total:,}] {total:,} messages")
        else:
            print(f"  [{total:,}] {total:,} messages — no more pages")

    # Extraction complete — remove progress file
    if PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    print(f"\nTotal: {len(all_messages):,} messages across {page_num} pages.\n")

    # ── Parse ─────────────────────────────────────────────────────────────────
    print("Parsing messages...")
    records = [_parse_message(msg, folder_map) for msg in all_messages]
    df = pd.DataFrame(records)
    df = _apply_dtypes(df)

    # Deduplicate — $skip-based pagination can return the same message twice
    # if the mailbox changes during a long extraction.
    total_messages_raw = len(df)
    n_dupes = df.duplicated(subset=["message_id"]).sum()
    if n_dupes:
        print(f"  Dropping {n_dupes:,} duplicate message_id rows (pagination artifact).")
        df = df.drop_duplicates(subset=["message_id"], keep="first")

    # ── Folder DataFrame ──────────────────────────────────────────────────────
    folders_df = pd.DataFrame(
        [{"folder_id": fid, "folder_name": fname} for fid, fname in folder_map.items()]
    )

    # ── Write outputs ─────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_DIR / "messages_mail.parquet", index=False)
    folders_df.to_parquet(OUTPUT_DIR / "folders.parquet", index=False)

    # ── Extraction report ─────────────────────────────────────────────────────
    date_first = None
    date_last = None
    if not df.empty and "received_datetime" in df.columns:
        valid_dates = df["received_datetime"].dropna()
        if not valid_dates.empty:
            date_first = str(valid_dates.min())
            date_last = str(valid_dates.max())

    folder_breakdown: dict[str, int] = {}
    if not df.empty and "folder_name" in df.columns:
        folder_breakdown = {
            str(k): int(v)
            for k, v in df["folder_name"].value_counts().items()
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_email": user_email,
        "user_name": user_name,
        "total_messages": len(df),
        "total_messages_raw": total_messages_raw,
        "duplicates_dropped": int(n_dupes),
        "date_first": date_first,
        "date_last": date_last,
        "folder_breakdown": folder_breakdown,
        "pages_fetched": page_num,
    }
    (OUTPUT_DIR / "extraction_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"--- Extraction summary ---")
    print(f"  Total messages:  {len(df):,}")
    print(f"  Date range:      {date_first} to {date_last}")
    print(f"  Pages fetched:   {page_num}")
    for folder_name, count in sorted(folder_breakdown.items(), key=lambda x: -x[1])[:10]:
        print(f"  {str(folder_name):<30} {count:,}")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  messages_mail.parquet     one row per email")
    print("  folders.parquet           folder ID → name mapping")
    print("  raw/                      raw JSON per page")
    print("  extraction_report.json    stats")
    print(f"\nNext: python mail-miner/21_clean_mail.py")


if __name__ == "__main__":
    main()
