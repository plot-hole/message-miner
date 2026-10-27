"""
02_extract.py — Message Extraction

Reads outputs/01_connect/config_snapshot.json for the list of included chats.
Paginates through /me/chats/{id}/messages for each included chat.
Writes raw API responses and flattened Parquet tables.

Outputs:
  outputs/02_extract/raw/{chat_id}.json    raw API response per chat
  outputs/02_extract/messages.parquet      one row per message
  outputs/02_extract/chats.parquet         one row per chat (with computed stats)
  outputs/02_extract/extraction_report.json  run summary

Run from project root:
    python message_miner/02_extract.py [--force]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from message_miner.helpers.auth import get_access_token
from message_miner.helpers.graph import graph_get_all

CONFIG_PATH = Path("outputs/01_connect/config_snapshot.json")
OUTPUT_DIR = Path("outputs/02_extract")
RAW_DIR = OUTPUT_DIR / "raw"


# ── Filename sanitization ─────────────────────────────────────────────────────

def _safe_filename(chat_id: str) -> str:
    """Replace characters invalid in filenames with underscores."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", chat_id)


# ── Message parsing ───────────────────────────────────────────────────────────

def _parse_message(msg: dict, chat_id: str, msg_index: int) -> dict:
    """Flatten one raw Graph API message object into a schema-aligned record."""
    sender = ((msg.get("from") or {}).get("user")) or {}
    body = msg.get("body") or {}
    attachments = msg.get("attachments") or []
    mentions = msg.get("mentions") or []
    reactions = msg.get("reactions") or []

    mention_ids = [
        ((m.get("mentioned") or {}).get("user") or {}).get("id")
        for m in mentions
    ]
    mention_ids = [mid for mid in mention_ids if mid]

    attachment_types = [
        a.get("contentType") for a in attachments if a.get("contentType")
    ]

    return {
        "chat_id": chat_id,
        "message_id": msg.get("id"),
        "msg_index": msg_index,
        "sender_id": sender.get("id"),
        "sender_name": sender.get("displayName"),
        "text": None,                          # populated by 03_clean
        "text_html": body.get("content"),
        "timestamp": msg.get("createdDateTime"),
        "last_edited": msg.get("lastEditedDateTime"),
        "message_type": msg.get("messageType") or "unknownFutureValue",
        "importance": msg.get("importance") or "normal",
        "has_attachments": len(attachments) > 0,
        "has_mentions": len(mentions) > 0,
        "mention_ids": ",".join(mention_ids) if mention_ids else None,
        "attachment_types": ",".join(attachment_types) if attachment_types else None,
        "reaction_count": len(reactions),
        "is_deleted": msg.get("deletedDateTime") is not None,
        "is_edited": msg.get("lastEditedDateTime") is not None,
    }


# ── Chat record ───────────────────────────────────────────────────────────────

def _build_chat_record(chat_entry: dict, messages: list[dict]) -> dict:
    """Build a chats.parquet row from a config snapshot entry + its parsed messages."""
    timestamps = sorted(
        pd.to_datetime(m["timestamp"], utc=True)
        for m in messages
        if m.get("timestamp")
    )

    first_ts = timestamps[0] if timestamps else None
    last_ts = timestamps[-1] if timestamps else None

    duration_minutes = None
    if first_ts is not None and last_ts is not None:
        duration_minutes = float((last_ts - first_ts).total_seconds() / 60.0)

    return {
        "chat_id": chat_entry["chat_id"],
        "chat_type": chat_entry.get("chat_type"),
        "topic": chat_entry.get("topic"),
        "created_at": chat_entry.get("created_at"),
        "last_updated": chat_entry.get("last_updated"),
        "member_count": chat_entry.get("member_count") or 0,
        "member_names": ",".join(chat_entry.get("member_names") or []) or None,
        "member_ids": ",".join(chat_entry.get("member_ids") or []) or None,
        "msg_count": len(messages),
        "duration_minutes": duration_minutes,
        "hour_of_day": first_ts.hour if first_ts is not None else None,
        "day_of_week": first_ts.dayofweek if first_ts is not None else None,
    }


# ── dtype coercion ────────────────────────────────────────────────────────────

def _apply_dtypes(
    messages_df: pd.DataFrame, chats_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # messages
    messages_df["msg_index"] = messages_df["msg_index"].astype("int32")
    messages_df["reaction_count"] = messages_df["reaction_count"].astype("int32")
    for col in ("has_attachments", "has_mentions", "is_deleted", "is_edited"):
        messages_df[col] = messages_df[col].astype(bool)
    for col in ("message_type", "importance"):
        messages_df[col] = messages_df[col].astype("category")
    for col in ("timestamp", "last_edited"):
        messages_df[col] = pd.to_datetime(messages_df[col], utc=True, errors="coerce")

    # chats
    chats_df["member_count"] = chats_df["member_count"].astype("int32")
    chats_df["msg_count"] = chats_df["msg_count"].astype("int32")
    chats_df["chat_type"] = chats_df["chat_type"].astype("category")
    for col in ("created_at", "last_updated"):
        chats_df[col] = pd.to_datetime(chats_df[col], utc=True, errors="coerce")
    # Nullable integer types for columns that can be null (chats with no messages)
    chats_df["hour_of_day"] = chats_df["hour_of_day"].astype("Int8")
    chats_df["day_of_week"] = chats_df["day_of_week"].astype("Int8")
    chats_df["duration_minutes"] = chats_df["duration_minutes"].astype("float32")

    return messages_df, chats_df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Teams messages via Graph API")
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output without prompting"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only extract the first N included chats (for testing)"
    )
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only extract messages since this date (YYYY-MM-DD). "
             "Stops API pagination early once older messages are reached."
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent chat extractions (default: 5). "
             "Higher values are faster but may hit Graph API rate limits."
    )
    args = parser.parse_args()

    if (OUTPUT_DIR / "messages.parquet").exists() and not args.force:
        answer = input(
            "outputs/02_extract/ already has output. Overwrite? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 02_extract ===\n")

    # ── Load config snapshot ──────────────────────────────────────────────────
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Run 01_connect.py first.")
        sys.exit(1)

    snapshot = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    all_chats = snapshot["chats"]
    included = [c for c in all_chats if c.get("include", True)]
    excluded_count = len(all_chats) - len(included)

    if args.limit:
        included = included[: args.limit]
        print(f"Config: {len(included)} chats included, {excluded_count} excluded (--limit {args.limit} applied).\n")
    else:
        print(f"Config: {len(included)} chats included, {excluded_count} excluded.\n")

    # ── Since filter ─────────────────────────────────────────────────────────
    since_iso = None
    if args.since:
        since_iso = f"{args.since}T00:00:00Z"
        print(f"Filtering to messages since {args.since} (pagination stops early).\n")

    # ── Auth ──────────────────────────────────────────────────────────────────
    token = get_access_token()
    print("Authenticated.\n")

    # ── Extract (concurrent) ────────────────────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _extract_one(chat_entry: dict) -> dict:
        """Extract one chat. Runs in a thread pool worker."""
        chat_id = chat_entry["chat_id"]
        label = chat_entry.get("topic") or chat_id[:24]

        try:
            raw_messages = graph_get_all(
                token,
                f"/me/chats/{chat_id}/messages",
                {"$top": "50"},
                since=since_iso,
            )
        except Exception as exc:
            return {"chat_entry": chat_entry, "label": label, "error": str(exc)}

        # Write raw JSON (one file per chat — no contention)
        raw_path = RAW_DIR / f"{_safe_filename(chat_id)}.json"
        raw_path.write_text(
            json.dumps(raw_messages, indent=2, default=str),
            encoding="utf-8",
        )

        raw_messages.sort(key=lambda m: m.get("createdDateTime") or "")
        parsed = [_parse_message(m, chat_id, i) for i, m in enumerate(raw_messages)]

        return {
            "chat_entry": chat_entry,
            "label": label,
            "error": None,
            "parsed": parsed,
            "msg_count": len(raw_messages),
        }

    all_messages: list[dict] = []
    all_chat_records: list[dict] = []
    failed_chats: list[dict] = []
    total_msg_count = 0

    concurrency = args.concurrency
    print(f"Extracting with {concurrency} concurrent workers...\n")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(_extract_one, c): c for c in included}

        for future in tqdm(
            as_completed(futures), total=len(included),
            desc="Extracting chats", unit="chat",
        ):
            result = future.result()
            if result["error"]:
                tqdm.write(f"  [skip] {result['label']}: {result['error']}")
                failed_chats.append({
                    "chat_id": result["chat_entry"]["chat_id"],
                    "error": result["error"],
                })
            else:
                total_msg_count += result["msg_count"]
                tqdm.write(
                    f"  > {result['label']}  "
                    f"({result['msg_count']:,} msgs, {total_msg_count:,} total)"
                )
                all_messages.extend(result["parsed"])
                all_chat_records.append(
                    _build_chat_record(result["chat_entry"], result["parsed"])
                )

    print(
        f"\nExtracted {len(all_messages):,} messages "
        f"from {len(all_chat_records)} chats."
    )
    if failed_chats:
        print(f"  {len(failed_chats)} chats failed — see extraction_report.json")

    # ── Build DataFrames ──────────────────────────────────────────────────────
    messages_df = pd.DataFrame(all_messages)
    chats_df = pd.DataFrame(all_chat_records)

    messages_df, chats_df = _apply_dtypes(messages_df, chats_df)

    # ── Write Parquet ─────────────────────────────────────────────────────────
    messages_df.to_parquet(OUTPUT_DIR / "messages.parquet", index=False)
    chats_df.to_parquet(OUTPUT_DIR / "chats.parquet", index=False)

    # ── Write report ──────────────────────────────────────────────────────────
    type_counts = chats_df["chat_type"].value_counts().to_dict()
    msg_type_counts = messages_df["message_type"].value_counts().to_dict()

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "since_filter": args.since,
        "chats_included": len(included),
        "chats_excluded": excluded_count,
        "chats_extracted": len(all_chat_records),
        "chats_failed": len(failed_chats),
        "failed_chats": failed_chats,
        "total_messages": len(all_messages),
        "by_chat_type": {str(k): int(v) for k, v in type_counts.items()},
        "by_message_type": {str(k): int(v) for k, v in msg_type_counts.items()},
    }
    (OUTPUT_DIR / "extraction_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n--- Extraction summary ---")
    print(f"  Total messages:  {len(all_messages):,}")
    for chat_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {str(chat_type):<22} {count} chats")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  raw/                    one JSON file per chat")
    print("  messages.parquet        all messages (text column filled by 03_clean)")
    print("  chats.parquet           one row per chat with stats")
    print("  extraction_report.json  run summary")
    print(f"\nNext: python message_miner/03_clean.py")


if __name__ == "__main__":
    main()
