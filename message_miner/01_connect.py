"""
01_connect.py — Auth + Chat Discovery

Authenticates as the signed-in user via browser-based Authorization Code flow,
discovers all accessible chats via /me/chats, and writes three output files:

  outputs/01_connect/chats_inventory.json   raw API response (all chats + members)
  outputs/01_connect/connection_report.json  run summary (counts by type)
  outputs/01_connect/config_snapshot.json   REVIEW THIS before running 02_extract

config_snapshot.json is the human-reviewable checkpoint between discovery and
extraction. Each chat entry has an "include" field set to true. Flip any to
false to exclude that chat from 02_extract.

Run from project root:
    python message_miner/01_connect.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is on sys.path so helpers resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

from message_miner.helpers.auth import get_access_token
from message_miner.helpers.graph import graph_get, graph_get_all

OUTPUT_DIR = Path("outputs/01_connect")


def _build_snapshot_entry(chat: dict, signed_in_user_name: str) -> dict:
    """Flatten a raw /me/chats item into a human-readable config_snapshot entry."""
    chat_type = chat.get("chatType", "unknown")
    members = chat.get("members", [])

    member_names = [m.get("displayName") for m in members if m.get("displayName")]
    member_ids = [m.get("userId") for m in members if m.get("userId")]

    # For 1:1 chats the topic is null; label it as the other person's name instead
    topic = chat.get("topic")
    if not topic and chat_type == "oneOnOne":
        other = [n for n in member_names if n != signed_in_user_name]
        topic = other[0] if other else "(unknown)"

    return {
        "chat_id": chat.get("id"),
        "chat_type": chat_type,
        "topic": topic,
        "member_count": len(members),
        "member_names": member_names,
        "member_ids": member_ids,
        "created_at": chat.get("createdDateTime"),
        "last_updated": chat.get("lastUpdatedDateTime"),
        "include": True,  # flip to false to exclude from 02_extract
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Guard against accidental overwrites
    if (OUTPUT_DIR / "config_snapshot.json").exists():
        answer = input(
            "outputs/01_connect/ already has output. Overwrite? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    print("=== message-miner: 01_connect ===\n")

    # ── Auth ──────────────────────────────────────────────────────────────────
    token = get_access_token()
    print("Authenticated.\n")

    # ── Current user ──────────────────────────────────────────────────────────
    me = graph_get(token, "/me")
    user_id = me.get("id")
    user_name = me.get("displayName")
    user_email = me.get("mail") or me.get("userPrincipalName")
    print(f"Signed in as: {user_name} ({user_email})\n")

    user_record = {"id": user_id, "name": user_name, "email": user_email}

    # ── Discover chats ────────────────────────────────────────────────────────
    print("Discovering chats...", end="", flush=True)
    chats = graph_get_all(
        token,
        "/me/chats",
        {"$expand": "members", "$top": "50"},
        on_page=lambda n: print(f"\r  Discovering chats... {n:,} found", end="", flush=True),
    )
    print(f"\r  Discovering chats... {len(chats):,} found.\n")

    # ── chats_inventory.json — raw API output ─────────────────────────────────
    inventory = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "user": user_record,
        "chat_count": len(chats),
        "chats": chats,
    }
    (OUTPUT_DIR / "chats_inventory.json").write_text(
        json.dumps(inventory, indent=2, default=str),
        encoding="utf-8",
    )

    # ── config_snapshot.json — human-reviewable checkpoint ───────────────────
    snapshot_entries = [_build_snapshot_entry(c, user_name) for c in chats]

    type_counts: dict[str, int] = {}
    for entry in snapshot_entries:
        key = entry["chat_type"]
        type_counts[key] = type_counts.get(key, 0) + 1

    snapshot = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user": user_record,
        "instructions": (
            "Review the chats below before running 02_extract. "
            "Set 'include' to false for any chat you want to skip. "
            "All other fields are read-only — do not edit them."
        ),
        "summary": {
            "total_chats": len(chats),
            "by_type": type_counts,
        },
        "chats": snapshot_entries,
    }
    (OUTPUT_DIR / "config_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, default=str),
        encoding="utf-8",
    )

    # ── connection_report.json — run metadata ─────────────────────────────────
    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "user": user_record,
        "chat_count": len(chats),
        "by_type": type_counts,
    }
    (OUTPUT_DIR / "connection_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("--- Discovery summary ---")
    for chat_type, count in sorted(type_counts.items()):
        label = {
            "oneOnOne": "1:1 chats",
            "group": "Group chats",
            "meeting": "Meeting chats",
        }.get(chat_type, chat_type)
        print(f"  {label:<18} {count}")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  chats_inventory.json    raw API response")
    print("  connection_report.json  run summary")
    print("  config_snapshot.json    <-- REVIEW THIS before running 02_extract")
    print("\nNext: edit config_snapshot.json (set 'include': false to exclude chats),")
    print("      then run:  python message_miner/02_extract.py")


if __name__ == "__main__":
    main()
