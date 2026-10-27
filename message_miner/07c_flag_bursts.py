"""
07c_flag_bursts.py — LLM-as-judge pass over long bursts.

After gap-only segmentation (07_segment.py), some bursts are long enough that
they probably contain more than one distinct conversation. This script asks an
LLM to flag those bursts and suggest where to split them, producing a short
list the user can review in the HTML labeler instead of reading thousands of
messages by hand.

Reads:
  outputs/07_segment/bursts.parquet
  outputs/07_segment/message_burst_map.parquet
  outputs/05_enrich/messages_enriched.parquet

Writes:
  outputs/07_segment/flagged_bursts.json

Run from project root:
    python message_miner/07c_flag_bursts.py [--min-msgs 100] [--concurrency 10]
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from message_miner.helpers.async_llm import async_call_api, process_batch

MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
BURSTS_PATH = Path("outputs/07_segment/bursts.parquet")
BURST_MAP_PATH = Path("outputs/07_segment/message_burst_map.parquet")
OUTPUT_PATH = Path("outputs/07_segment/flagged_bursts.json")

MODEL = "gpt-5.4-nano"
TEMPERATURE = 0
MAX_MESSAGE_CHARS = 400      # truncate each message text to this before sending to LLM
MAX_PROMPT_MESSAGES = 400    # hard cap on messages per burst sent to LLM (truncate head+tail if longer)

_PROMPT = """You are reviewing a continuous time block of chat messages to determine whether it contains a single conversation or multiple distinct conversations that happen to occur close together in time.

A "conversation" is a coherent thread with a shared topic, goal, or focus. Two conversations are distinct when:
- The topic clearly shifts and does NOT return
- Participants' focus shifts to a different task or problem
- The connection between messages is purely temporal (they happen near each other) rather than substantive

Do NOT split on:
- Brief tangents that return to the main topic
- Acknowledgements or small talk interleaved with the main topic
- Natural topic drift within the same broader thread

Return JSON only, with this shape:
{"multiple": true|false, "splits": [indices of message starting each new conversation AFTER the first], "reason": "one short sentence"}

Indices are 0-based within the messages below. Index 0 is always the start of conversation #1 — DO NOT include it in "splits". Only include indices where a NEW conversation begins after a previous one.

Examples:
- One conversation: {"multiple": false, "splits": [], "reason": "coherent thread about database migration"}
- Two conversations: {"multiple": true, "splits": [47], "reason": "standup updates end at [46]; separate deployment troubleshooting starts at [47]"}

Messages:
"""


def _first_name(display_name) -> str:
    if not display_name or pd.isna(display_name):
        return "Unknown"
    return str(display_name).split()[0]


def _format_gap(seconds) -> str:
    if pd.isna(seconds) or seconds <= 0:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def _format_messages(msgs: list[dict]) -> str:
    lines = []
    for i, m in enumerate(msgs):
        gap = f" ({m['gap']})" if m.get("gap") and i > 0 else ""
        text = (m.get("text") or "").strip()
        if len(text) > MAX_MESSAGE_CHARS:
            text = text[:MAX_MESSAGE_CHARS] + "…"
        lines.append(f"[{i}]{gap} {m['sender']}: {text}")
    return "\n".join(lines)


def _truncate_for_prompt(msgs: list[dict]) -> tuple[list[dict], dict]:
    """If burst is too long to send, keep head + tail. Return (sent_msgs, index_map).
    index_map: index-in-prompt → index-in-original-burst."""
    n = len(msgs)
    if n <= MAX_PROMPT_MESSAGES:
        return msgs, {i: i for i in range(n)}

    half = MAX_PROMPT_MESSAGES // 2
    head = msgs[:half]
    tail = msgs[-half:]
    sent = head + tail
    index_map = {}
    for i in range(len(head)):
        index_map[i] = i
    for j, i in enumerate(range(n - half, n)):
        index_map[len(head) + j] = i
    return sent, index_map


def _parse_llm_response(content: str) -> dict:
    """Extract {multiple, splits, reason} from the LLM reply, tolerating code fences."""
    s = content.strip()
    # Strip code fences if present
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:-1]) if len(lines) >= 3 else s.strip("`")
    # Find the first { and last }
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1:
        return {"multiple": False, "splits": [], "reason": "parse failed"}
    try:
        parsed = json.loads(s[start:end + 1])
    except json.JSONDecodeError:
        return {"multiple": False, "splits": [], "reason": "parse failed"}
    splits = parsed.get("splits") or []
    splits = sorted(set(int(x) for x in splits if isinstance(x, (int, float))))
    return {
        "multiple": bool(parsed.get("multiple", False)),
        "splits": splits,
        "reason": str(parsed.get("reason", "")).strip(),
    }


async def _judge_one(client, burst: dict) -> dict:
    """Ask the LLM whether this burst contains multiple conversations."""
    sent_msgs, index_map = _truncate_for_prompt(burst["messages"])
    prompt = _PROMPT + _format_messages(sent_msgs)

    try:
        content, usage = await async_call_api(
            client=client,
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        return {
            **burst,
            "multiple": False,
            "splits_in_burst": [],
            "reason": f"ERROR: {e}",
            "input_tokens": 0,
            "output_tokens": 0,
            "truncated": len(sent_msgs) < len(burst["messages"]),
        }

    parsed = _parse_llm_response(content)

    # Remap prompt-indices back to original burst-indices (only matters if truncated)
    splits_in_burst = sorted({index_map.get(i, i) for i in parsed["splits"] if i in index_map})

    return {
        **burst,
        "multiple": parsed["multiple"],
        "splits_in_burst": splits_in_burst,
        "reason": parsed["reason"],
        "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "truncated": len(sent_msgs) < len(burst["messages"]),
    }


def _build_burst_payloads(args) -> list[dict]:
    """Load bursts + messages and assemble the per-burst payloads for the LLM."""
    for path in (MESSAGES_PATH, BURSTS_PATH, BURST_MAP_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run 07_segment.py first.")
            sys.exit(1)

    msgs = pd.read_parquet(MESSAGES_PATH)
    if msgs["timestamp"].dt.tz is None:
        msgs["timestamp"] = pd.to_datetime(msgs["timestamp"], utc=True, errors="coerce")
    msgs = msgs[
        (msgs["message_type"] == "message")
        & (msgs["is_deleted"] == False)
        & (msgs["is_empty"] == False)
    ].sort_values(["chat_id", "timestamp"]).reset_index(drop=True)

    # Chat-level index = row position within the chat's filtered message list
    msgs["chat_msg_idx"] = msgs.groupby("chat_id").cumcount()

    bursts = pd.read_parquet(BURSTS_PATH)
    burst_map = pd.read_parquet(BURST_MAP_PATH)

    long_bursts = bursts[bursts["msg_count"] >= args.min_msgs].copy()
    long_bursts = long_bursts.sort_values("msg_count", ascending=False)

    # Index message rows by message_id for quick lookup
    msgs_by_id = msgs.set_index("message_id")

    payloads: list[dict] = []
    for _, burst_row in long_bursts.iterrows():
        burst_id = burst_row["burst_id"]
        burst_msg_ids = burst_map.loc[burst_map["burst_id"] == burst_id, "message_id"].tolist()
        burst_msgs = msgs_by_id.loc[burst_msg_ids].sort_values("timestamp").reset_index()

        message_list = []
        for i, (_, r) in enumerate(burst_msgs.iterrows()):
            message_list.append({
                "sender": _first_name(r.get("sender_name")),
                "gap": _format_gap(r.get("time_delta_seconds")) if i > 0 else "",
                "text": str(r.get("text") or "")[:1200],
                "chat_msg_idx": int(r["chat_msg_idx"]),
                "timestamp": r["timestamp"].isoformat() if pd.notna(r["timestamp"]) else None,
            })

        payloads.append({
            "burst_id": burst_id,
            "chat_id": burst_row["chat_id"],
            "chat_type": str(burst_row.get("chat_type") or ""),
            "msg_count": int(burst_row["msg_count"]),
            "duration_seconds": float(burst_row.get("duration_seconds") or 0),
            "start_time": burst_row["start_time"].isoformat() if pd.notna(burst_row.get("start_time")) else None,
            "messages": message_list,
        })

    return payloads


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-as-judge pass over long bursts")
    parser.add_argument("--min-msgs", type=int, default=100,
                        help="Minimum messages in a burst to consider it for review (default: 100)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max simultaneous API calls (default: 10)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of bursts to judge (for testing)")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env and re-run.")
        sys.exit(1)

    print("=== message-miner: 07c_flag_bursts ===\n")
    print(f"Flagging threshold: msg_count >= {args.min_msgs}")
    print(f"Model: {MODEL}   Concurrency: {args.concurrency}\n")

    payloads = _build_burst_payloads(args)
    print(f"Candidate bursts to judge: {len(payloads)}")
    if args.limit:
        payloads = payloads[:args.limit]
        print(f"  (capped to {len(payloads)} via --limit)")

    if not payloads:
        print("Nothing to judge. Exiting.")
        return

    results = asyncio.run(process_batch(
        items=payloads,
        process_fn=_judge_one,
        api_key=api_key,
        concurrency=args.concurrency,
        desc="Judging bursts",
    ))

    flagged = [r for r in results if r.get("multiple")]
    not_flagged = [r for r in results if not r.get("multiple")]
    errors = [r for r in results if str(r.get("reason", "")).startswith("ERROR")]

    total_in = sum(int(r.get("input_tokens", 0)) for r in results)
    total_out = sum(int(r.get("output_tokens", 0)) for r in results)

    print(f"\n--- Results ---")
    print(f"  Judged:       {len(results)}")
    print(f"  Flagged:      {len(flagged)}  (LLM thinks multiple conversations)")
    print(f"  Not flagged:  {len(not_flagged) - len(errors)}")
    print(f"  Errors:       {len(errors)}")
    print(f"  Input tokens: {total_in:,}")
    print(f"  Output tokens:{total_out:,}")

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "min_msgs_threshold": args.min_msgs,
        "counts": {
            "judged": len(results),
            "flagged": len(flagged),
            "not_flagged": len(not_flagged) - len(errors),
            "errors": len(errors),
        },
        "bursts": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"\nNext: python message_miner/label_boundaries.py review")


if __name__ == "__main__":
    main()
