"""
07b_segment_llm.py — LLM Conversation Segmentation

Uses GPT to identify semantic conversation boundaries within each chat, then
produces the exact same output schema as 07_segment.py. Run this instead of
07_segment when you have an OpenAI API key and want higher-quality segmentation.

Outputs intentionally overwrite 07_segment's output so 08_bertopic always reads
from the same location regardless of which segmenter was used.

Resume support: each chat's boundary result is cached to
  outputs/07_segment/llm_cache/{chat_id}.json
If the script is interrupted, re-running resumes from where it left off.
Use --force to clear the cache and re-process everything.

Reads:
  outputs/05_enrich/messages_enriched.parquet
  outputs/02_extract/chats.parquet

Writes:
  outputs/07_segment/bursts.parquet
  outputs/07_segment/message_burst_map.parquet
  outputs/07_segment/segmentation_report.json
  outputs/07_segment/llm_cache/          per-chat boundary cache

Run from project root:
    python message_miner/07b_segment_llm.py [--force] [--concurrency 10]
"""

import argparse
import asyncio
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

from message_miner.helpers.async_llm import async_call_api

MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
CHATS_PATH = Path("outputs/02_extract/chats.parquet")
OUTPUT_DIR = Path("outputs/07_segment")
CACHE_DIR = OUTPUT_DIR / "llm_cache"

MODEL = "gpt-5.4-nano"           # first-pass segmentation model
VERIFY_MODEL = "gpt-5.4"        # second-pass verification model (higher quality)
TEMPERATURE = 0
CHUNK_SIZE = 150                 # messages per LLM call
OVERLAP = 20                     # trailing messages repeated at next chunk start
SESSION_GAP_SECONDS = 14400     # 4-hour gap → automatic boundary, no LLM needed
GAP_FALLBACK_SECONDS = 1800     # time-gap threshold used when LLM fails for a chat
MAX_RETRIES = 5
CONFIDENCE_THRESHOLD = 0.7      # boundaries below this score get a second-pass verification
VERIFY_CONTEXT = 10             # messages shown before/after boundary in verification pass
CLIENT_TIMEOUT = 60.0           # seconds before an OpenAI request times out

# Approximate OpenAI pricing per 1M tokens (update if pricing changes)
INPUT_PRICE_PER_1M = 0.15
OUTPUT_PRICE_PER_1M = 0.60

# Markdown code fence pattern (copied from 07_segment)
_CODE_FENCE_RE = re.compile(r"```|~~~")


# ── Text / formatting helpers ──────────────────────────────────────────────────

def _first_name(display_name: str | None) -> str:
    if not display_name:
        return "Unknown"
    return display_name.split()[0]


def _build_text_concatenated(group: pd.DataFrame) -> str | None:
    lines = []
    for _, row in group.iterrows():
        if pd.isna(row["text"]) or not str(row["text"]).strip():
            continue
        lines.append(f"{_first_name(row.get('sender_name'))}: {row['text'].strip()}")
    return "\n".join(lines) if lines else None


def _format_gap(seconds) -> str:
    if pd.isna(seconds) or seconds <= 0:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m"
    elif s < 86400:
        return f"{s // 3600}h"
    else:
        return f"{s // 86400}d"


def _format_messages_for_prompt(msgs: pd.DataFrame) -> str:
    """Render messages as compact prompt text."""
    lines = []
    for i, (_, row) in enumerate(msgs.iterrows()):
        gap = _format_gap(row.get("time_delta_seconds"))
        gap_part = f" ({gap})" if gap and i > 0 else ""
        name = _first_name(row.get("sender_name"))
        text = str(row.get("text") or "").strip()[:300]  # cap very long messages
        lines.append(f"[{i}]{gap_part} {name}: {text}")
    return "\n".join(lines)


# ── LLM prompts ────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
You are analyzing a workplace chat conversation. Your task is to identify where distinct conversation topics begin and end.

Below is a sequence of messages. Each line has the format:
[index] (time gap from previous) sender: message text

Identify the message indices where a NEW conversation topic or purpose begins. A boundary means the participants shifted to a substantially different subject, task, or purpose — not minor tangents within the same discussion.

Rules:
- The first message (index 0) is always a boundary (start of first conversation).
- Only mark boundaries where the topic genuinely shifts, not where someone briefly goes off-topic and comes back.
- Time gaps are a strong signal. A gap of 1 day or more should almost always be treated as a boundary unless the very next message is an unmistakable direct continuation of the prior topic. A gap of 1 week or more is always a boundary.
- A short gap is not sufficient on its own — an immediate pivot to a new subject IS a boundary regardless of how little time passed.
- For each boundary, rate your confidence (0.0 to 1.0) that it represents a genuine topic shift. Use 1.0 for obvious shifts (long gap + clear new subject), lower scores for ambiguous cases.
- Return ONLY valid JSON: {{"boundaries": [{{"index": 0, "confidence": 1.0}}, {{"index": 15, "confidence": 0.9}}, ...]}}

Messages:
{messages}"""

_VERIFY_PROMPT_TEMPLATE = """\
You are reviewing a proposed conversation boundary in a workplace chat.

The messages below are shown with a [--- PROPOSED BOUNDARY ---] marker where a topic shift was tentatively identified. Review the messages before and after the marker.

Does this mark a genuine shift in conversation topic or purpose — meaning the participants moved to a substantially different subject, task, or context?

Return ONLY valid JSON: {{"is_boundary": true}} or {{"is_boundary": false}}

Messages:
{messages}"""


# ── OpenAI calls ──────────────────────────────────────────────────────────────

async def _call_llm(client, messages_text: str) -> tuple[str, object]:
    """Async first-pass segmentation call. Returns (content, usage)."""
    prompt = _PROMPT_TEMPLATE.format(messages=messages_text)
    return await async_call_api(
        client,
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        max_retries=MAX_RETRIES,
    )


async def _call_verify(client, messages_text: str) -> tuple[str, object]:
    """Async second-pass verification call. Returns (content, usage)."""
    prompt = _VERIFY_PROMPT_TEMPLATE.format(messages=messages_text)
    return await async_call_api(
        client,
        model=VERIFY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        response_format={"type": "json_object"},
        max_retries=MAX_RETRIES,
    )


def _parse_boundaries(content: str, n_messages: int) -> tuple[list[int], dict[int, float]] | tuple[None, None]:
    """
    Parse boundary list and confidence scores from LLM JSON.
    Returns (boundaries, confidences) or (None, None) on failure.
    confidences is a dict of {boundary_index: confidence_score}.
    Supports both new format (list of dicts) and old format (list of ints) as fallback.
    """
    try:
        data = json.loads(content)
        raw = data.get("boundaries", [])
        if not isinstance(raw, list) or not raw:
            return [0], {0: 1.0}

        # New format: [{"index": 0, "confidence": 1.0}, ...]
        if isinstance(raw[0], dict):
            boundaries = []
            confidences = {}
            for item in raw:
                idx = int(item["index"])
                conf = float(item.get("confidence", 1.0))
                boundaries.append(idx)
                confidences[idx] = conf
        else:
            # Fallback: plain list of ints (no confidence info)
            boundaries = [int(b) for b in raw]
            confidences = {b: 1.0 for b in boundaries}

        boundaries = sorted(set(boundaries))
        if any(b < 0 or b >= n_messages for b in boundaries):
            return None, None
        if 0 not in boundaries:
            boundaries = [0] + boundaries
            confidences[0] = 1.0
        return boundaries, confidences
    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return None, None


# ── Time-gap fallback ─────────────────────────────────────────────────────────

def _time_gap_boundaries(msgs: pd.DataFrame, threshold: int = GAP_FALLBACK_SECONDS) -> list[int]:
    boundaries = [0]
    for i in range(1, len(msgs)):
        delta = msgs.iloc[i]["time_delta_seconds"]
        if pd.isna(delta) or float(delta) > threshold:
            boundaries.append(i)
    return boundaries


# ── Session pre-partitioning ──────────────────────────────────────────────────

def _get_session_starts(chat_msgs: pd.DataFrame) -> list[int]:
    """
    Return local indices where sessions start, defined by gaps >= SESSION_GAP_SECONDS.
    Sessions are guaranteed boundaries — no LLM call needed to confirm them.
    """
    starts = [0]
    for i in range(1, len(chat_msgs)):
        delta = chat_msgs.iloc[i]["time_delta_seconds"]
        if pd.isna(delta) or float(delta) >= SESSION_GAP_SECONDS:
            starts.append(i)
    return starts


# ── Per-chat segmentation ─────────────────────────────────────────────────────

async def _segment_chat(
    chat_msgs: pd.DataFrame,
    chat_id: str,
    client,
    cache_dir: Path,
) -> tuple[list[int], str, int, int, int]:
    """
    Segment one chat's messages. Returns (global_boundaries, method, api_calls,
    input_tokens, output_tokens).
    method is "llm" or "fallback".
    Reads / writes cache transparently.
    """
    safe_id = chat_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    cache_file = cache_dir / f"{safe_id}.json"
    n = len(chat_msgs)
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_b = cached["boundaries"]
        # Validate boundaries are in range (cache may be stale after --since filtering)
        if all(0 <= b < n for b in cached_b):
            return cached_b, cached["method"], 0, 0, 0
        # Stale cache — re-segment
        cache_file.unlink()
    input_tokens = 0
    output_tokens = 0

    # Trivial chats — no LLM needed
    if n <= 2:
        cache_file.write_text(json.dumps({"boundaries": [0], "method": "llm"}), encoding="utf-8")
        return [0], "llm", 0, 0, 0

    global_boundaries: dict[int, float] = {0: 1.0}  # boundary_idx → confidence
    api_calls = 0
    method = "llm"
    chunk_fallbacks = 0
    force_verify_set: set[int] = set()

    # ── Pre-partition into sessions by large time gaps ────────────────────────
    session_starts = _get_session_starts(chat_msgs)
    if len(session_starts) > 1:
        print(f"      Pre-partitioned into {len(session_starts)} sessions (gap >= {SESSION_GAP_SECONDS // 3600}h)")

    # ── First pass: identify boundaries with confidence scores ────────────────
    step = CHUNK_SIZE - OVERLAP

    for sess_i, sess_start in enumerate(session_starts):
        sess_end = session_starts[sess_i + 1] if sess_i + 1 < len(session_starts) else n

        # Session gap boundaries are automatic — no LLM needed
        if sess_start > 0:
            global_boundaries[sess_start] = 1.0

        session = chat_msgs.iloc[sess_start:sess_end]
        sess_n = len(session)

        if sess_n <= 2:
            continue

        chunk_starts = list(range(0, sess_n, step))

        for chunk_idx, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + CHUNK_SIZE, sess_n)
            chunk = session.iloc[chunk_start:chunk_end]
            messages_text = _format_messages_for_prompt(chunk)

            boundaries = None
            confidences = None
            for attempt in range(2):
                try:
                    content, usage = await _call_llm(client, messages_text)
                    input_tokens += usage.prompt_tokens
                    output_tokens += usage.completion_tokens
                    api_calls += 1
                    boundaries, confidences = _parse_boundaries(content, len(chunk))
                    if boundaries is not None:
                        break
                    print(f"      [warn] Unparseable JSON (attempt {attempt + 1}), retrying prompt...")
                except Exception as e:
                    print(f"      [warn] API error: {e}")
                    break

            if boundaries is None:
                # Chunk-level fallback: use time-gap for just this chunk, continue with the rest
                print(f"      [chunk fallback] chunk {chunk_idx + 1}/{len(chunk_starts)} failed — time-gap for this window.")
                chunk_fallbacks += 1
                method = "partial_fallback"
                for local_b in _time_gap_boundaries(chunk):
                    if chunk_idx > 0 and local_b == 0:
                        continue
                    global_idx = sess_start + chunk_start + local_b
                    if global_idx not in global_boundaries:
                        global_boundaries[global_idx] = 1.0
                continue

            for local_b in boundaries:
                if chunk_idx > 0 and local_b == 0:
                    continue
                global_idx = sess_start + chunk_start + local_b
                conf = confidences.get(local_b, 1.0)
                # If seen from multiple chunks, keep higher confidence
                if global_idx not in global_boundaries or conf > global_boundaries[global_idx]:
                    global_boundaries[global_idx] = conf

                # Boundaries in the overlap zone are structurally higher-risk —
                # force Pass 2 verification regardless of confidence score
                if (chunk_idx > 0 and local_b < OVERLAP) or local_b > (len(chunk) - OVERLAP):
                    force_verify_set.add(global_idx)

    if chunk_fallbacks:
        print(f"      {chunk_fallbacks} chunk(s) used time-gap fallback; rest used LLM.")

    # ── Second pass: verify low-confidence and overlap-zone boundaries ────────
    low_conf = [
        b for b, c in global_boundaries.items()
        if b != 0 and (c < CONFIDENCE_THRESHOLD or b in force_verify_set)
    ]
    if low_conf:
        n_overlap = sum(1 for b in low_conf if b in force_verify_set and global_boundaries.get(b, 0) >= CONFIDENCE_THRESHOLD)
        print(
            f"      Verifying {len(low_conf)} boundaries with {VERIFY_MODEL} "
            f"({n_overlap} overlap-zone, {len(low_conf) - n_overlap} low-confidence)..."
        )
    for boundary_idx in low_conf:
        ctx_start = max(0, boundary_idx - VERIFY_CONTEXT)
        ctx_end = min(n, boundary_idx + VERIFY_CONTEXT + 1)
        context_chunk = chat_msgs.iloc[ctx_start:ctx_end]

        lines = []
        for i, (_, row) in enumerate(context_chunk.iterrows()):
            global_i = ctx_start + i
            gap = _format_gap(row.get("time_delta_seconds"))
            gap_part = f" ({gap})" if gap and global_i > ctx_start else ""
            name = _first_name(row.get("sender_name"))
            text = str(row.get("text") or "").strip()[:300]
            if global_i == boundary_idx:
                lines.append("[--- PROPOSED BOUNDARY ---]")
            lines.append(f"[{global_i}]{gap_part} {name}: {text}")
        context_text = "\n".join(lines)

        try:
            content, usage = await _call_verify(client, context_text)
            input_tokens += usage.prompt_tokens
            output_tokens += usage.completion_tokens
            api_calls += 1
            data = json.loads(content)
            if not data.get("is_boundary", True):
                del global_boundaries[boundary_idx]
        except Exception as e:
            print(f"      [warn] Verify failed for boundary {boundary_idx}: {e} — keeping it")

    result_boundaries = sorted(global_boundaries.keys())
    cache_file.write_text(
        json.dumps({"boundaries": result_boundaries, "method": method}),
        encoding="utf-8",
    )
    return result_boundaries, method, api_calls, input_tokens, output_tokens


# ── Burst aggregation (mirrors 07_segment) ────────────────────────────────────

def _aggregate_burst(group: pd.DataFrame, chat_type_map: dict) -> dict:
    burst_id = group["burst_id"].iloc[0]
    chat_id = group["chat_id"].iloc[0]
    burst_index = int(group["_burst_index"].iloc[0])

    timestamps = group["timestamp"].dropna()
    start_time = timestamps.min() if not timestamps.empty else pd.NaT
    end_time = timestamps.max() if not timestamps.empty else pd.NaT
    duration_seconds = (
        (end_time - start_time).total_seconds()
        if pd.notna(start_time) and pd.notna(end_time)
        else None
    )

    senders = group["sender_name"].dropna().unique().tolist()
    sender_ids = group["sender_id"].dropna().unique().tolist()
    self_msgs = int(group["is_self"].sum())
    msg_count = len(group)
    word_count_total = group["word_count"].dropna().sum()
    text_concat = _build_text_concatenated(group)

    has_code = bool(
        group["text"].dropna().apply(lambda t: bool(_CODE_FENCE_RE.search(t))).any()
    )
    has_attachments = bool(group["has_attachments"].any())

    return {
        "burst_id": burst_id,
        "chat_id": chat_id,
        "chat_type": chat_type_map.get(chat_id),
        "burst_index": burst_index,
        "start_time": start_time,
        "end_time": end_time,
        "duration_seconds": duration_seconds,
        "msg_count": msg_count,
        "participant_count": len(sender_ids),
        "participants": ",".join(senders) if senders else None,
        "self_msg_count": self_msgs,
        "self_msg_ratio": float(self_msgs / msg_count) if msg_count else None,
        "text_concatenated": text_concat,
        "word_count": int(word_count_total) if word_count_total else 0,
        "has_code": has_code,
        "has_attachments": has_attachments,
    }


# ── Report ────────────────────────────────────────────────────────────────────

def _build_report(
    bursts_df: pd.DataFrame,
    n_input: int,
    n_filtered: int,
    total_tokens: dict,
    total_api_calls: int,
    fallback_chats: int,
    chats_from_cache: int,
) -> dict:
    def _dist(s: pd.Series) -> dict:
        if s.empty:
            return {}
        desc = s.describe(percentiles=[0.25, 0.5, 0.75, 0.95])
        return {k: round(float(v), 2) for k, v in desc.items()}

    bursts_per_chat = bursts_df.groupby("chat_id").size()
    single_msg = int((bursts_df["msg_count"] == 1).sum())
    total_bursts = len(bursts_df)
    by_type = bursts_df["chat_type"].value_counts().to_dict()

    cost = (
        total_tokens["input"] / 1_000_000 * INPUT_PRICE_PER_1M
        + total_tokens["output"] / 1_000_000 * OUTPUT_PRICE_PER_1M
    )

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "segmentation_method": "llm",
        "model": MODEL,
        "verify_model": VERIFY_MODEL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "gap_threshold_seconds": GAP_FALLBACK_SECONDS,
        "input_messages": n_input,
        "filtered_messages": n_filtered,
        "total_bursts": total_bursts,
        "single_message_bursts": single_msg,
        "single_message_burst_pct": round(100 * single_msg / total_bursts, 2) if total_bursts else 0,
        "by_chat_type": {str(k): int(v) for k, v in by_type.items()},
        "burst_length_distribution": _dist(bursts_df["msg_count"].astype(float)),
        "burst_duration_distribution": _dist(bursts_df["duration_seconds"].dropna()),
        "bursts_per_chat_distribution": _dist(bursts_per_chat.astype(float)),
        "total_api_calls": total_api_calls,
        "fallback_chats": fallback_chats,
        "chats_from_cache": chats_from_cache,
        "total_input_tokens": total_tokens["input"],
        "total_output_tokens": total_tokens["output"],
        "estimated_cost_usd": round(cost, 4),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-based chat segmentation")
    parser.add_argument("--force", action="store_true", help="Clear LLM cache and re-process all chats")
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent chats processed simultaneously (default: 10)"
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env and re-run.")
        sys.exit(1)

    if (OUTPUT_DIR / "bursts.parquet").exists() and not args.force:
        answer = input(
            "outputs/07_segment/ already has output. Overwrite (keeps LLM cache unless --force)? [y/N] "
        ).strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    if args.force and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
        print("LLM cache cleared.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    print("=== message-miner: 07b_segment_llm ===\n")
    print(f"Model:       {MODEL}")
    print(f"Chunk size:  {CHUNK_SIZE} messages  (overlap: {OVERLAP})")
    print(f"Sessions:    pre-partitioned at {SESSION_GAP_SECONDS // 3600}h gaps (automatic boundaries)")
    print(f"Fallback:    time-gap @ {GAP_FALLBACK_SECONDS}s if LLM fails")
    print(f"Concurrency: {args.concurrency} chats\n")

    for path in (MESSAGES_PATH, CHATS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(MESSAGES_PATH)
    chats_df = pd.read_parquet(CHATS_PATH)
    n_input = len(df)
    print(f"Loaded {n_input:,} messages.\n")

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # ── Filter ────────────────────────────────────────────────────────────────
    mask = (
        (df["message_type"] == "message") &
        (df["is_deleted"] == False) &
        (df["is_empty"] == False)
    )
    df = df[mask].copy()
    n_filtered = len(df)
    print(f"After filtering (message, not deleted, not empty): {n_filtered:,} messages.\n")

    df = df.sort_values(["chat_id", "timestamp"]).reset_index(drop=True)

    chat_type_map = chats_df.set_index("chat_id")["chat_type"].astype(str).to_dict()

    # ── Per-chat segmentation (async concurrent) ─────────────────────────────
    chat_ids = df["chat_id"].unique().tolist()
    n_chats = len(chat_ids)
    print(f"Processing {n_chats:,} chats (concurrency={args.concurrency})...\n")

    # Pre-build per-chat DataFrames so the async tasks don't filter on the fly
    chat_dfs = {cid: df[df["chat_id"] == cid] for cid in chat_ids}

    async def _run_all():
        from openai import AsyncOpenAI
        from tqdm.asyncio import tqdm_asyncio

        client = AsyncOpenAI(api_key=api_key, timeout=CLIENT_TIMEOUT)
        semaphore = asyncio.Semaphore(args.concurrency)

        async def _process_one(chat_id: str):
            async with semaphore:
                chat_msgs = chat_dfs[chat_id]
                return await _segment_chat(
                    chat_msgs, chat_id, client, CACHE_DIR,
                )

        tasks = [_process_one(cid) for cid in chat_ids]
        results = await tqdm_asyncio.gather(*tasks, desc="  Segmenting chats")
        await client.close()
        return results

    results = asyncio.run(_run_all())

    total_tokens = {"input": 0, "output": 0}
    total_api_calls = 0
    fallback_chats = 0
    chats_from_cache = 0

    all_burst_ids: list[str] = []
    all_burst_indices: list[int] = []
    message_indices: list[int] = []  # positional indices into df

    for chat_id, (boundaries, method, calls, in_tok, out_tok) in zip(chat_ids, results):
        chat_msgs = chat_dfs[chat_id]

        total_tokens["input"] += in_tok
        total_tokens["output"] += out_tok
        total_api_calls += calls
        if calls == 0 and in_tok == 0:
            chats_from_cache += 1
        if method in ("fallback", "partial_fallback"):
            fallback_chats += 1

        # Assign burst indices from boundaries
        n = len(chat_msgs)
        b_sorted = sorted(boundaries)
        local_burst_indices = [0] * n
        for burst_num, start in enumerate(b_sorted):
            end = b_sorted[burst_num + 1] if burst_num + 1 < len(b_sorted) else n
            for pos in range(start, end):
                local_burst_indices[pos] = burst_num

        # Store globally
        for pos, (df_idx, _) in enumerate(chat_msgs.iterrows()):
            message_indices.append(df_idx)
            all_burst_indices.append(local_burst_indices[pos])
            all_burst_ids.append(f"{chat_id}__burst_{local_burst_indices[pos]}")

    print(f"\nSegmentation complete.  API calls: {total_api_calls}  |  Fallbacks: {fallback_chats}  |  From cache: {chats_from_cache}\n")

    # ── Attach burst columns to df ────────────────────────────────────────────
    idx_series = pd.Series(message_indices)
    df = df.loc[idx_series.values].copy()
    df["_burst_index"] = all_burst_indices
    df["burst_id"] = all_burst_ids

    total_bursts = df["burst_id"].nunique()
    print(f"Identified {total_bursts:,} bursts across {n_chats:,} chats.\n")

    # ── message_burst_map ─────────────────────────────────────────────────────
    burst_map = df[["message_id", "burst_id"]].copy()
    burst_map.to_parquet(OUTPUT_DIR / "message_burst_map.parquet", index=False)

    # ── Build bursts.parquet ──────────────────────────────────────────────────
    print("Aggregating burst records...")
    burst_records = []
    for burst_id, group in df.groupby("burst_id", sort=False):
        burst_records.append(_aggregate_burst(group, chat_type_map))

    bursts_df = pd.DataFrame(burst_records)
    bursts_df["burst_index"] = bursts_df["burst_index"].astype("int32")
    bursts_df["msg_count"] = bursts_df["msg_count"].astype("int32")
    bursts_df["participant_count"] = bursts_df["participant_count"].astype("int32")
    bursts_df["self_msg_count"] = bursts_df["self_msg_count"].astype("int32")
    bursts_df["word_count"] = bursts_df["word_count"].astype("int32")
    bursts_df["self_msg_ratio"] = bursts_df["self_msg_ratio"].astype("float32")
    bursts_df["chat_type"] = bursts_df["chat_type"].astype("category")
    for col in ("start_time", "end_time"):
        bursts_df[col] = pd.to_datetime(bursts_df[col], utc=True, errors="coerce")

    bursts_df.to_parquet(OUTPUT_DIR / "bursts.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    report = _build_report(
        bursts_df, n_input, n_filtered, total_tokens,
        total_api_calls, fallback_chats, chats_from_cache,
    )
    (OUTPUT_DIR / "segmentation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ── Console summary ───────────────────────────────────────────────────────
    bl = report["burst_length_distribution"]
    bd = report["burst_duration_distribution"]

    print(f"\n--- LLM Segmentation summary ---")
    print(f"  Input messages:       {n_input:,}")
    print(f"  Filtered messages:    {n_filtered:,}")
    print(f"  Total bursts:         {total_bursts:,}")
    print(f"  Single-msg bursts:    {report['single_message_bursts']:,}  ({report['single_message_burst_pct']}%)")
    print(f"  LLM fallback chats:   {fallback_chats}")
    print(f"  Chats from cache:     {chats_from_cache}")
    print(f"  Total API calls:      {total_api_calls}")
    print(f"  Tokens (in/out):      {total_tokens['input']:,} / {total_tokens['output']:,}")
    print(f"  Est. cost:            ${report['estimated_cost_usd']:.4f}")

    print(f"\n  Burst length (messages):")
    print(f"    median  {bl.get('50%', 0):.0f}")
    print(f"    p95     {bl.get('95%', 0):.0f}")
    print(f"    max     {bl.get('max', 0):.0f}")

    print(f"\n  Burst duration (seconds):")
    print(f"    median  {bd.get('50%', 0):.0f}s  ({bd.get('50%', 0)/60:.1f} min)")
    print(f"    p95     {bd.get('95%', 0):.0f}s  ({bd.get('95%', 0)/60:.1f} min)")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  bursts.parquet            one row per burst")
    print("  message_burst_map.parquet message_id to burst_id")
    print("  segmentation_report.json  summary stats + LLM cost")
    print(f"  llm_cache/                {len(list(CACHE_DIR.iterdir()))} cached chats")
    print(f"\nNext: python message_miner/08_bertopic.py")


if __name__ == "__main__":
    main()
