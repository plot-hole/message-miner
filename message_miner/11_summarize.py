"""
11_summarize.py — Summarize Conversation Bursts

Calls the OpenAI API to produce a 1-2 sentence summary of each conversation
burst produced by 07b_segment_llm. Creates a compression layer that downstream
scripts (topic modeling, classification, semantic search) can consume instead
of raw message text.

Reads:
  outputs/07_segment/bursts.parquet
  outputs/07_segment/message_burst_map.parquet
  outputs/05_enrich/messages_enriched.parquet

Writes:
  outputs/11_summarize/summaries.parquet
  outputs/11_summarize/llm_cache/         per-burst cached JSON responses
  outputs/11_summarize/summarization_report.json

Run from project root:
    python message_miner/11_summarize.py [--force] [--concurrency 10]
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import os

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from message_miner.helpers.async_llm import async_call_api, process_batch

BURSTS_PATH      = Path("outputs/07_segment/bursts.parquet")
BURST_MAP_PATH   = Path("outputs/07_segment/message_burst_map.parquet")
MESSAGES_PATH    = Path("outputs/05_enrich/messages_enriched.parquet")
OUTPUT_DIR       = Path("outputs/11_summarize")
CACHE_DIR        = OUTPUT_DIR / "llm_cache"

MODEL            = "gpt-5.4-nano"
TEMPERATURE      = 0
MIN_WORD_COUNT   = 10
MAX_RETRIES      = 5

# Approximate nano pricing per 1M tokens
INPUT_PRICE_PER_1M  = 0.15
OUTPUT_PRICE_PER_1M = 0.60


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Summarize this workplace chat conversation in 1-2 sentences. Capture: what was discussed, what was decided or resolved (if anything), and who was involved. Be specific — use names, project names, and concrete details. Don't be generic.

If the conversation is purely social/casual with no work content, say so in one sentence.

Conversation:
{conversation}

Summary:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_name(display_name: str | None) -> str:
    if not display_name:
        return "Unknown"
    return display_name.split()[0]


def _format_conversation(msgs: pd.DataFrame) -> str:
    """Render burst messages as a readable conversation for the prompt."""
    lines = []
    for _, row in msgs.sort_values("timestamp").iterrows():
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        name = _first_name(row.get("sender_name"))
        # Local time display — convert UTC timestamp to local
        ts = row.get("timestamp")
        try:
            ts_local = pd.to_datetime(ts, utc=True).tz_convert(None)
            time_str = ts_local.strftime("%I:%M %p").lstrip("0") or "12:00 AM"
        except Exception:
            time_str = str(ts)[:16]
        lines.append(f"{name} ({time_str}): {text}")
    return "\n".join(lines)


def _call_api(client, prompt: str) -> tuple[str, object]:
    """Call the API with retry on rate limits. Returns (content, usage)."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip(), response.usage
        except Exception as e:
            err = str(e)
            if ("429" in err or "rate_limit" in err.lower()) and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"    Rate limited — waiting {wait}s (retry {attempt + 2}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def _cache_path(burst_id: str) -> Path:
    safe = burst_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.json"


def _load_cache(burst_id: str) -> dict | None:
    path = _cache_path(burst_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(burst_id: str, data: dict) -> None:
    path = _cache_path(burst_id)
    path.write_text(json.dumps(data), encoding="utf-8")


def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1_000_000 * INPUT_PRICE_PER_1M +
        output_tokens / 1_000_000 * OUTPUT_PRICE_PER_1M
    )


# ── Main ──────────────────────────────────────────────────────────────────────

async def _process_one(client, item: dict) -> dict:
    """Async process function for a single uncached burst. Called by process_batch."""
    burst_id = item["burst_id"]
    chat_id  = item["chat_id"]
    prompt   = item["prompt"]

    try:
        summary, usage = await async_call_api(
            client,
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
        )
        in_tok  = usage.prompt_tokens
        out_tok = usage.completion_tokens

        _save_cache(burst_id, {
            "summary":       summary,
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
        })

        return {
            "burst_id":      burst_id,
            "chat_id":       chat_id,
            "summary":       summary,
            "skip_reason":   None,
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
        }
    except Exception as e:
        print(f"    [error] burst {burst_id}: {e}")
        return {
            "burst_id":      burst_id,
            "chat_id":       chat_id,
            "summary":       None,
            "skip_reason":   "error",
            "input_tokens":  0,
            "output_tokens": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize conversation bursts via OpenAI")
    parser.add_argument("--force", action="store_true", help="Re-summarize everything, ignoring cache")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls (default: 10)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    if args.force:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print("Cache cleared.")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=== message-miner: 11_summarize ===\n")

    for path in (BURSTS_PATH, BURST_MAP_PATH, MESSAGES_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    print("Loading data...")
    bursts      = pd.read_parquet(BURSTS_PATH)
    burst_map   = pd.read_parquet(BURST_MAP_PATH)
    messages    = pd.read_parquet(MESSAGES_PATH)
    print(f"  {len(bursts):,} bursts")
    print(f"  {len(burst_map):,} message-burst mappings")
    print(f"  {len(messages):,} messages\n")

    # ── OpenAI client ─────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    # ── Build per-burst message lookup ────────────────────────────────────────
    # Join map → messages to get full text per burst
    msg_cols = ["message_id", "chat_id", "sender_name", "timestamp",
                "text", "message_type", "is_deleted", "is_self"]
    msg_lookup = messages[msg_cols].copy()

    burst_to_msgs = (
        burst_map
        .merge(msg_lookup, on="message_id", how="left")
    )
    # Group by burst_id for fast lookup
    burst_groups = {
        bid: grp for bid, grp in burst_to_msgs.groupby("burst_id")
    }

    # ── Phase 1: Sync — classify each burst (skip / cached / needs API) ──────
    results = []           # final result dicts (skips + cached)
    uncached_items = []    # items that need async API calls
    # Track insertion order: list of (source, index) where source is
    # "sync" (index into results) or "async" (index into uncached_items)
    ordering = []

    total_input_tokens  = 0
    total_output_tokens = 0
    skipped_too_short   = 0
    skipped_monologue   = 0
    cached_hits         = 0

    total = len(bursts)
    print(f"Sorting {total:,} bursts (skip / cached / needs-API)...")

    for _, burst in bursts.iterrows():
        burst_id = burst["burst_id"]
        chat_id  = burst["chat_id"]

        # ── Filter: too short ─────────────────────────────────────────────────
        if burst["word_count"] < MIN_WORD_COUNT:
            ordering.append(("sync", len(results)))
            results.append({
                "burst_id":      burst_id,
                "chat_id":       chat_id,
                "summary":       None,
                "skip_reason":   "too_short",
                "input_tokens":  0,
                "output_tokens": 0,
            })
            skipped_too_short += 1
            continue

        # ── Filter: monologue (all messages same sender) ───────────────────────
        burst_msgs = burst_groups.get(burst_id)
        if burst_msgs is None or burst_msgs.empty:
            ordering.append(("sync", len(results)))
            results.append({
                "burst_id":      burst_id,
                "chat_id":       chat_id,
                "summary":       None,
                "skip_reason":   "too_short",
                "input_tokens":  0,
                "output_tokens": 0,
            })
            skipped_too_short += 1
            continue

        # Only consider real user messages
        real_msgs = burst_msgs[
            (burst_msgs["message_type"] == "message") &
            (burst_msgs["is_deleted"] == False)
        ]
        unique_senders = real_msgs["sender_name"].dropna().nunique()
        if unique_senders <= 1:
            ordering.append(("sync", len(results)))
            results.append({
                "burst_id":      burst_id,
                "chat_id":       chat_id,
                "summary":       None,
                "skip_reason":   "monologue",
                "input_tokens":  0,
                "output_tokens": 0,
            })
            skipped_monologue += 1
            continue

        # ── Check cache ───────────────────────────────────────────────────────
        cached = _load_cache(burst_id)
        if cached is not None:
            ordering.append(("sync", len(results)))
            results.append({
                "burst_id":      burst_id,
                "chat_id":       chat_id,
                "summary":       cached["summary"],
                "skip_reason":   None,
                "input_tokens":  cached["input_tokens"],
                "output_tokens": cached["output_tokens"],
            })
            total_input_tokens  += cached["input_tokens"]
            total_output_tokens += cached["output_tokens"]
            cached_hits += 1
            continue

        # ── Needs API call — build prompt, defer to Phase 2 ───────────────────
        conversation_text = _format_conversation(real_msgs)
        prompt = _PROMPT_TEMPLATE.format(conversation=conversation_text)
        ordering.append(("async", len(uncached_items)))
        uncached_items.append({
            "burst_id": burst_id,
            "chat_id":  chat_id,
            "prompt":   prompt,
        })

    summarized = cached_hits
    print(f"  Skipped (too short): {skipped_too_short:,}")
    print(f"  Skipped (monologue): {skipped_monologue:,}")
    print(f"  Cached hits:         {cached_hits:,}")
    print(f"  Need API calls:      {len(uncached_items):,}\n")

    # ── Phase 2: Async — process uncached items concurrently ──────────────────
    skipped_error = 0
    async_results = []

    if uncached_items:
        print(f"Calling API with concurrency={args.concurrency}...")
        async_results = asyncio.run(
            process_batch(
                items=uncached_items,
                process_fn=_process_one,
                api_key=api_key,
                concurrency=args.concurrency,
                desc="Summarizing",
            )
        )

        # Accumulate tokens and counts from async results
        for r in async_results:
            total_input_tokens  += r["input_tokens"]
            total_output_tokens += r["output_tokens"]
            if r["skip_reason"] is None:
                summarized += 1
            elif r["skip_reason"] == "error":
                skipped_error += 1

    # ── Phase 3: Sync — merge results in original burst order, write outputs ──
    merged = []
    for source, idx in ordering:
        if source == "sync":
            merged.append(results[idx])
        else:
            merged.append(async_results[idx])

    # ── Write summaries.parquet ───────────────────────────────────────────────
    print(f"\nWriting summaries.parquet...")
    summaries = pd.DataFrame(merged)
    summaries["input_tokens"]  = summaries["input_tokens"].astype("int32")
    summaries["output_tokens"] = summaries["output_tokens"].astype("int32")
    summaries.to_parquet(OUTPUT_DIR / "summaries.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    total_cost = _cost(total_input_tokens, total_output_tokens)

    summarized_df = summaries[summaries["summary"].notna()]
    avg_summary_words = (
        summarized_df["summary"].str.split().str.len().mean()
        if not summarized_df.empty else 0
    )

    report = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "model":                 MODEL,
        "total_bursts":          total,
        "summarized":            summarized,
        "cached_hits":           cached_hits,
        "skipped": {
            "too_short":  skipped_too_short,
            "monologue":  skipped_monologue,
            "error":      skipped_error,
        },
        "total_input_tokens":    total_input_tokens,
        "total_output_tokens":   total_output_tokens,
        "total_cost_usd":        round(total_cost, 4),
        "avg_summary_length_words": round(avg_summary_words, 1),
    }
    (OUTPUT_DIR / "summarization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\n--- Summarization complete ---")
    print(f"  Total bursts:       {total:,}")
    print(f"  Summarized:         {summarized:,}  ({cached_hits:,} from cache)")
    print(f"  Skipped too short:  {skipped_too_short:,}")
    print(f"  Skipped monologue:  {skipped_monologue:,}")
    print(f"  Errors:             {skipped_error:,}")
    print(f"  Input tokens:       {total_input_tokens:,}")
    print(f"  Output tokens:      {total_output_tokens:,}")
    print(f"  Total cost:         ${total_cost:.4f}")
    print(f"  Avg summary length: {avg_summary_words:.1f} words")
    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print(f"  summaries.parquet")
    print(f"  llm_cache/           ({len(list(CACHE_DIR.glob('*.json'))):,} cached bursts)")
    print(f"  summarization_report.json")
    print(f"\nNext: python message_miner/12_classify.py")


if __name__ == "__main__":
    main()
