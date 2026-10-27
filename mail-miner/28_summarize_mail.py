"""
28_summarize_mail.py — Email Thread Summarization

Summarizes each active email thread in 1-3 sentences using an LLM.
One API call per thread. Resume-safe via per-thread JSON cache.

Reads:
  outputs/27_segment_mail/email_threads.parquet

Writes:
  outputs/28_summarize_mail/
    email_summaries.parquet     -- one row per thread (all threads, not just summarized)
    llm_cache/                  -- one JSON file per thread_id
    summarization_report.json   -- stats and cost

Run from project root:
    python mail-miner/28_summarize_mail.py [--force] [--concurrency 10]
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

THREADS_PATH = Path("outputs/27_segment_mail/email_threads.parquet")
OUTPUT_DIR   = Path("outputs/28_summarize_mail")
CACHE_DIR    = OUTPUT_DIR / "llm_cache"

MODEL                 = "gpt-5.4-nano"
TEMPERATURE           = 0
MAX_TOKENS            = 300
INPUT_COST_PER_MTOK   = 0.05
OUTPUT_COST_PER_MTOK  = 0.40
MIN_WORD_COUNT        = 15
MAX_INPUT_WORDS       = 4000


# ── Prompt ────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Summarize this email thread in 1-3 sentences. Focus on: what was discussed, what was decided or requested, and any commitments or next steps. Include participant names.

Subject: {subject}
Participants: {participants}

{thread_text}

Summary:"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1_000_000 * INPUT_COST_PER_MTOK +
        output_tokens / 1_000_000 * OUTPUT_COST_PER_MTOK
    )


def _cache_path(thread_id: str) -> Path:
    safe = thread_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    return CACHE_DIR / f"{safe}.json"


def _load_cache(thread_id: str) -> dict | None:
    path = _cache_path(thread_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(thread_id: str, data: dict) -> None:
    _cache_path(thread_id).write_text(json.dumps(data), encoding="utf-8")


def _build_prompt(row: pd.Series) -> str:
    subject = row.get("subject") or "(no subject)"
    participants = row.get("participants") or ""
    thread_text = row.get("thread_text") or ""

    # Safety truncation to 4000 words
    words = thread_text.split()
    if len(words) > MAX_INPUT_WORDS:
        thread_text = " ".join(words[:MAX_INPUT_WORDS])

    return _PROMPT_TEMPLATE.format(
        subject=subject,
        participants=participants,
        thread_text=thread_text,
    )


# ── Async worker ──────────────────────────────────────────────────────────────

async def _process_one(client, item: dict) -> dict:
    thread_id = item["thread_id"]
    prompt    = item["prompt"]

    try:
        summary, usage = await async_call_api(
            client,
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        summary = (summary or "").strip() or None
        in_tok  = usage.prompt_tokens
        out_tok = usage.completion_tokens

        _save_cache(thread_id, {
            "thread_id":     thread_id,
            "summary":       summary,
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
        })

        return {
            "thread_id":     thread_id,
            "summary":       summary,
            "skip_reason":   None,
            "input_tokens":  in_tok,
            "output_tokens": out_tok,
        }
    except Exception as e:
        print(f"    [error] thread {thread_id}: {e}")
        return {
            "thread_id":     thread_id,
            "summary":       None,
            "skip_reason":   "error",
            "input_tokens":  0,
            "output_tokens": 0,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize email threads via LLM")
    parser.add_argument("--force", action="store_true", help="Re-summarize everything, ignoring cache")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls (default: 10)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    print("=== message-miner: 28_summarize_mail ===\n")

    if not THREADS_PATH.exists():
        print(f"ERROR: {THREADS_PATH} not found. Run 27_segment_mail.py first.")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)

    if args.force:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print("Cache cleared.\n")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading threads...")
    threads = pd.read_parquet(THREADS_PATH)
    total_threads = len(threads)
    print(f"  {total_threads:,} total threads")

    # Eligible: skip==False and word_count >= 15
    eligible_mask = (~threads["skip"].fillna(False)) & (threads["word_count"].fillna(0) >= MIN_WORD_COUNT)
    eligible = threads[eligible_mask]
    ineligible = threads[~eligible_mask]

    skip_single  = int((threads["skip_reason"] == "single_inbound").sum())
    skip_nopart  = int((threads["skip_reason"] == "no_participation").sum())
    skip_short   = int(((~threads["skip"].fillna(False)) & (threads["word_count"].fillna(0) < MIN_WORD_COUNT)).sum())

    print(f"  {len(eligible):,} eligible for summarization")
    print(f"  {total_threads - len(eligible):,} skipped ({skip_single:,} single inbound, {skip_nopart:,} no participation, {skip_short:,} too short)")

    # ── Phase 1: classify each eligible thread ────────────────────────────────
    print(f"\nSummarizing {len(eligible):,} threads...")

    results       = {}   # thread_id -> result dict (skips + cached)
    uncached_items = []
    uncached_order = []  # thread_ids in async submission order

    total_input_tokens  = 0
    total_output_tokens = 0
    cached_hits = 0

    for _, row in eligible.iterrows():
        thread_id = row["thread_id"]

        cached = _load_cache(thread_id)
        if cached is not None:
            results[thread_id] = {
                "thread_id":     thread_id,
                "summary":       cached["summary"],
                "skip_reason":   None,
                "input_tokens":  cached["input_tokens"],
                "output_tokens": cached["output_tokens"],
            }
            total_input_tokens  += cached["input_tokens"]
            total_output_tokens += cached["output_tokens"]
            cached_hits += 1
            continue

        prompt = _build_prompt(row)
        uncached_items.append({"thread_id": thread_id, "prompt": prompt})
        uncached_order.append(thread_id)

    print(f"  {cached_hits:,} from cache, {len(uncached_items):,} need API calls\n")

    # ── Phase 2: async API calls ──────────────────────────────────────────────
    async_results = []
    if uncached_items:
        async_results = asyncio.run(
            process_batch(
                items=uncached_items,
                process_fn=_process_one,
                api_key=api_key,
                concurrency=args.concurrency,
                desc="Summarizing",
            )
        )
        for r in async_results:
            results[r["thread_id"]] = r
            total_input_tokens  += r["input_tokens"]
            total_output_tokens += r["output_tokens"]

    # ── Phase 3: build output for ALL threads ─────────────────────────────────
    rows = []
    for _, thread in threads.iterrows():
        thread_id = thread["thread_id"]
        base = {
            "thread_id":    thread_id,
            "subject":      thread.get("subject"),
            "participants": thread.get("participants"),
            "msg_count":    thread.get("msg_count"),
            "start_time":   thread.get("start_time"),
        }

        if thread_id in results:
            r = results[thread_id]
            base.update({
                "summary":       r["summary"],
                "skip_reason":   r["skip_reason"],
                "input_tokens":  r["input_tokens"],
                "output_tokens": r["output_tokens"],
            })
        else:
            # Was ineligible (skipped before API phase)
            reason = thread.get("skip_reason")
            if not reason and thread["word_count"] < MIN_WORD_COUNT:
                reason = "too_short"
            base.update({
                "summary":       None,
                "skip_reason":   reason,
                "input_tokens":  0,
                "output_tokens": 0,
            })
        rows.append(base)

    out_df = pd.DataFrame(rows)
    out_df["msg_count"]      = out_df["msg_count"].astype("Int32")
    out_df["input_tokens"]   = out_df["input_tokens"].astype("Int32")
    out_df["output_tokens"]  = out_df["output_tokens"].astype("Int32")

    # ── Write ─────────────────────────────────────────────────────────────────
    out_df.to_parquet(OUTPUT_DIR / "email_summaries.parquet", index=False)

    # ── Report ────────────────────────────────────────────────────────────────
    summarized   = int(out_df["summary"].notna().sum())
    errors       = int((out_df["skip_reason"] == "error").sum())
    total_cost   = _cost(total_input_tokens, total_output_tokens)

    in_tok_series  = out_df[out_df["input_tokens"] > 0]["input_tokens"]
    out_tok_series = out_df[out_df["output_tokens"] > 0]["output_tokens"]

    report = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "model":              MODEL,
        "total_threads":      total_threads,
        "eligible_threads":   len(eligible),
        "summarized":         summarized,
        "cached_hits":        cached_hits,
        "skipped":            total_threads - len(eligible),
        "errors":             errors,
        "skip_reasons": {
            "single_inbound":   skip_single,
            "no_participation": skip_nopart,
            "too_short":        skip_short,
            "error":            errors,
        },
        "avg_input_tokens":   round(float(in_tok_series.mean()), 0) if not in_tok_series.empty else 0,
        "avg_output_tokens":  round(float(out_tok_series.mean()), 0) if not out_tok_series.empty else 0,
        "total_input_tokens":  total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd":      round(total_cost, 4),
    }
    (OUTPUT_DIR / "summarization_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"\nDone.")
    print(f"  Wrote {OUTPUT_DIR}/email_summaries.parquet ({total_threads:,} rows)")
    print(f"  Wrote {OUTPUT_DIR}/summarization_report.json")
    print(f"  Summarized: {summarized:,}  ({cached_hits:,} from cache)")
    print(f"  Errors:     {errors:,}")
    print(f"  Total cost: ${total_cost:.4f}")
    print(f"\nNext: python mail-miner/29_classify_mail.py")


if __name__ == "__main__":
    main()
