"""
11c_classify.py — LLM Conversation Classification

Single-pass LLM classification over conversation summaries. One API call per
summary extracts function (12 categories), depth (1-5), decisions, and action
items simultaneously.

Uses async API calls with configurable concurrency for throughput.

Reads:
  outputs/11_summarize/summaries.parquet
  outputs/07_segment/bursts.parquet

Writes:
  outputs/11c_classify/
  ├── classifications.parquet      one row per burst
  ├── decisions.parquet            one row per extracted decision
  ├── action_items.parquet         one row per extracted action item
  ├── llm_cache/                   one JSON file per burst_id (resume support)
  └── classification_report.json

Run from project root:
    python message_miner/11c_classify.py [--force] [--concurrency N]
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

# ── Paths ──────────────────────────────────────────────────────────────────────

SUMMARIES_PATH = Path("outputs/11_summarize/summaries.parquet")
BURSTS_PATH    = Path("outputs/07_segment/bursts.parquet")
OUTPUT_DIR     = Path("outputs/11c_classify")
CACHE_DIR      = OUTPUT_DIR / "llm_cache"

# ── Model & cost ───────────────────────────────────────────────────────────────

MODEL                = "gpt-5.4-nano"
TEMPERATURE          = 0
MAX_RETRIES          = 5
INPUT_COST_PER_MTOK  = 0.20   # $ per 1M input tokens
OUTPUT_COST_PER_MTOK = 1.25   # $ per 1M output tokens

VALID_FUNCTIONS = frozenset({
    "decision_making", "troubleshooting", "status_update", "scheduling",
    "knowledge_sharing", "feedback_review", "social_casual", "escalation",
    "delegation", "brainstorming", "information_request", "announcement",
})

PROGRESS_INTERVAL = 250

# ── Prompt ─────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Analyze this workplace chat conversation. Return a JSON object with exactly these fields:

"function": The primary purpose. Choose ONE of: decision_making, troubleshooting, status_update, scheduling, knowledge_sharing, feedback_review, social_casual, escalation, delegation, brainstorming, information_request, announcement

"depth": Substantiveness rating, integer 1-5:
1 = trivial (scheduling, one-word acknowledgments)
2 = routine coordination
3 = substantive discussion with nuance
4 = deep collaboration or complex problem-solving
5 = critical decision or strategic discussion

"decisions": Array of strings describing specific decisions made. Empty array [] if none.

"action_items": Array of objects with "owner" (person name) and "task" (what they committed to do). Work-related tasks only — exclude personal items such as appointments, personal errands, or non-work commitments. Empty array [] if none.

Participants: {participant_names}
Summary: {summary_text}

Return ONLY valid JSON."""


# ── Cost helper ────────────────────────────────────────────────────────────────

def _cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens  / 1_000_000 * INPUT_COST_PER_MTOK +
        output_tokens / 1_000_000 * OUTPUT_COST_PER_MTOK
    )


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _safe_name(burst_id: str) -> str:
    return burst_id.replace(":", "_").replace("/", "_").replace("\\", "_")


def _cache_path(burst_id: str) -> Path:
    return CACHE_DIR / f"{_safe_name(burst_id)}.json"


def _load_cache(burst_id: str) -> dict | None:
    path = _cache_path(burst_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _save_cache(burst_id: str, data: dict) -> None:
    _cache_path(burst_id).write_text(json.dumps(data), encoding="utf-8")


# ── Response parsing & validation ──────────────────────────────────────────────

def _validate(raw: dict) -> dict:
    """
    Validate and normalize a parsed JSON response dict.
    All fields have well-defined fallbacks for malformed values.
    """
    # function: must be one of 12 values
    func = raw.get("function", "")
    if not isinstance(func, str) or func not in VALID_FUNCTIONS:
        func = "unknown"

    # depth: integer 1-5
    depth_raw = raw.get("depth")
    try:
        depth = int(depth_raw)
        if depth < 1 or depth > 5:
            depth = 3
    except (TypeError, ValueError):
        depth = 3

    # decisions: array of strings
    dec_raw = raw.get("decisions", [])
    if isinstance(dec_raw, list):
        decisions = [str(d) for d in dec_raw if isinstance(d, str) and str(d).strip()]
    else:
        decisions = []

    # action_items: array of {owner, task}
    ai_raw = raw.get("action_items", [])
    if isinstance(ai_raw, list):
        action_items = [
            {"owner": str(item["owner"]), "task": str(item["task"])}
            for item in ai_raw
            if isinstance(item, dict) and "owner" in item and "task" in item
        ]
    else:
        action_items = []

    return {
        "function":     func,
        "depth":        depth,
        "decisions":    decisions,
        "action_items": action_items,
    }


def _parse_response(content: str) -> dict | None:
    """
    Parse and validate JSON response content.
    Returns validated dict, or None if JSON parsing fails entirely.
    """
    try:
        raw = json.loads(content)
        return _validate(raw)
    except (json.JSONDecodeError, ValueError):
        return None


def _resolve_owner(owner: str, participants: list[str]) -> str | None:
    """
    Match an owner name from the LLM to a canonical participant name.
    Handles full names, first-name-only, and partial matches.
    Returns the canonical participant name, or None if no match found.
    """
    owner_lower = owner.lower().strip()
    if not owner_lower:
        return None

    # Pass 1: exact match
    for p in participants:
        if owner_lower == p.lower().strip():
            return p

    # Pass 2: owner is a first or last name of a participant
    for p in participants:
        parts = p.lower().split()
        if owner_lower in parts:
            return p

    # Pass 3: any participant name part appears in owner (e.g. "davey" ~ "David")
    # Only accept if the substring is at least 4 chars to avoid false positives
    for p in participants:
        parts = p.lower().split()
        for part in parts:
            if len(part) >= 4 and (part in owner_lower or owner_lower in part):
                return p

    return None


def _filter_action_items(action_items: list[dict], participants: list[str]) -> list[dict]:
    """Resolve owners to canonical participant names; drop items with no match."""
    result = []
    for item in action_items:
        canonical = _resolve_owner(item["owner"], participants)
        if canonical is not None:
            result.append({"owner": canonical, "task": item["task"]})
    return result


# ── Async processing function ─────────────────────────────────────────────────

async def _classify_one(client, item: dict) -> dict:
    """
    Classify a single burst asynchronously.

    Parameters
    ----------
    client : AsyncOpenAI
        Shared async client from process_batch.
    item : dict
        Must contain 'burst_id' and 'prompt'.

    Returns
    -------
    dict with function, depth, decisions, action_items, input_tokens, output_tokens
    """
    burst_id = item["burst_id"]
    prompt   = item["prompt"]

    result  = None
    in_tok  = 0
    out_tok = 0

    for attempt in range(2):
        try:
            content, usage = await async_call_api(
                client,
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                response_format={"type": "json_object"},
                max_retries=MAX_RETRIES,
            )
            in_tok  = usage.prompt_tokens
            out_tok = usage.completion_tokens
            parsed  = _parse_response(content)
            if parsed is not None:
                result = parsed
                break
            print(
                f"    [warn] JSON parse failed for {burst_id[:40]} "
                f"(attempt {attempt + 1}) — retrying..."
            )
        except Exception as e:
            print(f"    [error] burst {burst_id[:40]}: {e}")
            break

    if result is None:
        print(f"    [error] giving up on {burst_id[:40]}")
        result = {
            "function":     "error",
            "depth":        None,
            "decisions":    [],
            "action_items": [],
        }
        result["_error"] = True
    else:
        result["_error"] = False

    result["input_tokens"]  = in_tok
    result["output_tokens"] = out_tok

    _save_cache(burst_id, {
        "function":     result["function"],
        "depth":        result["depth"],
        "decisions":    result["decisions"],
        "action_items": result["action_items"],
        "input_tokens":  in_tok,
        "output_tokens": out_tok,
    })

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM conversation classification")
    parser.add_argument("--force", action="store_true",
                        help="Clear cache and re-classify everything")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls (default: 10)")
    args = parser.parse_args()

    print("=== message-miner: 11c_classify ===\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

    if args.force:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print("Cache cleared.\n")

    # ── Check API key ────────────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set. Add it to .env and re-run.")
        sys.exit(1)

    # ── Check inputs ─────────────────────────────────────────────────────────────
    for path in (SUMMARIES_PATH, BURSTS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────────
    print("Loading data...")
    summaries = pd.read_parquet(SUMMARIES_PATH)
    bursts    = pd.read_parquet(BURSTS_PATH)
    print(f"  {len(summaries):,} summary rows")
    print(f"  {len(bursts):,} bursts")

    # ── Filter: only bursts with a non-null summary and no skip_reason ────────────
    to_classify = summaries[
        summaries["summary"].notna() &
        summaries["skip_reason"].isna()
    ].copy()
    print(f"  {len(to_classify):,} bursts eligible for classification\n")

    # ── Build burst lookup: burst_id → {chat_id, participants, start_time} ────────
    burst_lookup = (
        bursts[["burst_id", "chat_id", "participants", "start_time"]]
        .set_index("burst_id")
        .to_dict("index")
    )

    # ══════════════════════════════════════════════════════════════════════════════
    # Phase 1 (sync): Separate cached vs uncached items
    # ══════════════════════════════════════════════════════════════════════════════

    # Each entry: (burst_id, chat_id, participants, start_time, result_or_None)
    all_items: list[dict] = []
    uncached_items: list[dict] = []

    total_input_tokens  = 0
    total_output_tokens = 0
    cached_hits         = 0
    total               = len(to_classify)

    print(f"Checking cache for {total:,} bursts...")

    for _, summary_row in to_classify.iterrows():
        burst_id     = summary_row["burst_id"]
        chat_id      = summary_row["chat_id"]
        burst_meta   = burst_lookup.get(burst_id, {})
        participants = str(burst_meta.get("participants") or "Unknown")
        start_time   = burst_meta.get("start_time")

        entry = {
            "burst_id":     burst_id,
            "chat_id":      chat_id,
            "participants": participants,
            "start_time":   start_time,
        }

        cached = _load_cache(burst_id)
        if cached is not None:
            entry["result"]  = cached
            in_tok  = cached.get("input_tokens",  0)
            out_tok = cached.get("output_tokens", 0)
            total_input_tokens  += in_tok
            total_output_tokens += out_tok
            cached_hits += 1
        else:
            prompt = _PROMPT_TEMPLATE.format(
                participant_names=participants,
                summary_text=summary_row["summary"],
            )
            entry["result"] = None
            entry["prompt"] = prompt
            uncached_items.append(entry)

        all_items.append(entry)

    print(f"  {cached_hits:,} cached, {len(uncached_items):,} to classify\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # Phase 2 (async): Process uncached items concurrently
    # ══════════════════════════════════════════════════════════════════════════════

    classified = cached_hits
    errors     = 0

    if uncached_items:
        print(f"Classifying {len(uncached_items):,} bursts "
              f"(concurrency={args.concurrency})...\n")

        async_results = asyncio.run(
            process_batch(
                items=uncached_items,
                process_fn=_classify_one,
                api_key=api_key,
                concurrency=args.concurrency,
                desc="Classifying",
            )
        )

        # Merge async results back into all_items
        uncached_idx = 0
        for entry in all_items:
            if entry["result"] is None:
                result = async_results[uncached_idx]
                uncached_idx += 1

                entry["result"] = result
                total_input_tokens  += result.get("input_tokens",  0)
                total_output_tokens += result.get("output_tokens", 0)

                if result.get("_error", False):
                    errors += 1
                else:
                    classified += 1
    else:
        print("All bursts served from cache — no API calls needed.\n")

    # ══════════════════════════════════════════════════════════════════════════════
    # Phase 3 (sync): Merge results, build DataFrames, write outputs
    # ══════════════════════════════════════════════════════════════════════════════

    classification_rows: list[dict] = []
    decision_rows:       list[dict] = []
    action_item_rows:    list[dict] = []

    for entry in all_items:
        burst_id     = entry["burst_id"]
        chat_id      = entry["chat_id"]
        participants = entry["participants"]
        start_time   = entry["start_time"]
        result       = entry["result"]

        # Resolve owners to canonical participant names
        participant_list = [p.strip() for p in participants.split(",") if p.strip()]
        clean_action_items = _filter_action_items(result["action_items"], participant_list)

        classification_rows.append({
            "burst_id":          burst_id,
            "chat_id":           chat_id,
            "function":          result["function"],
            "depth":             result["depth"],
            "decision_count":    len(result["decisions"]),
            "action_item_count": len(clean_action_items),
        })

        for decision_text in result["decisions"]:
            decision_rows.append({
                "burst_id":      burst_id,
                "chat_id":       chat_id,
                "decision_text": decision_text,
                "participants":  participants,
                "timestamp":     start_time,
            })

        for ai in clean_action_items:
            action_item_rows.append({
                "burst_id":     burst_id,
                "chat_id":      chat_id,
                "owner":        ai["owner"],
                "task":         ai["task"],
                "participants": participants,
                "timestamp":    start_time,
            })

    # ── Build DataFrames ──────────────────────────────────────────────────────────
    print(f"\nBuilding output tables...")

    classifications_df = pd.DataFrame(classification_rows)
    classifications_df["decision_count"]    = classifications_df["decision_count"].astype("int32")
    classifications_df["action_item_count"] = classifications_df["action_item_count"].astype("int32")
    # depth: nullable int8 (error rows have None)
    classifications_df["depth"] = pd.array(
        classifications_df["depth"], dtype=pd.Int8Dtype()
    )
    classifications_df["function"] = classifications_df["function"].astype("category")

    decisions_df = pd.DataFrame(decision_rows)
    if not decisions_df.empty:
        decisions_df["timestamp"] = pd.to_datetime(
            decisions_df["timestamp"], utc=True, errors="coerce"
        )

    action_items_df = pd.DataFrame(action_item_rows)
    if not action_items_df.empty:
        action_items_df["timestamp"] = pd.to_datetime(
            action_items_df["timestamp"], utc=True, errors="coerce"
        )

    # ── Write outputs ─────────────────────────────────────────────────────────────
    print("Writing outputs...")
    classifications_df.to_parquet(OUTPUT_DIR / "classifications.parquet", index=False)
    print(f"  classifications.parquet   ({len(classifications_df):,} rows)")

    decisions_df.to_parquet(OUTPUT_DIR / "decisions.parquet", index=False)
    print(f"  decisions.parquet         ({len(decisions_df):,} rows)")

    action_items_df.to_parquet(OUTPUT_DIR / "action_items.parquet", index=False)
    print(f"  action_items.parquet      ({len(action_items_df):,} rows)")

    # ── classification_report.json ────────────────────────────────────────────────
    total_cost = _cost(total_input_tokens, total_output_tokens)

    func_dist = (
        classifications_df["function"]
        .value_counts()
        .to_dict()
    )
    # Ensure string keys (category dtype values → str)
    func_dist = {str(k): int(v) for k, v in func_dist.items()}

    depth_dist = (
        classifications_df["depth"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    depth_dist = {str(k): int(v) for k, v in depth_dist.items()}

    report = {
        "generated_at":                datetime.now(timezone.utc).isoformat(),
        "model":                       MODEL,
        "total_classified":            classified,
        "errors":                      errors,
        "cached_hits":                 cached_hits,
        "function_distribution":       func_dist,
        "depth_distribution":          depth_dist,
        "total_decisions_extracted":   len(decisions_df),
        "total_action_items_extracted": len(action_items_df),
        "total_input_tokens":          total_input_tokens,
        "total_output_tokens":         total_output_tokens,
        "total_cost_usd":              round(total_cost, 4),
    }
    (OUTPUT_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("  classification_report.json\n")

    # ── Console summary ───────────────────────────────────────────────────────────
    print("-" * 52)
    print("  11c_classify — complete")
    print("-" * 52)
    print(f"  Eligible bursts:           {total:>8,}")
    print(f"  Classified:                {classified:>8,}  ({cached_hits:,} from cache)")
    print(f"  Errors:                    {errors:>8,}")
    print(f"  Decisions extracted:       {len(decisions_df):>8,}")
    print(f"  Action items extracted:    {len(action_items_df):>8,}")
    print(f"  Input tokens:              {total_input_tokens:>8,}")
    print(f"  Output tokens:             {total_output_tokens:>8,}")
    print(f"  Total cost:                ${total_cost:.4f}")
    print()

    if func_dist:
        top_funcs = sorted(func_dist.items(), key=lambda x: x[1], reverse=True)[:6]
        print("  Top conversation functions:")
        for func, count in top_funcs:
            pct = count / classified * 100 if classified > 0 else 0
            print(f"    {func:<25} {count:>6,}  ({pct:.1f}%)")
        print()

    if depth_dist:
        print("  Depth distribution:")
        for depth_val in sorted(depth_dist):
            count = depth_dist[depth_val]
            pct   = count / classified * 100 if classified > 0 else 0
            bar   = "█" * int(pct / 2)
            print(f"    {depth_val}  {count:>6,}  ({pct:4.1f}%)  {bar}")
        print()

    cache_count = len(list(CACHE_DIR.glob("*.json")))
    print(f"Outputs written to {OUTPUT_DIR}/")
    print(f"  classifications.parquet    ({len(classifications_df):,} rows)")
    print(f"  decisions.parquet          ({len(decisions_df):,} rows)")
    print(f"  action_items.parquet       ({len(action_items_df):,} rows)")
    print(f"  llm_cache/                 ({cache_count:,} cached bursts)")
    print(f"  classification_report.json")
    print(f"\nNext: python message_miner/12_wrapped.py")


if __name__ == "__main__":
    main()
