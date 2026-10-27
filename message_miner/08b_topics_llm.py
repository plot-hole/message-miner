"""
08b_topics_llm.py — LLM Topic Classification

Two-pass approach:
  Pass 1: Sample 400 summaries → GPT-5.4 → topic taxonomy
  Pass 2: Classify every summary with GPT-5.4-nano using the taxonomy

Reads:
  outputs/11_summarize/summaries.parquet
  outputs/07_segment/bursts.parquet

Writes:
  outputs/08b_topics_llm/
  ├── taxonomy.json                # Generated topic taxonomy
  ├── burst_topics_llm.parquet     # Per-burst topic assignments
  ├── topic_info_llm.json          # Topic labels/stats (read by 12_wrapped.py)
  ├── llm_cache/                   # One JSON per burst_id (resume support)
  └── topics_llm_report.json       # Stats, cost, metadata

Run from project root:
    python message_miner/08b_topics_llm.py                 # Normal run
    python message_miner/08b_topics_llm.py --force          # Re-classify (keep taxonomy)
    python message_miner/08b_topics_llm.py --force-taxonomy # Regen taxonomy + re-classify
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from message_miner.helpers.async_llm import async_call_api, process_batch

# ── Paths ───────────────────────────────────────────────────────────────────────

SUMMARIES_PATH = Path("outputs/11_summarize/summaries.parquet")
BURSTS_PATH    = Path("outputs/07_segment/bursts.parquet")
OUTPUT_DIR     = Path("outputs/08b_topics_llm")
CACHE_DIR      = OUTPUT_DIR / "llm_cache"
TAXONOMY_PATH  = OUTPUT_DIR / "taxonomy.json"

# ── Pass 1 constants (gpt-5.4) ──────────────────────────────────────────────────

TAXONOMY_MODEL          = "gpt-5.4"
TAXONOMY_TEMPERATURE    = 0.3
TAXONOMY_MAX_TOKENS     = 2000
TAXONOMY_SAMPLE_SIZE    = 400
P1_INPUT_COST_PER_MTOK  = 2.50
P1_OUTPUT_COST_PER_MTOK = 15.00

# ── Pass 2 constants (gpt-5.4-nano) ────────────────────────────────────────────

CLASSIFY_MODEL          = "gpt-5.4-nano"
CLASSIFY_TEMPERATURE    = 0
CLASSIFY_MAX_TOKENS     = 200
P2_INPUT_COST_PER_MTOK  = 0.05
P2_OUTPUT_COST_PER_MTOK = 0.40

MAX_RETRIES       = 5
PROGRESS_INTERVAL = 500

# ── Cost helpers ────────────────────────────────────────────────────────────────

def _p1_cost(in_tok: int, out_tok: int) -> float:
    return in_tok / 1_000_000 * P1_INPUT_COST_PER_MTOK + out_tok / 1_000_000 * P1_OUTPUT_COST_PER_MTOK

def _p2_cost(in_tok: int, out_tok: int) -> float:
    return in_tok / 1_000_000 * P2_INPUT_COST_PER_MTOK + out_tok / 1_000_000 * P2_OUTPUT_COST_PER_MTOK

# ── Cache helpers ───────────────────────────────────────────────────────────────

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

# ── API call with retry ─────────────────────────────────────────────────────────

def _call_api(client, model: str, temperature: float, max_tokens: int,
              messages: list, use_json_format: bool = True) -> tuple[str, object]:
    kwargs = dict(
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        messages=messages,
    )
    if use_json_format:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip(), response.usage
        except Exception as e:
            err = str(e)
            if ("429" in err or "rate_limit" in err.lower()) and attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"    Rate limited -- waiting {wait}s (retry {attempt + 2}/{MAX_RETRIES})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

# ── Pass 1: Taxonomy generation ─────────────────────────────────────────────────

_TAXONOMY_PROMPT = """\
You are analyzing workplace chat conversations to create a topic taxonomy. Below are {n} conversation summaries from a single person's messaging history at a company.

Your task: propose a topic taxonomy of 15-25 categories that naturally describe the work happening in these conversations. The categories should be:

- Specific enough to be meaningful (not "General Discussion")
- Broad enough that each category covers multiple conversations
- Mutually understandable by anyone in a workplace
- Based on the ACTUAL content you see, not generic workplace categories

For each category, provide:
- A short label (2-4 words)
- A one-sentence description of what conversations in this category are about
- 3-5 example keywords or phrases you'd expect to see

Return ONLY valid JSON in this format:
{{
  "taxonomy": [
    {{
      "id": 1,
      "label": "Security Incident Response",
      "description": "Conversations about investigating, triaging, or resolving security incidents and alerts.",
      "keywords": ["incident", "alert", "remediation", "phishing", "compromise"]
    }}
  ]
}}

Here are the {n} conversation summaries:

{summaries_block}"""


def run_pass1(client, eligible: pd.DataFrame, burst_lookup: dict) -> list[dict]:
    """Sample summaries, call GPT-5.4, return taxonomy list."""
    sample = eligible.sample(
        n=min(TAXONOMY_SAMPLE_SIZE, len(eligible)),
        random_state=42
    )

    lines = []
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        meta = burst_lookup.get(row["burst_id"], {})
        participants = str(meta.get("participants") or "Unknown")
        lines.append(f"{i}. [Participants: {participants}] {row['summary']}")
    summaries_block = "\n".join(lines)

    prompt = _TAXONOMY_PROMPT.format(n=len(sample), summaries_block=summaries_block)

    print(f"Pass 1: generating taxonomy from {len(sample)} sampled summaries...")
    print(f"  Model: {TAXONOMY_MODEL}  temperature={TAXONOMY_TEMPERATURE}")

    content, usage = _call_api(
        client, TAXONOMY_MODEL, TAXONOMY_TEMPERATURE, TAXONOMY_MAX_TOKENS,
        [{"role": "user", "content": prompt}],
        use_json_format=True,
    )

    in_tok  = usage.prompt_tokens
    out_tok = usage.completion_tokens
    cost    = _p1_cost(in_tok, out_tok)
    print(f"  Tokens: {in_tok} in / {out_tok} out  |  Cost: ${cost:.4f}")

    parsed = json.loads(content)
    taxonomy = parsed.get("taxonomy", [])

    # Validate entries
    valid = []
    for t in taxonomy:
        if all(k in t for k in ("id", "label", "description", "keywords")):
            valid.append(t)
        else:
            print(f"  [warn] skipping malformed taxonomy entry: {t}")

    return valid, in_tok, out_tok


# ── Pass 2: Classification ───────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
Classify this workplace conversation into 1-2 topics from the taxonomy below.

Taxonomy:
{taxonomy_block}

Return ONLY valid JSON:
{{
  "primary_topic": <topic_id>,
  "secondary_topic": <topic_id or null>
}}

Choose the single best match as primary_topic. Only set secondary_topic if the conversation clearly spans two distinct topics. When in doubt, use only primary_topic and set secondary_topic to null.

Participants: {participant_names}
Summary: {summary_text}"""


def _build_taxonomy_block(taxonomy: list[dict]) -> str:
    return "\n".join(
        f"{t['id']}. {t['label']} -- {t['description']}"
        for t in taxonomy
    )


def _validate_classification(raw: dict, valid_ids: set[int]) -> dict:
    primary = raw.get("primary_topic")
    secondary = raw.get("secondary_topic")

    try:
        primary = int(primary)
        if primary not in valid_ids:
            primary = None
    except (TypeError, ValueError):
        primary = None

    try:
        secondary = int(secondary) if secondary is not None else None
        if secondary is not None and secondary not in valid_ids:
            secondary = None
        if secondary == primary:
            secondary = None
    except (TypeError, ValueError):
        secondary = None

    return {"primary_topic": primary, "secondary_topic": secondary}


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM topic taxonomy + classification")
    parser.add_argument("--force", action="store_true",
                        help="Re-classify all bursts (keep existing taxonomy)")
    parser.add_argument("--force-taxonomy", action="store_true",
                        help="Regenerate taxonomy AND re-classify all bursts")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent async API calls for Pass 2 (default: 10)")
    args = parser.parse_args()

    if args.force_taxonomy:
        args.force = True  # taxonomy change means all cache is stale

    print("=" * 60)
    print("  08b_topics_llm -- LLM Topic Classification")
    print("=" * 60)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)

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

    eligible = summaries[
        summaries["summary"].notna() &
        summaries["skip_reason"].isna()
    ].copy()
    print(f"  {len(eligible):,} bursts eligible for classification")
    print()

    burst_lookup = (
        bursts[["burst_id", "chat_id", "participants", "start_time"]]
        .set_index("burst_id")
        .to_dict("index")
    )

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # ── Pass 1: Taxonomy ─────────────────────────────────────────────────────────
    p1_in_tok = p1_out_tok = 0

    if TAXONOMY_PATH.exists() and not args.force_taxonomy:
        print(f"Taxonomy already exists at {TAXONOMY_PATH} -- skipping Pass 1.")
        print("  (Use --force-taxonomy to regenerate.)")
        taxonomy = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
        print(f"  Loaded {len(taxonomy)} categories.")
        print()
    else:
        if args.force_taxonomy:
            print("--force-taxonomy: regenerating taxonomy and clearing classification cache.")
            for f in CACHE_DIR.glob("*.json"):
                f.unlink()
            print(f"  Classification cache cleared.")
            print()

        taxonomy, p1_in_tok, p1_out_tok = run_pass1(client, eligible, burst_lookup)

        TAXONOMY_PATH.write_text(
            json.dumps(taxonomy, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"  Taxonomy saved ({len(taxonomy)} categories): {TAXONOMY_PATH}")
        print()

        # Print taxonomy for user review
        print("-" * 60)
        print("  GENERATED TAXONOMY")
        print("-" * 60)
        for t in taxonomy:
            print(f"  {t['id']:>2}. {t['label']}")
            print(f"      {t['description']}")
            print(f"      Keywords: {', '.join(t['keywords'])}")
            print()

        print("-" * 60)
        try:
            input(
                f"Taxonomy generated with {len(taxonomy)} categories. Review above.\n"
                "Press Enter to proceed with classification, or Ctrl+C to abort and edit taxonomy.json manually.\n"
            )
        except KeyboardInterrupt:
            print("\nAborted. Edit taxonomy.json then re-run (taxonomy will be reused).")
            sys.exit(0)

    if not taxonomy:
        print("ERROR: Taxonomy is empty. Edit taxonomy.json and re-run.")
        sys.exit(1)

    valid_ids = {int(t["id"]) for t in taxonomy}
    id_to_label = {int(t["id"]): t["label"] for t in taxonomy}
    taxonomy_block = _build_taxonomy_block(taxonomy)

    # ── Pass 2: Classification ───────────────────────────────────────────────────
    if args.force and not args.force_taxonomy:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink()
        print("Classification cache cleared (--force).\n")

    total = len(eligible)
    print(f"Pass 2: classifying {total:,} bursts...")
    print(f"  Model: {CLASSIFY_MODEL}  temperature={CLASSIFY_TEMPERATURE}")
    print(f"  Concurrency: {args.concurrency}")
    print()

    # ── Phase 1 (sync): check cache, build work list ────────────────────────────
    rows: list[dict] = []           # will hold ALL results (cached + async)
    cached_results: list[dict] = [] # cached results with ordering index
    uncached_items: list[dict] = [] # items needing API calls
    p2_in_tok = p2_out_tok = 0
    cached_hits = 0

    for idx, (_, summary_row) in enumerate(eligible.iterrows()):
        burst_id = summary_row["burst_id"]
        chat_id  = summary_row["chat_id"]
        meta     = burst_lookup.get(burst_id, {})
        participants = str(meta.get("participants") or "Unknown")

        cached = _load_cache(burst_id)
        if cached is not None:
            p2_in_tok  += cached.get("input_tokens", 0)
            p2_out_tok += cached.get("output_tokens", 0)
            cached_hits += 1
            cached_results.append({
                "idx": idx,
                "burst_id": burst_id,
                "chat_id": chat_id,
                "primary_topic": cached["primary_topic"],
                "secondary_topic": cached["secondary_topic"],
            })
        else:
            prompt = _CLASSIFY_PROMPT.format(
                taxonomy_block=taxonomy_block,
                participant_names=participants,
                summary_text=summary_row["summary"],
            )
            uncached_items.append({
                "idx": idx,
                "burst_id": burst_id,
                "chat_id": chat_id,
                "messages": [{"role": "user", "content": prompt}],
            })

    print(f"  {cached_hits:,} cached, {len(uncached_items):,} to classify\n")

    # ── Phase 2 (async): classify uncached items ────────────────────────────────

    async def _classify_one(client, item: dict) -> dict:
        """Classify a single burst via the async API."""
        burst_id = item["burst_id"]
        result   = None
        in_tok_r = out_tok_r = 0

        for attempt in range(2):
            try:
                content, usage = await async_call_api(
                    client,
                    model=CLASSIFY_MODEL,
                    messages=item["messages"],
                    temperature=CLASSIFY_TEMPERATURE,
                    response_format={"type": "json_object"},
                    max_tokens=CLASSIFY_MAX_TOKENS,
                )
                in_tok_r  = usage.prompt_tokens
                out_tok_r = usage.completion_tokens
                parsed    = _validate_classification(json.loads(content), valid_ids)
                if parsed["primary_topic"] is not None:
                    result = parsed
                    break
                tqdm.write(
                    f"    [warn] invalid topics for {burst_id[:40]} "
                    f"(attempt {attempt + 1}) -- retrying..."
                )
            except Exception as e:
                tqdm.write(f"    [error] burst {burst_id[:40]}: {e}")
                break

        is_error = False
        if result is None:
            tqdm.write(f"    [error] giving up on {burst_id[:40]}")
            result = {"primary_topic": None, "secondary_topic": None}
            is_error = True

        _save_cache(burst_id, {
            **result,
            "input_tokens":  in_tok_r,
            "output_tokens": out_tok_r,
        })

        return {
            "idx":             item["idx"],
            "burst_id":        item["burst_id"],
            "chat_id":         item["chat_id"],
            "primary_topic":   result["primary_topic"],
            "secondary_topic": result["secondary_topic"],
            "input_tokens":    in_tok_r,
            "output_tokens":   out_tok_r,
            "is_error":        is_error,
        }

    if uncached_items:
        async_results = asyncio.run(process_batch(
            items=uncached_items,
            process_fn=_classify_one,
            api_key=api_key,
            concurrency=args.concurrency,
            desc="Classifying",
        ))
    else:
        async_results = []

    # ── Phase 3 (sync): merge results, build output rows ────────────────────────
    classified = cached_hits
    errors = 0

    for r in async_results:
        p2_in_tok  += r["input_tokens"]
        p2_out_tok += r["output_tokens"]
        if r["is_error"]:
            errors += 1
        else:
            classified += 1

    # Combine cached + async results, sort by original index to preserve order
    all_results = cached_results + [
        {
            "idx":             r["idx"],
            "burst_id":        r["burst_id"],
            "chat_id":         r["chat_id"],
            "primary_topic":   r["primary_topic"],
            "secondary_topic": r["secondary_topic"],
        }
        for r in async_results
    ]
    all_results.sort(key=lambda x: x["idx"])

    for r in all_results:
        primary_id   = r["primary_topic"]
        secondary_id = r["secondary_topic"]
        rows.append({
            "burst_id":              r["burst_id"],
            "chat_id":               r["chat_id"],
            "primary_topic_id":      primary_id,
            "primary_topic_label":   id_to_label.get(primary_id) if primary_id else None,
            "secondary_topic_id":    secondary_id,
            "secondary_topic_label": id_to_label.get(secondary_id) if secondary_id else None,
        })

    # ── Build burst_topics_llm.parquet ───────────────────────────────────────────
    print(f"\nBuilding output tables...")

    bt_df = pd.DataFrame(rows)
    bt_df["primary_topic_id"]   = pd.array(bt_df["primary_topic_id"],   dtype=pd.Int32Dtype())
    bt_df["secondary_topic_id"] = pd.array(bt_df["secondary_topic_id"], dtype=pd.Int32Dtype())

    bt_df.to_parquet(OUTPUT_DIR / "burst_topics_llm.parquet", index=False)
    print(f"  burst_topics_llm.parquet  ({len(bt_df):,} rows)")

    # ── topic_info_llm.json ──────────────────────────────────────────────────────
    primary_counts   = bt_df["primary_topic_id"].value_counts().to_dict()
    secondary_counts = bt_df["secondary_topic_id"].dropna().value_counts().to_dict()

    topics_out = []
    for t in taxonomy:
        tid    = int(t["id"])
        p_cnt  = int(primary_counts.get(tid, 0))
        s_cnt  = int(secondary_counts.get(tid, 0))
        topics_out.append({
            "id":              tid,
            "label":           t["label"],
            "description":     t["description"],
            "keywords":        t["keywords"],
            "burst_count":     p_cnt + s_cnt,
            "primary_count":   p_cnt,
            "secondary_count": s_cnt,
        })
    topics_out.sort(key=lambda x: x["burst_count"], reverse=True)

    dual_count = int(bt_df["secondary_topic_id"].notna().sum())
    total_classified = int(bt_df["primary_topic_id"].notna().sum())

    topic_info = {
        "topics":             topics_out,
        "total_classified":   total_classified,
        "single_topic_pct":   round((total_classified - dual_count) / total_classified * 100, 1) if total_classified else 0,
        "dual_topic_pct":     round(dual_count / total_classified * 100, 1) if total_classified else 0,
    }
    (OUTPUT_DIR / "topic_info_llm.json").write_text(
        json.dumps(topic_info, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"  topic_info_llm.json        ({len(topics_out)} topics)")

    # ── topics_llm_report.json ───────────────────────────────────────────────────
    p1_cost_total = _p1_cost(p1_in_tok, p1_out_tok)
    p2_cost_total = _p2_cost(p2_in_tok, p2_out_tok)

    topic_dist = {t["label"]: t["burst_count"] for t in topics_out}

    report = {
        "generated_at":         datetime.now(timezone.utc).isoformat(),
        "taxonomy_model":       TAXONOMY_MODEL,
        "classification_model": CLASSIFY_MODEL,
        "taxonomy_size":        len(taxonomy),
        "sample_size":          min(TAXONOMY_SAMPLE_SIZE, len(eligible)),
        "total_eligible":       total,
        "total_classified":     classified,
        "errors":               errors,
        "cached_hits":          cached_hits,
        "topic_distribution":   topic_dist,
        "pass1_input_tokens":   p1_in_tok,
        "pass1_output_tokens":  p1_out_tok,
        "pass1_cost_usd":       round(p1_cost_total, 4),
        "pass2_input_tokens":   p2_in_tok,
        "pass2_output_tokens":  p2_out_tok,
        "pass2_cost_usd":       round(p2_cost_total, 4),
        "total_cost_usd":       round(p1_cost_total + p2_cost_total, 4),
    }
    (OUTPUT_DIR / "topics_llm_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"  topics_llm_report.json")
    print()

    # ── Console summary ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("  08b_topics_llm -- complete")
    print("=" * 60)
    print(f"  Eligible bursts:     {total:>8,}")
    print(f"  Classified:          {classified:>8,}  ({cached_hits:,} from cache)")
    print(f"  Errors:              {errors:>8,}")
    print(f"  Taxonomy categories: {len(taxonomy):>8,}")
    print(f"  Dual-topic bursts:   {dual_count:>8,}  ({topic_info['dual_topic_pct']:.1f}%)")
    print()
    print(f"  Pass 1 cost:         ${p1_cost_total:.4f}")
    print(f"  Pass 2 cost:         ${p2_cost_total:.4f}")
    print(f"  Total cost:          ${p1_cost_total + p2_cost_total:.4f}")
    print()

    print("  Top topics by burst count:")
    for t in topics_out[:10]:
        pct = t["burst_count"] / total_classified * 100 if total_classified else 0
        print(f"    {t['label']:<35} {t['burst_count']:>5,}  ({pct:.1f}%)")
    print()

    cache_count = len(list(CACHE_DIR.glob("*.json")))
    print(f"Outputs: {OUTPUT_DIR}/")
    print(f"  burst_topics_llm.parquet  ({len(bt_df):,} rows)")
    print(f"  topic_info_llm.json       ({len(topics_out)} topics)")
    print(f"  taxonomy.json             ({len(taxonomy)} categories)")
    print(f"  llm_cache/                ({cache_count:,} cached bursts)")
    print(f"  topics_llm_report.json")
    print()
    print("Next: python message_miner/12_wrapped.py")


if __name__ == "__main__":
    main()
