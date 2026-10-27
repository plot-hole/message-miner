"""run_all.py -- Run the full message-miner pipeline end to end."""

import argparse
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path

SCRIPTS = [
    {
        "path": "message_miner/01_connect.py",
        "name": "Connect & Authenticate",
        "description": "Signs you in to Microsoft Graph and retrieves your identity.",
        "estimate": "~30 seconds (requires browser sign-in)",
        "interactive": True,
        "output_check": "outputs/01_connect/connection_report.json",
    },
    {
        "path": "message_miner/02_extract.py",
        "name": "Extract Raw Data",
        "description": "Downloads your Teams chats and messages from the Graph API.",
        "estimate": "~2-10 minutes depending on message volume",
        "output_check": "outputs/02_extract/chats.parquet",
    },
    {
        "path": "message_miner/03_clean.py",
        "name": "Clean Messages",
        "description": "Normalizes text, removes system messages, standardizes timestamps.",
        "estimate": "~30 seconds",
        "output_check": "outputs/03_clean/messages_clean.parquet",
    },
    {
        "path": "message_miner/04_edge_cases.py",
        "name": "Flag Edge Cases",
        "description": "Identifies bots, cards, file shares, emoji-only, and deleted messages.",
        "estimate": "~30 seconds",
        "output_check": "outputs/04_edge_cases/messages_flagged.parquet",
    },
    {
        "path": "message_miner/05_enrich.py",
        "name": "Enrich Messages",
        "description": "Adds response times, word counts, and the is_self flag.",
        "estimate": "~1 minute",
        "output_check": "outputs/05_enrich/messages_enriched.parquet",
    },
    {
        "path": "message_miner/07_segment.py",
        "name": "Segment Conversations",
        "description": "Groups messages into conversation bursts using a 1-day time-gap threshold.",
        "estimate": "~30 seconds",
        "output_check": "outputs/07_segment/bursts.parquet",
        "parallel_group": "after_enrich",
    },
    {
        "path": "message_miner/09_emoji.py",
        "name": "Emoji Analysis",
        "description": "Extracts and aggregates emoji usage per message and per sender.",
        "estimate": "~1 minute",
        "output_check": "outputs/09_emoji/emoji_summary.parquet",
        "parallel_group": "after_enrich",
    },
    {
        "path": "message_miner/11_summarize.py",
        "name": "Summarize Conversations",
        "description": "Generates a plain-English summary of each conversation burst.",
        "estimate": "~10-20 minutes, ~$1.50",
        "output_check": "outputs/11_summarize/summaries.parquet",
    },
    {
        "path": "message_miner/11b_nlp_features.py",
        "name": "NLP Features",
        "description": "Computes vocabulary, readability, named entities, and question detection.",
        "estimate": "~5-15 minutes",
        "output_check": "outputs/11b_nlp_features/sender_profiles.parquet",
    },
    {
        "path": "message_miner/11c_classify.py",
        "name": "Classify Conversations",
        "description": "Classifies each conversation by function, depth, decisions, and action items.",
        "estimate": "~10-15 minutes, ~$1.50",
        "output_check": "outputs/11c_classify/classifications.parquet",
    },
    {
        "path": "message_miner/08b_topics_llm.py",
        "name": "Topic Classification (LLM)",
        "description": "Generates a topic taxonomy and classifies each conversation by topic.",
        "estimate": "~10-15 minutes, ~$0.25",
        "interactive": True,
        "output_check": "outputs/08b_topics_llm/burst_topics_llm.parquet",
    },
    {
        "path": "message_miner/12_wrapped.py",
        "name": "Generate Wrapped Report",
        "description": "Computes all metrics, generates a narrative, and renders your report.",
        "estimate": "~30 seconds",
        "output_check": "outputs/12_wrapped/wrapped.html",
    },
]

BORDER = "=" * 62


# ── Pre-flight checks ───────────────────────────────────────────────────────────

def preflight_checks() -> bool:
    """Run environment checks before starting the pipeline. Returns True if all pass."""
    print("  Pre-flight checks:")
    all_ok = True

    # 1. .env file and required keys
    import os
    env_path = Path(".env")
    required_keys = ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "OPENAI_API_KEY")
    if not env_path.exists():
        print("    [FAIL] .env file not found.")
        print("           Create .env with AZURE_TENANT_ID, AZURE_CLIENT_ID, OPENAI_API_KEY.")
        all_ok = False
    else:
        env_text = env_path.read_text(encoding="utf-8")
        missing = [k for k in required_keys if k not in env_text and k not in os.environ]
        if missing:
            print(f"    [FAIL] .env file is missing: {', '.join(missing)}")
            all_ok = False
        else:
            print("    [OK]   .env file found with required keys")

    # 2. spaCy model
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("    [OK]   spaCy en_core_web_sm model loaded")
    except Exception:
        print("    [FAIL] spaCy model not found.")
        print("           Run: python -m spacy download en_core_web_sm")
        all_ok = False

    # 4. Key Python packages
    required_packages = {
        "pandas": "pandas",
        "openai": "openai",
        "emoji": "emoji",
        "jinja2": "Jinja2",
    }
    missing_packages = []
    for import_name, pip_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        print(f"    [FAIL] Missing packages: {', '.join(missing_packages)}")
        print("           Run: pip install -r requirements.txt")
        all_ok = False
    else:
        print("    [OK]   All required Python packages installed")

    print()
    return all_ok


# ── Skip logic ──────────────────────────────────────────────────────────────────

def ask_skip(script_info: dict, step_num: int, total: int) -> bool:
    """Ask the user whether to skip a step whose outputs already exist. Default: skip."""
    print(f"  Existing output found for step {step_num}.")
    try:
        answer = input("  Skip this step? [Y/n]: ").strip().lower()
    except EOFError:
        answer = ""
    return answer not in ("n", "no")


# ── Cost estimate after extraction ─────────────────────────────────────────────

def _compute_estimates(msg_count: int, concurrency: int = 10) -> tuple[float, float, float, float]:
    """Return (low_cost, high_cost, low_hours, high_hours) for a given message count."""
    estimated_bursts = msg_count / 10

    cost_11  = estimated_bursts * 0.00015
    cost_11c = estimated_bursts * 0.00015
    cost_08b = estimated_bursts * 0.000025
    cost_12  = 0.01
    total_cost = cost_11 + cost_11c + cost_08b + cost_12

    # LLM steps benefit from concurrency — cap speedup at 8x to account for rate limits
    speedup = min(concurrency, 8)
    time_07  = 0.5                        # gap-only segmentation; near-constant time
    time_11  = estimated_bursts / 60 / speedup
    time_11c = estimated_bursts / 60 / speedup
    time_08b = estimated_bursts / 60 / speedup
    time_03  = 0.5
    time_04  = 0.5
    time_05  = 1.0
    time_09  = 1.0
    time_11b = msg_count / 10000
    time_12  = 0.5
    total_minutes = (time_07 + time_11 + time_11c + time_08b +
                     time_03 + time_04 + time_05 + time_09 +
                     time_11b + time_12)

    return (
        total_cost * 0.8,
        total_cost * 1.3,
        (total_minutes * 0.8) / 60,
        (total_minutes * 1.3) / 60,
    )


def show_cost_estimate(concurrency: int = 10) -> bool:
    """Read extracted data, show cost/time estimates, offer date filter. Returns True to continue."""
    import pandas as pd

    msgs_path  = Path("outputs/02_extract/messages.parquet")
    chats_path = Path("outputs/02_extract/chats.parquet")

    if not msgs_path.exists():
        return True

    try:
        msgs  = pd.read_parquet(msgs_path)
        chats = pd.read_parquet(chats_path) if chats_path.exists() else None
    except Exception:
        return True

    msg_count  = len(msgs)
    chat_count = len(chats) if chats is not None else "?"

    # Date range
    ts = pd.to_datetime(msgs["timestamp"], utc=True, errors="coerce").dropna()
    earliest = ts.min()
    latest   = ts.max()
    span_years = (latest - earliest).days / 365.25 if len(ts) > 0 else 0

    earliest_str = earliest.strftime("%Y-%m-%d") if len(ts) > 0 else "unknown"
    latest_str   = latest.strftime("%Y-%m-%d")   if len(ts) > 0 else "unknown"

    low_cost, high_cost, low_hours, high_hours = _compute_estimates(msg_count, concurrency)

    print(BORDER)
    print(f"  Extraction complete: {msg_count:,} messages across {chat_count} chats.")
    print(f"  Date range: {earliest_str} to {latest_str}")
    print()
    print(f"  Estimated LLM cost:       ~${low_cost:.2f} - ${high_cost:.2f}")
    print(f"  Estimated time remaining: ~{low_hours:.1f} - {high_hours:.1f} hours")
    print("  (LLM steps run concurrently — actual time depends on --concurrency setting)")
    print()

    # Date filter option — only offered if span is 2+ years
    if span_years >= 2:
        print("  [1] Continue with all messages")
        print("  [2] Limit to a custom start date")
        print("  [3] Abort")
        print()
        try:
            choice = input("  Choose [1/2/3]: ").strip()
        except EOFError:
            choice = "1"

        if choice == "3":
            print(BORDER)
            print()
            return False

        if choice == "2":
            # Prompt for start year
            while True:
                try:
                    year_str = input("  Start from year (e.g. 2021): ").strip()
                except EOFError:
                    year_str = ""
                try:
                    start_year = int(year_str)
                    if start_year < 2000 or start_year > latest.year:
                        print(f"  Please enter a year between 2000 and {latest.year}.")
                        continue
                    break
                except ValueError:
                    print("  Please enter a valid 4-digit year.")

            cutoff = pd.Timestamp(year=start_year, month=1, day=1, tz="UTC")
            filtered_msgs = msgs[pd.to_datetime(msgs["timestamp"], utc=True, errors="coerce") >= cutoff]
            removed = msg_count - len(filtered_msgs)

            # Filter chats to those that still have messages
            if chats is not None:
                active_chat_ids = filtered_msgs["chat_id"].unique()
                filtered_chats  = chats[chats["chat_id"].isin(active_chat_ids)]
            else:
                filtered_chats = None

            print()
            print(f"  Removed {removed:,} messages before {start_year} "
                  f"({removed / msg_count * 100:.1f}% of total).")
            print(f"  Keeping {len(filtered_msgs):,} messages across "
                  f"{len(filtered_chats) if filtered_chats is not None else '?'} chats.")

            # Overwrite parquet files with filtered data
            filtered_msgs.to_parquet(msgs_path, index=False)
            if filtered_chats is not None:
                filtered_chats.to_parquet(chats_path, index=False)
            print("  messages.parquet and chats.parquet updated.")

            # Recalculate estimates
            new_count = len(filtered_msgs)
            low_cost, high_cost, low_hours, high_hours = _compute_estimates(new_count, concurrency)
            print()
            print(f"  Revised estimate -- {new_count:,} messages:")
            print(f"  Estimated LLM cost:       ~${low_cost:.2f} - ${high_cost:.2f}")
            print(f"  Estimated time remaining: ~{low_hours:.1f} - {high_hours:.1f} hours")
            print()
            try:
                answer = input("  Continue? [Y/n]: ").strip().lower()
            except EOFError:
                answer = ""
            print(BORDER)
            print()
            return answer not in ("n", "no")

        # choice == "1" or anything else: fall through to continue
    else:
        # Span < 2 years: just Continue / Abort
        try:
            answer = input("  Continue? [Y/n]: ").strip().lower()
        except EOFError:
            answer = ""
        print(BORDER)
        print()
        return answer not in ("n", "no")

    print(BORDER)
    print()
    return True


# ── Run a single step ───────────────────────────────────────────────────────────

def _build_extra_args(script_info: dict, args) -> list[str] | None:
    """Build extra CLI args for a step based on pipeline-level flags."""
    extra = []
    path = script_info["path"]
    if args.fresh:
        extra.append("--force")
    if path == "message_miner/02_extract.py":
        if args.since:
            extra.extend(["--since", args.since])
        extra.extend(["--concurrency", str(args.concurrency)])
    if path in (
        "message_miner/11_summarize.py",
        "message_miner/11c_classify.py",
        "message_miner/08b_topics_llm.py",
    ):
        extra.extend(["--concurrency", str(args.concurrency)])
    if path == "message_miner/07_segment.py":
        extra.extend(["--threshold", "86400"])
    if path == "message_miner/12_wrapped.py" and args.since:
        extra.extend(["--since", args.since])
    return extra or None


def run_parallel_steps(
    scripts: list[dict], step_nums: list[int], total: int,
    args,
) -> bool:
    """Run multiple independent steps simultaneously. Returns True if all succeed."""
    import subprocess

    names = [s["name"] for s in scripts]
    print(BORDER)
    print(f"  Steps {', '.join(str(n) for n in step_nums)} of {total}  (running in parallel)")
    for s in scripts:
        print(f"    • {s['name']}: {s['description']}")
    print(BORDER)
    print()

    start = time.time()
    processes = []
    for script_info in scripts:
        cmd = [sys.executable, script_info["path"]]
        extra = _build_extra_args(script_info, args)
        if extra:
            cmd.extend(extra)
        p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        processes.append((p, script_info["name"]))

    # Wait for all to finish
    all_ok = True
    for p, name in processes:
        p.wait()
        if p.returncode != 0:
            print(f"  [FAIL] {name} failed (exit code {p.returncode})")
            all_ok = False

    elapsed = int(time.time() - start)
    elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"
    print()
    if all_ok:
        print(f"  [OK] Parallel group completed in {elapsed_str}: {', '.join(names)}")
    else:
        try:
            answer = input("  Some parallel steps failed. Continue? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in ("y", "yes"):
            return False
    print()
    return True


def run_step(script_info: dict, step_num: int, total: int,
             extra_args: list[str] | None = None) -> bool:
    """Run one pipeline step. Returns True on success."""
    import subprocess

    name        = script_info["name"]
    description = script_info["description"]
    estimate    = script_info["estimate"]
    interactive = script_info.get("interactive", False)

    print(BORDER)
    print(f"  Step {step_num} of {total}: {name}")
    print(f"  {description}")
    print(f"  Estimated time: {estimate}")
    if interactive:
        print()
        print("  ** This step requires your input -- follow the prompts below **")
    print(BORDER)
    print()

    start = time.time()

    cmd = [sys.executable, script_info["path"]]
    if extra_args:
        cmd.extend(extra_args)

    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    process.wait()

    elapsed = int(time.time() - start)
    elapsed_str = f"{elapsed // 60}m {elapsed % 60}s" if elapsed >= 60 else f"{elapsed}s"

    print()
    if process.returncode == 0:
        print(f"  [OK] {name} completed in {elapsed_str}")
        print()
        return True
    else:
        print(f"  [FAIL] {name} failed (exit code {process.returncode}) after {elapsed_str}")
        try:
            answer = input("  Continue to next step? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        print()
        return answer in ("y", "yes")


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full message-miner pipeline.")
    parser.add_argument("--fresh", action="store_true",
                        help="Re-run all steps even if outputs already exist.")
    parser.add_argument("--since", type=str, default=None,
                        help="Only extract messages since this date (YYYY-MM-DD). "
                             "Dramatically reduces processing time for large histories.")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent LLM API calls for async steps (default: 10)")
    args = parser.parse_args()

    total_steps = len(SCRIPTS)
    pipeline_start = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print()
    print(BORDER)
    print("  MESSAGE MINER -- Full Pipeline")
    print(f"  Started at {now}")
    print(f"  {total_steps} steps to your Wrapped report")
    print(BORDER)
    print()
    print("  NOTE: This pipeline will generate a report containing personal")
    print("  messaging data including names, message statistics, conversation")
    print("  summaries, and quoted messages. Review before sharing.")
    print()
    if args.since:
        print(f"  Date filter: only messages since {args.since}")
        print()
    print("  Estimated total time: ~30-60 minutes (with default concurrency)")
    print("  Estimated total LLM cost: ~$3-5")
    print()
    print(BORDER)
    print()

    # Pre-flight checks
    if not preflight_checks():
        print("  Aborting. Fix the issues above and re-run.")
        sys.exit(1)

    # Run each step (with parallel group detection)
    i = 0
    while i < len(SCRIPTS):
        script_info = SCRIPTS[i]
        step_num = i + 1

        # ── Detect parallel groups ───────────────────────────────────────────
        group = script_info.get("parallel_group")
        if group:
            # Collect all consecutive steps in this parallel group
            group_scripts = []
            group_step_nums = []
            j = i
            while j < len(SCRIPTS) and SCRIPTS[j].get("parallel_group") == group:
                group_scripts.append(SCRIPTS[j])
                group_step_nums.append(j + 1)
                j += 1

            # Skip logic: remove steps whose output already exists
            if not args.fresh:
                to_run = []
                to_run_nums = []
                for s, n in zip(group_scripts, group_step_nums):
                    oc = s.get("output_check")
                    if oc and Path(oc).exists():
                        if ask_skip(s, n, total_steps):
                            print(f"  Skipping step {n}: {s['name']}")
                            print()
                            continue
                    to_run.append(s)
                    to_run_nums.append(n)
            else:
                to_run = group_scripts
                to_run_nums = group_step_nums

            if len(to_run) > 1:
                success = run_parallel_steps(to_run, to_run_nums, total_steps, args)
            elif len(to_run) == 1:
                extra = _build_extra_args(to_run[0], args)
                success = run_step(to_run[0], to_run_nums[0], total_steps,
                                   extra_args=extra)
            else:
                success = True  # all skipped

            if not success:
                print("  Aborting pipeline.")
                sys.exit(1)

            i = j
            continue

        # ── Normal sequential step ───────────────────────────────────────────

        # Skip logic
        if not args.fresh:
            output_check = script_info.get("output_check")
            if output_check and Path(output_check).exists():
                if ask_skip(script_info, step_num, total_steps):
                    print(f"  Skipping step {step_num}: {script_info['name']}")
                    print()
                    i += 1
                    continue

        extra_args = _build_extra_args(script_info, args)

        # Run the step
        success = run_step(script_info, step_num, total_steps,
                           extra_args=extra_args)

        if not success:
            print("  Aborting pipeline.")
            sys.exit(1)

        # After extraction (step 2), show cost estimate
        if script_info["path"] == "message_miner/02_extract.py":
            if not show_cost_estimate(concurrency=args.concurrency):
                print("  Aborting pipeline.")
                sys.exit(0)

        i += 1

    # Closing banner
    total_elapsed = int(time.time() - pipeline_start)
    hours   = total_elapsed // 3600
    minutes = (total_elapsed % 3600) // 60
    seconds = total_elapsed % 60
    if hours > 0:
        elapsed_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        elapsed_str = f"{minutes}m {seconds}s"
    else:
        elapsed_str = f"{seconds}s"

    print(BORDER)
    print("  MESSAGE MINER -- Pipeline Complete")
    print(f"  Total time: {elapsed_str}")
    print()
    print("  Your report is ready:")
    print("    outputs/12_wrapped/wrapped.html")
    print()

    report_path = Path("outputs/12_wrapped/wrapped.html").resolve()
    if report_path.exists():
        webbrowser.open(report_path.as_uri())
        print("  Opening your report in a browser...")
    else:
        print("  Open outputs/12_wrapped/wrapped.html in a browser to see your report.")

    print(BORDER)
    print()


if __name__ == "__main__":
    main()
