"""
26_notification_analysis.py — Automated Email Analysis

Analyzes non-human mail (automated notifications, system alerts, newsletters,
calendar invites, and other machine-generated messages).

Reads:
  outputs/23_enrich_mail/messages_mail_enriched.parquet
  outputs/22_edge_cases_mail/messages_mail_flagged.parquet
  outputs/20_extract_mail/extraction_report.json

Writes:
  outputs/26_notification_analysis/
    notification_analysis.json
    notification_report.json

Run from project root:
    python mail-miner/26_notification_analysis.py [--force]
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

ENRICHED_PATH = Path("outputs/23_enrich_mail/messages_mail_enriched.parquet")
FLAGGED_PATH = Path("outputs/22_edge_cases_mail/messages_mail_flagged.parquet")
EXTRACTION_REPORT_PATH = Path("outputs/20_extract_mail/extraction_report.json")
OUTPUT_DIR = Path("outputs/26_notification_analysis")

FLAG_COLS = [
    "is_notification",
    "is_newsletter",
    "is_calendar",
    "is_auto_reply",
    "is_read_receipt",
    "is_distribution_list",
    "is_encrypted",
    "is_draft",
]

SENDER_CATEGORIES = [
    (re.compile(r"jira|atlassian", re.IGNORECASE), "Ticketing"),
    (re.compile(r"servicenow|snow", re.IGNORECASE), "Ticketing"),
    (re.compile(r"github|gitlab|bitbucket", re.IGNORECASE), "Code/DevOps"),
    (re.compile(r"azure|microsoft|teams", re.IGNORECASE), "Microsoft/Cloud"),
    (re.compile(r"okta|duo|ping", re.IGNORECASE), "Identity/SSO"),
    (re.compile(r"slack|zoom|webex", re.IGNORECASE), "Collaboration"),
    (re.compile(r"salesforce|hubspot", re.IGNORECASE), "CRM"),
    (re.compile(r"noreply|no-reply|donotreply", re.IGNORECASE), "Generic No-Reply"),
    (re.compile(r"calendar|invite", re.IGNORECASE), "Calendar"),
    (re.compile(r"newsletter|digest|updates", re.IGNORECASE), "Newsletter"),
]


def _categorize_sender(email: str, name: str) -> str:
    combined = f"{email} {name}"
    for pattern, category in SENDER_CATEGORIES:
        if pattern.search(combined):
            return category
    return "Other"


def _normalize_subject(subject: str) -> str:
    if not isinstance(subject, str):
        return ""
    s = subject[:50]
    # Replace UUIDs
    s = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "{id}", s, flags=re.IGNORECASE)
    # Replace IPs
    s = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "{id}", s)
    # Replace long hex strings (8+ chars)
    s = re.sub(r"\b[0-9a-f]{8,}\b", "{id}", s, flags=re.IGNORECASE)
    # Replace numbers with #
    s = re.sub(r"\d+", "#", s)
    return s.strip()


def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


# ── Analysis 1: Flag Breakdown ────────────────────────────────────────────────

def flag_breakdown(df: pd.DataFrame) -> dict:
    print("  Computing flag breakdown...")
    total = len(df)
    result = {}
    for col in FLAG_COLS:
        if col in df.columns:
            count = int(df[col].sum())
            result[col] = {"count": count, "pct": round(count / total * 100, 1) if total else 0}
    return result


# ── Analysis 2: Top Notification Senders ─────────────────────────────────────

def top_senders(df: pd.DataFrame) -> list[dict]:
    print("  Computing top senders...")
    total = len(df)

    df = df.copy()
    df["sender_email"] = df["sender_email"].fillna("").str.lower().str.strip()
    df["sender_name"] = df["sender_name"].fillna("")
    df["received_datetime"] = pd.to_datetime(df["received_datetime"], utc=True, errors="coerce")

    grouped = df.groupby("sender_email")
    results = []
    for email, group in grouped:
        name = group["sender_name"].mode().iloc[0] if not group["sender_name"].mode().empty else ""
        count = len(group)
        first_seen = group["received_datetime"].min()
        last_seen = group["received_datetime"].max()

        if pd.notna(first_seen) and pd.notna(last_seen):
            weeks = max((last_seen - first_seen).days / 7, 1)
            avg_per_week = round(count / weeks, 1)
        else:
            avg_per_week = None

        results.append({
            "sender_email": email,
            "sender_name": name,
            "count": count,
            "pct_of_notifications": round(count / total * 100, 2) if total else 0,
            "first_seen": first_seen.strftime("%Y-%m-%d") if pd.notna(first_seen) else None,
            "last_seen": last_seen.strftime("%Y-%m-%d") if pd.notna(last_seen) else None,
            "avg_per_week": avg_per_week,
            "category": _categorize_sender(email, name),
        })

    results.sort(key=lambda x: x["count"], reverse=True)
    return results[:30]


# ── Analysis 3: By Category ───────────────────────────────────────────────────

def by_category(df: pd.DataFrame) -> list[dict]:
    print("  Computing by-category breakdown...")
    total = len(df)
    df = df.copy()
    df["sender_email"] = df["sender_email"].fillna("").str.lower().str.strip()
    df["sender_name"] = df["sender_name"].fillna("")
    df["category"] = df.apply(lambda r: _categorize_sender(r["sender_email"], r["sender_name"]), axis=1)

    agg = df.groupby("category").agg(
        email_count=("message_id", "count"),
        sender_count=("sender_email", "nunique"),
    ).reset_index()

    agg = agg.sort_values("email_count", ascending=False)
    result = []
    for _, row in agg.iterrows():
        result.append({
            "category": row["category"],
            "sender_count": int(row["sender_count"]),
            "email_count": int(row["email_count"]),
            "pct": round(row["email_count"] / total * 100, 1) if total else 0,
        })
    return result


# ── Analysis 4: Volume Over Time ──────────────────────────────────────────────

def volume_over_time(df: pd.DataFrame) -> dict:
    print("  Computing volume over time...")
    df = df.copy()
    df["received_datetime"] = pd.to_datetime(df["received_datetime"], utc=True, errors="coerce")
    df["sender_email"] = df["sender_email"].fillna("").str.lower().str.strip()
    df["sender_name"] = df["sender_name"].fillna("")
    df["category"] = df.apply(lambda r: _categorize_sender(r["sender_email"], r["sender_name"]), axis=1)
    df = df.dropna(subset=["received_datetime"])
    df["month"] = df["received_datetime"].dt.to_period("M").astype(str)
    df["date"] = df["received_datetime"].dt.strftime("%Y-%m-%d")

    # Top 5 categories
    top5 = (
        df.groupby("category").size()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    # Monthly by category
    months_sorted = sorted(df["month"].unique().tolist())
    series = []
    for cat in top5:
        cat_df = df[df["category"] == cat]
        monthly = cat_df.groupby("month").size()
        values = [int(monthly.get(m, 0)) for m in months_sorted]
        series.append({"category": cat, "values": values})

    monthly_by_category = {"months": months_sorted, "series": series}

    # Daily overall
    daily = df.groupby("date").size().reset_index(name="count")
    daily_volume = [
        {"date": row["date"], "count": int(row["count"])}
        for _, row in daily.sort_values("date").iterrows()
    ]

    # Spike detection
    counts = daily["count"].values.astype(float)
    mean_daily = float(np.mean(counts))
    std_daily = float(np.std(counts))
    threshold = mean_daily + 2 * std_daily

    spikes = []
    for _, row in daily[daily["count"] > threshold].sort_values("count", ascending=False).iterrows():
        day = row["date"]
        day_df = df[df["date"] == day]
        top_sender = day_df.groupby("sender_email").size().idxmax() if len(day_df) else ""
        spikes.append({
            "date": day,
            "count": int(row["count"]),
            "avg_daily": round(mean_daily, 1),
            "std_dev_above": round((row["count"] - mean_daily) / std_daily, 1) if std_daily else 0,
            "top_sender": top_sender,
        })

    return {
        "monthly_by_category": monthly_by_category,
        "daily_volume": daily_volume,
        "spikes": spikes,
        "mean_daily": round(mean_daily, 1),
        "std_daily": round(std_daily, 1),
    }


# ── Analysis 5: Subject Line Analysis ────────────────────────────────────────

def subject_analysis(df: pd.DataFrame) -> dict:
    print("  Computing subject line analysis...")
    df = df.copy()
    df["sender_email"] = df["sender_email"].fillna("").str.lower().str.strip()
    df["subject"] = df["subject"].fillna("")
    df["normalized"] = df["subject"].apply(_normalize_subject)

    # Subject patterns
    pattern_df = df.groupby("normalized").agg(
        count=("message_id", "count"),
        sender=("sender_email", lambda x: x.mode().iloc[0] if not x.mode().empty else ""),
    ).reset_index().sort_values("count", ascending=False).head(25)

    subject_patterns = [
        {"pattern": row["normalized"], "count": int(row["count"]), "sender": row["sender"]}
        for _, row in pattern_df.iterrows()
    ]

    # Importance distribution
    total = len(df)
    imp_dist = {}
    for level in ["low", "normal", "high"]:
        count = int((df["importance"].str.lower() == level).sum()) if "importance" in df.columns else 0
        imp_dist[level] = {"count": count, "pct": round(count / total * 100, 1) if total else 0}

    # High importance senders
    if "importance" in df.columns:
        high_df = df[df["importance"].str.lower() == "high"]
        high_senders = (
            high_df.groupby("sender_email").size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(10)
        )
        high_importance_senders = [
            {"sender_email": row["sender_email"], "count": int(row["count"])}
            for _, row in high_senders.iterrows()
        ]
    else:
        high_importance_senders = []

    return {
        "subject_patterns": subject_patterns,
        "importance_distribution": imp_dist,
        "high_importance_senders": high_importance_senders,
    }


# ── Analysis 6: Read Rate ─────────────────────────────────────────────────────

def read_rate(df: pd.DataFrame) -> dict:
    print("  Computing read rate...")
    df = df.copy()
    df["sender_email"] = df["sender_email"].fillna("").str.lower().str.strip()
    df["sender_name"] = df["sender_name"].fillna("")
    df["category"] = df.apply(lambda r: _categorize_sender(r["sender_email"], r["sender_name"]), axis=1)

    readable = df[df["is_read"].notna()] if "is_read" in df.columns else pd.DataFrame()

    if readable.empty:
        return {"overall_read_pct": None, "by_category": [], "by_importance": {}}

    overall_pct = round(readable["is_read"].mean() * 100, 1)

    by_cat = readable.groupby("category")["is_read"].mean().reset_index()
    by_category_list = [
        {"category": row["category"], "read_pct": round(row["is_read"] * 100, 1)}
        for _, row in by_cat.sort_values("is_read", ascending=False).iterrows()
    ]

    by_imp = {}
    if "importance" in readable.columns:
        for level in ["low", "normal", "high"]:
            subset = readable[readable["importance"].str.lower() == level]
            by_imp[level] = {"read_pct": round(subset["is_read"].mean() * 100, 1) if len(subset) else None}

    return {
        "overall_read_pct": overall_pct,
        "by_category": by_category_list,
        "by_importance": by_imp,
    }


# ── Analysis 7: Hourly Distribution ──────────────────────────────────────────

def hourly_distribution(df: pd.DataFrame) -> list[int]:
    print("  Computing hourly distribution...")
    df = df.copy()
    df["received_datetime"] = pd.to_datetime(df["received_datetime"], utc=True, errors="coerce")
    df = df.dropna(subset=["received_datetime"])
    hourly = df.groupby(df["received_datetime"].dt.hour).size()
    return [int(hourly.get(h, 0)) for h in range(24)]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="26_notification_analysis — Automated Email Analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = parser.parse_args()

    output_analysis = OUTPUT_DIR / "notification_analysis.json"
    output_report = OUTPUT_DIR / "notification_report.json"

    if not args.force and output_analysis.exists() and output_report.exists():
        print("Outputs already exist. Use --force to overwrite.")
        return

    # Load data
    print("Loading data...")
    enriched = pd.read_parquet(ENRICHED_PATH)
    flagged = pd.read_parquet(FLAGGED_PATH)

    # Merge: use enriched as base (already has all flag cols), join any flagged-only cols
    flag_only_cols = [c for c in flagged.columns if c not in enriched.columns]
    if flag_only_cols:
        df = enriched.merge(flagged[["message_id"] + flag_only_cols], on="message_id", how="left")
    else:
        df = enriched.copy()

    # Filter to non-human mail
    df = df[df["is_human_mail"] == False].copy()
    print(f"Non-human mail: {len(df):,} messages")

    # Load extraction report for metadata
    user_email = ""
    date_range_start = ""
    date_range_end = ""
    if EXTRACTION_REPORT_PATH.exists():
        with open(EXTRACTION_REPORT_PATH) as f:
            extraction = json.load(f)
        user_email = extraction.get("user_email", "")
        date_range_start = extraction.get("date_range", {}).get("start", "")
        date_range_end = extraction.get("date_range", {}).get("end", "")

    # Run analyses
    print("Running analyses...")
    a1 = flag_breakdown(df)
    a2 = top_senders(df)
    a3 = by_category(df)
    a4 = volume_over_time(df)
    a5 = subject_analysis(df)
    a6 = read_rate(df)
    a7 = hourly_distribution(df)

    analysis = {
        "flag_breakdown": a1,
        "top_senders": a2,
        "by_category": a3,
        "monthly_by_category": a4["monthly_by_category"],
        "daily_volume": a4["daily_volume"],
        "spikes": a4["spikes"],
        "subject_patterns": a5["subject_patterns"],
        "importance_distribution": a5["importance_distribution"],
        "high_importance_senders": a5["high_importance_senders"],
        "read_rate": a6,
        "hourly_distribution": a7,
    }

    # Build report highlights
    total = len(df)

    # Noisiest sender
    noisiest_sender_str = ""
    if a2:
        top = a2[0]
        noisiest_sender_str = f"{top['sender_email']} ({top['count']:,} emails)"

    # Noisiest category
    noisiest_category_str = ""
    if a3:
        top_cat = a3[0]
        noisiest_category_str = f"{top_cat['category']} ({top_cat['pct']}%)"

    # Avg notifications per day
    avg_per_day = a4["mean_daily"]

    # Spike info
    spike_days = len(a4["spikes"])
    worst_spike_str = ""
    if a4["spikes"]:
        worst = a4["spikes"][0]
        worst_spike_str = f"{worst['date']} — {worst['count']:,} notifications"

    # Read rate
    overall_read_rate_str = (
        f"{a6['overall_read_pct']}%" if a6["overall_read_pct"] is not None else "N/A"
    )

    # Least/most read category
    least_read_str = ""
    most_read_str = ""
    if a6["by_category"]:
        sorted_cats = sorted(a6["by_category"], key=lambda x: x["read_pct"] or 0)
        least_read_str = f"{sorted_cats[0]['category']} ({sorted_cats[0]['read_pct']}% read)"
        most_read_str = f"{sorted_cats[-1]['category']} ({sorted_cats[-1]['read_pct']}% read)"

    # High importance pct
    high_imp_pct_str = ""
    if a5["importance_distribution"].get("high"):
        high_imp_pct_str = f"{a5['importance_distribution']['high']['pct']}%"

    # Date range from data if not in extraction report
    if not date_range_start or not date_range_end:
        dt_col = pd.to_datetime(df["received_datetime"], utc=True, errors="coerce").dropna()
        if not dt_col.empty:
            date_range_start = dt_col.min().strftime("%Y-%m-%d")
            date_range_end = dt_col.max().strftime("%Y-%m-%d")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_non_human_mail": total,
        "date_range": f"{date_range_start} to {date_range_end}",
        "highlights": {
            "noisiest_sender": noisiest_sender_str,
            "noisiest_category": noisiest_category_str,
            "avg_notifications_per_day": avg_per_day,
            "spike_days": spike_days,
            "worst_spike": worst_spike_str,
            "overall_read_rate": overall_read_rate_str,
            "least_read_category": least_read_str,
            "most_read_category": most_read_str,
            "high_importance_pct": high_imp_pct_str,
        },
    }

    # Write outputs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_analysis, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    with open(output_report, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nWrote: {output_analysis}")
    print(f"Wrote: {output_report}")
    print(f"\nHighlights:")
    for k, v in report["highlights"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
