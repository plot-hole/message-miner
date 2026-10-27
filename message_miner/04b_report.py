"""
04b_report.py — Data Inventory Dashboard

Reads messages_flagged.parquet and chats.parquet and produces six PNG figures
plus a machine-readable inventory_report.json.

Figures:
  volume_timeseries.png       daily message volume + 7-day rolling average
  temporal_heatmap.png        day-of-week x hour-of-day activity heatmap
  message_length_dist.png     text length distribution (chars), clean messages only
  sender_dist.png             top 25 senders by message count (bots excluded)
  chat_type_breakdown.png     message and chat counts by chat type
  member_count_dist.png       member count distribution across chats

Outputs:
  outputs/04b_report/inventory_report.json
  outputs/04b_report/figures/*.png

Run from project root:
    python message_miner/04b_report.py [--force]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/04_edge_cases/messages_flagged.parquet")
CHATS_PATH = Path("outputs/02_extract/chats.parquet")
OUTPUT_DIR = Path("outputs/04b_report")
FIGURES_DIR = OUTPUT_DIR / "figures"

DPI = 150
FIGSIZE = (12, 6)
COLOR_PRIMARY = "#2E86AB"
COLOR_ACCENT = "#E84855"


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


# ── 1. Volume time series ─────────────────────────────────────────────────────

def fig_volume_timeseries(df: pd.DataFrame) -> plt.Figure:
    daily = (
        df.dropna(subset=["timestamp"])
        .assign(date=df["timestamp"].dt.date)
        .groupby("date")
        .size()
        .reset_index(name="messages")
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")
    daily["rolling_7d"] = daily["messages"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(daily["date"], daily["messages"], color=COLOR_PRIMARY, alpha=0.5, label="Daily messages", width=1.0)
    ax.plot(daily["date"], daily["rolling_7d"], color=COLOR_ACCENT, linewidth=1.8, label="7-day average")
    ax.set_title("Daily Message Volume", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Messages")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    fig.tight_layout()
    return fig


# ── 2. Temporal heatmap ───────────────────────────────────────────────────────

def fig_temporal_heatmap(df: pd.DataFrame) -> plt.Figure:
    heat = (
        df.dropna(subset=["timestamp"])
        .assign(hour=df["timestamp"].dt.hour, dow=df["timestamp"].dt.dayofweek)
        .groupby(["dow", "hour"])
        .size()
        .reset_index(name="count")
    )
    pivot = (
        heat.pivot(index="dow", columns="hour", values="count")
        .reindex(index=range(7), columns=range(24), fill_value=0)
        .fillna(0)
    )

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Messages")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(7))
    ax.set_yticklabels(day_labels)
    ax.set_title("Message Activity: Day of Week x Hour of Day (UTC)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Hour (UTC)")
    fig.tight_layout()
    return fig


# ── 3. Message length distribution ───────────────────────────────────────────

def fig_message_length(df: pd.DataFrame) -> plt.Figure:
    clean = df[~df["is_empty"] & ~df["is_deleted"] & ~df["is_bot_message"]]
    lengths = clean["text"].dropna().str.len()
    p99 = int(lengths.quantile(0.99))
    lengths_capped = lengths[lengths <= p99]
    median = lengths.median()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(lengths_capped, bins=60, color=COLOR_PRIMARY, alpha=0.8, edgecolor="white")
    ax.axvline(median, color=COLOR_ACCENT, linewidth=2, linestyle="--", label=f"Median = {median:.0f}")
    ax.set_title(f"Message Length Distribution (chars, capped at p99={p99:,})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Characters")
    ax.set_ylabel("Messages")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()
    fig.tight_layout()
    return fig


# ── 4. Sender distribution ────────────────────────────────────────────────────

def fig_sender_dist(df: pd.DataFrame) -> plt.Figure:
    human = df[~df["is_bot_message"] & ~df["is_deleted"]]
    top = (
        human.groupby("sender_name", dropna=True)
        .size()
        .nlargest(25)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.35)))
    bars = ax.barh(top.index, top.values, color=COLOR_PRIMARY, alpha=0.85)
    for bar, val in zip(bars, top.values):
        ax.text(bar.get_width() + max(top.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_title("Top 25 Senders (bots excluded)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Messages")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(right=max(top.values) * 1.15)
    fig.tight_layout()
    return fig


# ── 5. Chat type breakdown ────────────────────────────────────────────────────

def fig_chat_type_breakdown(df: pd.DataFrame, chats_df: pd.DataFrame) -> plt.Figure:
    msg_by_type = (
        df.merge(chats_df[["chat_id", "chat_type"]], on="chat_id", how="left")
        .groupby("chat_type", dropna=False, observed=True)
        .size()
    )
    chat_counts = chats_df.groupby("chat_type", dropna=False, observed=True).size()

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    for ax, data, title in [
        (axes[0], msg_by_type, "Messages by Chat Type"),
        (axes[1], chat_counts, "Chats by Chat Type"),
    ]:
        labels = [str(l) for l in data.index]
        ax.pie(data.values, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=[COLOR_PRIMARY, COLOR_ACCENT, "#F4A261"])
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle("Chat Type Breakdown", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 6. Member count distribution ─────────────────────────────────────────────

def fig_member_count_dist(chats_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    chat_types = chats_df["chat_type"].dropna().unique()
    colors = [COLOR_PRIMARY, COLOR_ACCENT, "#F4A261"]
    for chat_type, color in zip(chat_types, colors):
        subset = chats_df[chats_df["chat_type"] == chat_type]["member_count"].dropna()
        ax.hist(subset, bins=20, alpha=0.6, label=str(chat_type), color=color, edgecolor="white")
    ax.set_title("Member Count Distribution by Chat Type", fontsize=14, fontweight="bold")
    ax.set_xlabel("Members in Chat")
    ax.set_ylabel("Chats")
    ax.legend()
    fig.tight_layout()
    return fig


# ── inventory_report.json ─────────────────────────────────────────────────────

def build_report(df: pd.DataFrame, chats_df: pd.DataFrame) -> dict:
    human = df[~df["is_bot_message"] & ~df["is_deleted"] & ~df["is_empty"]]
    text_lengths = human["text"].dropna().str.len()

    ts = df["timestamp"].dropna()
    date_range = {
        "first": ts.min().isoformat() if not ts.empty else None,
        "last": ts.max().isoformat() if not ts.empty else None,
    }

    top_senders = (
        human.groupby("sender_name", dropna=True)
        .size()
        .nlargest(10)
        .to_dict()
    )

    flag_cols = [
        "is_system_event", "is_bot_message", "is_adaptive_card",
        "is_file_share", "is_meeting_chat", "is_forwarded",
        "is_loop_component", "is_deleted", "is_emoji_only", "is_empty",
    ]
    flag_counts = {col: int(df[col].sum()) for col in flag_cols if col in df.columns}

    chat_type_counts = chats_df["chat_type"].value_counts().to_dict()
    msg_type_counts = df["message_type"].value_counts().to_dict()

    length_stats: dict = {}
    if not text_lengths.empty:
        desc = text_lengths.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
        length_stats = {k: round(float(v), 1) for k, v in desc.items()}

    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_messages": len(df),
        "total_chats": len(chats_df),
        "date_range": date_range,
        "by_chat_type": {str(k): int(v) for k, v in chat_type_counts.items()},
        "by_message_type": {str(k): int(v) for k, v in msg_type_counts.items()},
        "flag_counts": flag_counts,
        "top_10_senders": {str(k): int(v) for k, v in top_senders.items()},
        "text_length_stats": length_stats,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data inventory dashboard")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if (OUTPUT_DIR / "inventory_report.json").exists() and not args.force:
        answer = input("outputs/04b_report/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 04b_report ===\n")

    for path in (MESSAGES_PATH, CHATS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    df = pd.read_parquet(MESSAGES_PATH)
    chats_df = pd.read_parquet(CHATS_PATH)
    print(f"Loaded {len(df):,} messages, {len(chats_df):,} chats.\n")

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    print("Generating figures...")
    _save(fig_volume_timeseries(df), "volume_timeseries")
    _save(fig_temporal_heatmap(df), "temporal_heatmap")
    _save(fig_message_length(df), "message_length_dist")
    _save(fig_sender_dist(df), "sender_dist")
    _save(fig_chat_type_breakdown(df, chats_df), "chat_type_breakdown")
    _save(fig_member_count_dist(chats_df), "member_count_dist")

    print("\nBuilding inventory report...")
    report = build_report(df, chats_df)
    (OUTPUT_DIR / "inventory_report.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"\n--- Inventory summary ---")
    print(f"  Total messages:   {report['total_messages']:,}")
    print(f"  Total chats:      {report['total_chats']:,}")
    print(f"  Date range:       {report['date_range']['first'][:10]} to {report['date_range']['last'][:10]}")
    print(f"\n  By chat type:")
    for k, v in report["by_chat_type"].items():
        print(f"    {str(k):<22} {v:,} chats")
    print(f"\n  Top senders:")
    for name, count in report["top_10_senders"].items():
        print(f"    {str(name):<30} {count:,}")

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  inventory_report.json")
    print("  figures/volume_timeseries.png")
    print("  figures/temporal_heatmap.png")
    print("  figures/message_length_dist.png")
    print("  figures/sender_dist.png")
    print("  figures/chat_type_breakdown.png")
    print("  figures/member_count_dist.png")
    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
