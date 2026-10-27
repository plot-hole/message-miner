"""
09_emoji.py — Emoji Analysis

Extracts emoji usage from message text and reactions from raw API JSON.
Independent of 07/08 — reads directly from messages_enriched.parquet.

Three sources:
  1. Inline emoji in message text  (emoji library)
  2. is_emoji_only flag             (already in 04_edge_cases output)
  3. Reactions                      (raw JSON in outputs/02_extract/raw/)

Reads:
  outputs/05_enrich/messages_enriched.parquet
  outputs/02_extract/chats.parquet
  outputs/02_extract/raw/           raw API responses with reaction data

Writes:
  outputs/09_emoji/emoji_messages.parquet      per-message emoji extraction
  outputs/09_emoji/emoji_summary.parquet       per-emoji aggregate stats
  outputs/09_emoji/reaction_summary.parquet    per-reaction-type aggregate stats
  outputs/09_emoji/figures/                    visualizations
  outputs/09_emoji/emoji_report.json           run summary

Run from project root:
    python message_miner/09_emoji.py [--force]
"""

import argparse
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).parent.parent))

MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
CHATS_PATH    = Path("outputs/02_extract/chats.parquet")
RAW_DIR       = Path("outputs/02_extract/raw")
OUTPUT_DIR    = Path("outputs/09_emoji")
FIGURES_DIR   = OUTPUT_DIR / "figures"


# ── Reaction parsing ──────────────────────────────────────────────────────────

def _parse_reactions_from_raw(raw_dir: Path) -> dict[str, list[dict]]:
    """
    Walk all raw JSON files and return a dict of:
      message_id → [{"reaction_type": str, "display_name": str, "user_id": str}, ...]
    """
    from tqdm import tqdm as _tqdm
    reactions: dict[str, list[dict]] = defaultdict(list)
    files = list(raw_dir.glob("*.json"))
    print(f"  Scanning {len(files):,} raw files for reactions...")

    for path in _tqdm(files, desc="  Parsing reactions", unit="file"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            msgs = data if isinstance(data, list) else data.get("value", data.get("messages", []))
            for msg in msgs:
                msg_id = msg.get("id")
                raw_reactions = msg.get("reactions")
                if not msg_id or not raw_reactions:
                    continue
                for r in raw_reactions:
                    user_id = (
                        (r.get("user") or {}).get("user", {}).get("id") or ""
                    )
                    reactions[msg_id].append({
                        "reaction_type": r.get("reactionType", ""),
                        "display_name":  r.get("displayName", ""),
                        "user_id":       user_id,
                    })
        except Exception:
            continue

    total = sum(len(v) for v in reactions.values())
    print(f"  Found {total:,} reactions across {len(reactions):,} messages.")
    return dict(reactions)


# ── Emoji extraction ──────────────────────────────────────────────────────────

def _extract_emoji(text: str, emoji_lib) -> list[str]:
    """Return list of emoji characters found in text."""
    if not text or not isinstance(text, str):
        return []
    return [e["emoji"] for e in emoji_lib.emoji_list(text)]


def _emoji_name(char: str, emoji_lib) -> str:
    """Human-readable name for an emoji character."""
    try:
        return emoji_lib.demojize(char).strip(":").replace("_", " ")
    except Exception:
        return char


# ── Figures ───────────────────────────────────────────────────────────────────

# Teams reaction type → emoji character
REACTION_EMOJI: dict[str, str] = {
    "like":      "👍",
    "heart":     "❤️",
    "laugh":     "😂",
    "surprised": "😮",
    "sad":       "😢",
    "angry":     "😡",
}

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(size=14),
    xaxis=dict(showgrid=False, linecolor="#cccccc"),
    yaxis=dict(gridcolor="#eeeeee", linecolor="#cccccc"),
)


def _save_html(fig, path: Path) -> None:
    fig.write_html(str(path), include_plotlyjs="cdn", full_html=True)


def _make_figures(emoji_summary: pd.DataFrame, reaction_summary: pd.DataFrame,
                  emoji_messages: pd.DataFrame, messages_df: pd.DataFrame,
                  reactions_given: dict[str, list[tuple[str, int]]] | None = None,
                  chats_df: pd.DataFrame | None = None) -> None:
    import plotly.graph_objects as go
    import plotly.express as px

    FIGURES_DIR.mkdir(exist_ok=True)

    # 1. Top 25 emoji by usage — horizontal bar, emoji chars on y-axis
    try:
        # Normalize: strip variation selector U+FE0F so multi-codepoint emoji
        # (e.g. 👁️, 🤷‍♀️) render as single glyphs, then re-aggregate to merge
        # any pairs that were only distinguished by the variation selector.
        VS16 = "\uFE0F"
        es_norm = emoji_summary.copy()
        es_norm["emoji"] = es_norm["emoji"].str.replace(VS16, "", regex=False)
        es_norm = es_norm.groupby("emoji", as_index=False)["total_uses"].sum()
        top25 = es_norm.nlargest(25, "total_uses").iloc[::-1]  # ascending for barh
        fig = go.Figure(go.Bar(
            x=top25["total_uses"],
            y=top25["emoji"],
            orientation="h",
            marker_color="#636EFA",
            text=top25["total_uses"],
            textposition="outside",
        ))
        fig.update_layout(
            title="Top 25 Emoji by Usage",
            xaxis_title="Total uses",
            yaxis_title=None,
            height=900,
            yaxis=dict(tickfont=dict(size=20), tickmode="linear", dtick=1,
                       gridcolor="#eeeeee", linecolor="#cccccc"),
            xaxis=dict(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=14),
            margin=dict(l=60, r=60, t=60, b=40),
        )
        _save_html(fig, FIGURES_DIR / "top_emoji_bar.html")
        print("  Saved top_emoji_bar.html")
    except Exception as e:
        print(f"  [warn] top_emoji_bar.html: {e}")

    # 2a. My emoji usage — top 15 emoji used by self, horizontal bar
    try:
        self_summary = emoji_summary[emoji_summary["self_uses"] > 0].nlargest(15, "self_uses").iloc[::-1]
        fig = go.Figure(go.Bar(
            x=self_summary["self_uses"],
            y=self_summary["emoji"],
            orientation="h",
            marker_color="#EF553B",
            text=self_summary["self_uses"],
            textposition="outside",
        ))
        fig.update_layout(
            title="My Top 15 Emoji",
            xaxis_title="Times used",
            yaxis_title=None,
            height=600,
            yaxis=dict(tickfont=dict(size=20), gridcolor="#eeeeee", linecolor="#cccccc"),
            xaxis=dict(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=14),
            margin=dict(l=60, r=60, t=60, b=40),
        )
        _save_html(fig, FIGURES_DIR / "emoji_by_self.html")
        print("  Saved emoji_by_self.html")
    except Exception as e:
        print(f"  [warn] emoji_by_self.html: {e}")

    # 2b. Top 10 other senders — card grid, each card shows their top 10 emoji + counts
    try:
        sender_emoji = (
            emoji_messages[(emoji_messages["emoji_count"] > 0) & (~emoji_messages["is_self"])]
            .assign(emoji_list_parsed=lambda df: df["emoji_list"].apply(json.loads))
            .explode("emoji_list_parsed")
            .rename(columns={"emoji_list_parsed": "emoji"})
        )
        top_senders = (
            sender_emoji.groupby("sender_name")["emoji"].count()
            .nlargest(30).index.tolist()
        )
        # Per-sender top 10 emoji — typed emoji + reactions given merged
        sender_top: dict[str, list[tuple[str, int]]] = {}
        for sender in top_senders:
            combined: dict[str, int] = defaultdict(int)
            # Typed emoji
            for e in sender_emoji[sender_emoji["sender_name"] == sender]["emoji"]:
                combined[e] += 1
            # Reactions given
            if reactions_given and sender in reactions_given:
                for e, c in reactions_given[sender]:
                    combined[e] += c
            sender_top[sender] = sorted(combined.items(), key=lambda x: -x[1])

        # Build HTML card grid
        cards_html = ""
        for sender in top_senders:
            emoji_items = "".join(
                f'<div class="ei"><span class="ec">{e}</span>'
                f'<span class="cnt">{c}</span></div>'
                for e, c in sender_top[sender]
            )
            # Truncate long names so cards stay uniform
            display_name = sender if len(sender) <= 22 else sender[:20] + "…"
            cards_html += (
                f'<div class="card">'
                f'<div class="name">{display_name}</div>'
                f'<div class="emoji-row">{emoji_items}</div>'
                f'</div>\n'
            )

        subtitle = "Top 30 senders by total emoji used (excluding self) · all emoji per person (typed + reactions given)"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Emoji by Sender</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #fff; color: #222; margin: 0; padding: 24px; }}
  h1   {{ font-size: 22px; margin: 0 0 6px; }}
  p.sub {{ font-size: 13px; color: #888; margin: 0 0 24px; }}
  .grid {{ display: grid;
           grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
           gap: 16px; }}
  .card {{ border: 1px solid #e0e0e0; border-radius: 10px;
           padding: 16px 14px 14px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }}
  .name {{ font-size: 14px; font-weight: 600; margin-bottom: 12px;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .emoji-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .ei  {{ display: flex; flex-direction: column; align-items: center;
          min-width: 38px; }}
  .ec  {{ font-size: 26px; line-height: 1.2; }}
  .cnt {{ font-size: 11px; color: #666; margin-top: 2px; }}
</style>
</head>
<body>
<h1>Top 10 Senders by Emoji Usage</h1>
<p class="sub">{subtitle}</p>
<div class="grid">
{cards_html}
</div>
</body>
</html>"""

        (FIGURES_DIR / "emoji_by_others.html").write_text(html, encoding="utf-8")
        print("  Saved emoji_by_others.html")
    except Exception as e:
        print(f"  [warn] emoji_by_others.html: {e}")

    # 3. Monthly emoji usage volume — line chart, top 5 emoji as separate lines in legend
    try:
        ts_map = messages_df.set_index("message_id")["timestamp"]
        em = emoji_messages[emoji_messages["emoji_count"] > 0].copy()
        em["timestamp"] = em["message_id"].map(ts_map)
        em["month"] = pd.to_datetime(em["timestamp"]).dt.to_period("M").astype(str)

        # Explode emoji list for per-emoji monthly counts
        em_exploded = (
            em.assign(emoji_list_parsed=em["emoji_list"].apply(json.loads))
            .explode("emoji_list_parsed")
            .rename(columns={"emoji_list_parsed": "emoji"})
        )
        top5 = em_exploded["emoji"].value_counts().head(5).index.tolist()

        monthly_total = em.groupby("month")["emoji_count"].sum().reset_index()
        monthly_total.columns = ["month", "count"]

        colors = px.colors.qualitative.Plotly
        fig = go.Figure()

        # Total volume as a faint area trace
        fig.add_trace(go.Scatter(
            x=monthly_total["month"], y=monthly_total["count"],
            mode="lines+markers",
            name="total",
            line=dict(color="#cccccc", width=1),
            fill="tozeroy",
            fillcolor="rgba(200,200,200,0.15)",
        ))

        # One line per top-5 emoji, label in legend is the emoji char
        for i, char in enumerate(top5):
            monthly_e = (
                em_exploded[em_exploded["emoji"] == char]
                .groupby("month").size()
                .reset_index(name="count")
            )
            fig.add_trace(go.Scatter(
                x=monthly_e["month"], y=monthly_e["count"],
                mode="lines+markers",
                name=char,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=5),
            ))

        fig.update_layout(
            title="Monthly Emoji Usage Volume",
            xaxis_title="Month",
            yaxis_title="Emoji used",
            legend=dict(title=dict(text="Emoji", font=dict(size=16)),
                        font=dict(size=20)),
            height=450,
            xaxis=dict(tickangle=-45, showgrid=False, linecolor="#cccccc"),
            yaxis=dict(gridcolor="#eeeeee", linecolor="#cccccc"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=14),
        )
        _save_html(fig, FIGURES_DIR / "emoji_over_time.html")
        print("  Saved emoji_over_time.html")
    except Exception as e:
        print(f"  [warn] emoji_over_time.html: {e}")

    # 4. Emoji density — emoji per 100 words per sender
    try:
        # Sum emoji counts per sender
        emoji_per_sender = (
            emoji_messages.groupby("sender_name")["emoji_count"]
            .sum()
            .rename("total_emoji")
        )
        # Sum word counts per sender from messages_df
        words_per_sender = (
            messages_df[messages_df["message_type"] == "message"]
            .groupby("sender_name")["word_count"]
            .sum()
            .rename("total_words")
        )
        density = (
            pd.concat([emoji_per_sender, words_per_sender], axis=1)
            .dropna()
            .query("total_words >= 100")
        )
        density["emoji_per_100w"] = (density["total_emoji"] / density["total_words"] * 100).round(2)
        density = density.sort_values("emoji_per_100w", ascending=False).head(25)
        density = density.reset_index()  # sender_name becomes a column

        # Corpus-wide average (all senders with ≥100 words, not just top 25)
        corpus_avg = (
            emoji_per_sender.sum() / words_per_sender[words_per_sender >= 100].sum() * 100
        )

        # Identify self sender name
        self_names = set(
            messages_df[messages_df["is_self"] == True]["sender_name"].dropna().unique()
        )

        colors = [
            "#EF553B" if row["sender_name"] in self_names else "#636EFA"
            for _, row in density.iterrows()
        ]

        fig = go.Figure(go.Bar(
            x=density["emoji_per_100w"],
            y=density["sender_name"],
            orientation="h",
            marker_color=colors,
            text=density["emoji_per_100w"].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
        ))
        # Corpus average as a vertical dashed line
        fig.add_vline(
            x=corpus_avg,
            line_dash="dash",
            line_color="#888888",
            line_width=1.5,
            annotation_text=f"avg {corpus_avg:.1f}",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="#888888",
        )
        fig.update_layout(
            title="Emoji Density: Who Uses the Most Emoji Relative to How Much They Write?",
            xaxis_title="Emoji per 100 words",
            yaxis_title=None,
            height=700,
            yaxis=dict(autorange="reversed", tickfont=dict(size=13),
                       showgrid=False, linecolor="#cccccc"),
            xaxis=dict(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=14),
            margin=dict(l=160, r=80, t=60, b=50),
        )
        _save_html(fig, FIGURES_DIR / "emoji_density.html")
        print("  Saved emoji_density.html")
    except Exception as e:
        print(f"  [warn] emoji_density.html: {e}")

    # 6. Emoji density by 1:1 chat partner
    try:
        if chats_df is None:
            raise ValueError("chats_df not provided")

        # Identify self name from messages
        self_names = set(
            messages_df[messages_df["is_self"] == True]["sender_name"].dropna().unique()
        )
        if not self_names:
            raise ValueError("Could not determine self sender name")
        self_name = next(iter(self_names))

        # 1:1 chat IDs
        one_on_one_ids = set(
            chats_df[chats_df["chat_type"] == "oneOnOne"]["chat_id"]
        )

        # Build chat_id → partner name map
        def _partner(member_names_str: str) -> str:
            parts = [p.strip() for p in member_names_str.split(",")]
            others = [p for p in parts if p != self_name]
            return others[0] if others else member_names_str

        chat_to_partner = (
            chats_df[chats_df["chat_type"] == "oneOnOne"]
            .set_index("chat_id")["member_names"]
            .apply(_partner)
            .to_dict()
        )

        # Self messages in 1:1 chats only
        self_1on1 = messages_df[
            (messages_df["is_self"] == True) &
            (messages_df["chat_id"].isin(one_on_one_ids)) &
            (messages_df["message_type"] == "message")
        ].copy()
        self_1on1["partner"] = self_1on1["chat_id"].map(chat_to_partner)

        # Emoji counts for self in 1:1 chats (from emoji_messages)
        self_emoji_1on1 = emoji_messages[
            (emoji_messages["is_self"] == True) &
            (emoji_messages["chat_id"].isin(one_on_one_ids))
        ].copy()
        self_emoji_1on1["partner"] = self_emoji_1on1["chat_id"].map(chat_to_partner)

        per_partner_emoji = self_emoji_1on1.groupby("partner")["emoji_count"].sum()
        per_partner_words = self_1on1.groupby("partner")["word_count"].sum()

        density_p = (
            pd.concat([per_partner_emoji.rename("total_emoji"),
                       per_partner_words.rename("total_words")], axis=1)
            .dropna()
            .query("total_words >= 50")
        )
        density_p["emoji_per_100w"] = (
            density_p["total_emoji"] / density_p["total_words"] * 100
        ).round(2)
        density_p = density_p.sort_values("emoji_per_100w", ascending=False).head(25)
        density_p = density_p.reset_index()

        # Overall average across all qualifying 1:1 chats
        overall_avg = (
            per_partner_emoji.sum() /
            per_partner_words[per_partner_words >= 50].sum() * 100
        )

        fig = go.Figure(go.Bar(
            x=density_p["emoji_per_100w"],
            y=density_p["partner"],
            orientation="h",
            marker_color="#636EFA",
            text=density_p["emoji_per_100w"].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
        ))
        fig.add_vline(
            x=overall_avg,
            line_dash="dash",
            line_color="#888888",
            line_width=1.5,
            annotation_text=f"your avg {overall_avg:.1f}",
            annotation_position="top",
            annotation_font_size=11,
            annotation_font_color="#888888",
        )
        fig.update_layout(
            title="Your Emoji Density by 1:1 Chat Partner",
            xaxis_title="Your emoji per 100 words",
            yaxis_title=None,
            height=700,
            yaxis=dict(autorange="reversed", tickfont=dict(size=13),
                       showgrid=False, linecolor="#cccccc"),
            xaxis=dict(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=14),
            margin=dict(l=180, r=80, t=60, b=50),
        )
        _save_html(fig, FIGURES_DIR / "emoji_density_by_partner.html")
        print("  Saved emoji_density_by_partner.html")
    except Exception as e:
        print(f"  [warn] emoji_density_by_partner.html: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Emoji and reaction analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    if (OUTPUT_DIR / "emoji_report.json").exists() and not args.force:
        answer = input("outputs/09_emoji/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 09_emoji ===\n")

    for path in (MESSAGES_PATH, CHATS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)

    import emoji as emoji_lib

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading messages...")
    df = pd.read_parquet(MESSAGES_PATH)
    print(f"  {len(df):,} messages loaded.")
    chats_df = pd.read_parquet(CHATS_PATH)
    print(f"  {len(chats_df):,} chats loaded.\n")

    # Only user messages (not system events, not deleted)
    msg_df = df[
        (df["message_type"] == "message") &
        (df["is_deleted"] == False)
    ].copy()

    # ── Parse reactions from raw JSON ─────────────────────────────────────────
    print("Parsing reactions from raw JSON...")
    reactions_map: dict[str, list[dict]] = {}
    if RAW_DIR.exists():
        reactions_map = _parse_reactions_from_raw(RAW_DIR)
    else:
        print(f"  [warn] {RAW_DIR} not found — reaction data will be empty.")
    print()

    # ── Extract emoji from text ───────────────────────────────────────────────
    print("Extracting emoji from message text...")
    from tqdm import tqdm as _tqdm
    emoji_rows = []
    for _, row in _tqdm(msg_df.iterrows(), total=len(msg_df), desc="  Emoji extract", unit="msg"):
        msg_id = row["message_id"]
        text = row.get("text") or ""
        emojis = _extract_emoji(text, emoji_lib)
        rxns = reactions_map.get(msg_id, [])
        reaction_types = [r["reaction_type"] for r in rxns]

        emoji_rows.append({
            "message_id":    msg_id,
            "chat_id":       row["chat_id"],
            "sender_name":   row.get("sender_name"),
            "is_self":       bool(row.get("is_self", False)),
            "timestamp":     row.get("timestamp"),
            "emoji_list":    json.dumps(emojis),
            "emoji_count":   len(emojis),
            "is_emoji_only": bool(row.get("is_emoji_only", False)),
            "reaction_count": len(rxns),
            "reaction_types": json.dumps(reaction_types),
        })

    emoji_messages = pd.DataFrame(emoji_rows)
    emoji_messages["emoji_count"] = emoji_messages["emoji_count"].astype("int32")
    emoji_messages["reaction_count"] = emoji_messages["reaction_count"].astype("int32")

    has_emoji = (emoji_messages["emoji_count"] > 0).sum()
    has_reactions = (emoji_messages["reaction_count"] > 0).sum()
    print(f"  Messages with emoji:     {has_emoji:,}")
    print(f"  Messages with reactions: {has_reactions:,}\n")

    emoji_messages.to_parquet(OUTPUT_DIR / "emoji_messages.parquet", index=False)

    # ── emoji_summary.parquet ─────────────────────────────────────────────────
    print("Building emoji summary...")
    emoji_counts: dict[str, dict] = defaultdict(lambda: {
        "total_uses": 0, "senders": set(), "self_uses": 0, "other_uses": 0
    })

    for _, row in emoji_messages[emoji_messages["emoji_count"] > 0].iterrows():
        emojis = json.loads(row["emoji_list"])
        is_self = row["is_self"]
        sender = row["sender_name"] or "Unknown"
        for e in emojis:
            emoji_counts[e]["total_uses"] += 1
            emoji_counts[e]["senders"].add(sender)
            if is_self:
                emoji_counts[e]["self_uses"] += 1
            else:
                emoji_counts[e]["other_uses"] += 1

    emoji_summary_rows = []
    for char, stats in emoji_counts.items():
        emoji_summary_rows.append({
            "emoji":          char,
            "emoji_name":     _emoji_name(char, emoji_lib),
            "total_uses":     stats["total_uses"],
            "unique_senders": len(stats["senders"]),
            "self_uses":      stats["self_uses"],
            "other_uses":     stats["other_uses"],
        })

    emoji_summary = pd.DataFrame(emoji_summary_rows).sort_values("total_uses", ascending=False)
    emoji_summary["total_uses"] = emoji_summary["total_uses"].astype("int32")
    emoji_summary["unique_senders"] = emoji_summary["unique_senders"].astype("int32")
    emoji_summary["self_uses"] = emoji_summary["self_uses"].astype("int32")
    emoji_summary["other_uses"] = emoji_summary["other_uses"].astype("int32")
    emoji_summary.to_parquet(OUTPUT_DIR / "emoji_summary.parquet", index=False)
    print(f"  {len(emoji_summary):,} distinct emoji found.\n")

    # ── reaction_summary.parquet ──────────────────────────────────────────────
    print("Building reaction summary...")
    # Build message → is_self map
    self_map = msg_df.set_index("message_id")["is_self"].to_dict()

    rxn_stats: dict[str, dict] = defaultdict(lambda: {
        "display_name": "",
        "total_count": 0,
        "reactors": set(),
        "messages": set(),
        "self_messages": set(),
        "other_messages": set(),
    })

    for msg_id, rxns in reactions_map.items():
        is_self_msg = bool(self_map.get(msg_id, False))
        for r in rxns:
            rtype = r["reaction_type"]
            rxn_stats[rtype]["display_name"] = r.get("display_name", "")
            rxn_stats[rtype]["total_count"] += 1
            rxn_stats[rtype]["reactors"].add(r["user_id"])
            rxn_stats[rtype]["messages"].add(msg_id)
            if is_self_msg:
                rxn_stats[rtype]["self_messages"].add(msg_id)
            else:
                rxn_stats[rtype]["other_messages"].add(msg_id)

    reaction_rows = []
    for rtype, stats in rxn_stats.items():
        reaction_rows.append({
            "reaction_type":            rtype,
            "display_name":             stats["display_name"],
            "total_count":              stats["total_count"],
            "unique_reactors":          len(stats["reactors"]),
            "messages_receiving":       len(stats["messages"]),
            "self_messages_receiving":  len(stats["self_messages"]),
            "other_messages_receiving": len(stats["other_messages"]),
        })

    reaction_summary = pd.DataFrame(reaction_rows).sort_values("total_count", ascending=False)
    if not reaction_summary.empty:
        for col in ("total_count", "unique_reactors", "messages_receiving",
                    "self_messages_receiving", "other_messages_receiving"):
            reaction_summary[col] = reaction_summary[col].astype("int32")
    reaction_summary.to_parquet(OUTPUT_DIR / "reaction_summary.parquet", index=False)
    print(f"  {len(reaction_summary):,} distinct reaction types.\n")

    # ── Reactions given per sender ────────────────────────────────────────────
    # Build user_id → sender_name from messages, then tally reactions given
    uid_to_name = (
        msg_df[msg_df["sender_id"].notna()]
        .drop_duplicates("sender_id")
        .set_index("sender_id")["sender_name"]
        .to_dict()
    )
    rxn_given_raw: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rxns in reactions_map.values():
        for r in rxns:
            uid = r.get("user_id", "")
            name = uid_to_name.get(uid)
            if not name:
                continue
            rtype = r.get("reaction_type", "")
            # Named reactions (like, heart, laugh...) → map to emoji char
            # Emoji reactions (already a character) → use directly
            emoji_char = REACTION_EMOJI.get(rtype.lower())
            if emoji_char is None:
                emoji_char = rtype if rtype and not rtype.isascii() else None
            if emoji_char is None:
                continue  # skip anything that can't be represented as an emoji
            rxn_given_raw[name][emoji_char] += 1

    # Sort each sender's reactions by count descending
    reactions_given: dict[str, list[tuple[str, int]]] = {
        name: sorted(counts.items(), key=lambda x: -x[1])
        for name, counts in rxn_given_raw.items()
    }

    # ── Figures ───────────────────────────────────────────────────────────────
    print("Generating figures...")
    _make_figures(emoji_summary, reaction_summary, emoji_messages, msg_df, reactions_given, chats_df)
    print()

    # ── Report ────────────────────────────────────────────────────────────────
    top10_emoji = emoji_summary.head(10)[["emoji", "emoji_name", "total_uses"]].to_dict(orient="records")
    top10_reactions = reaction_summary.head(10)[["reaction_type", "display_name", "total_count"]].to_dict(orient="records") if not reaction_summary.empty else []

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "total_messages_analyzed": len(msg_df),
        "messages_with_emoji": int(has_emoji),
        "messages_with_reactions": int(has_reactions),
        "distinct_emoji": len(emoji_summary),
        "distinct_reaction_types": len(reaction_summary),
        "total_emoji_uses": int(emoji_summary["total_uses"].sum()),
        "total_reactions": int(reaction_summary["total_count"].sum()) if not reaction_summary.empty else 0,
        "emoji_only_messages": int(emoji_messages["is_emoji_only"].sum()),
        "top_10_emoji": top10_emoji,
        "top_10_reactions": top10_reactions,
    }
    (OUTPUT_DIR / "emoji_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    # ── Console summary ───────────────────────────────────────────────────────
    print("--- Emoji summary ---")
    print(f"  Messages analyzed:     {len(msg_df):,}")
    print(f"  Messages with emoji:   {has_emoji:,}")
    print(f"  Messages with reactions:{has_reactions:,}")
    print(f"  Distinct emoji:        {len(emoji_summary):,}")
    print(f"  Total emoji uses:      {report['total_emoji_uses']:,}")
    print(f"  Total reactions:       {report['total_reactions']:,}")
    print(f"\n  Top 10 emoji:")
    for row in top10_emoji:
        line = f"    {row['emoji_name']:<25} ({row['total_uses']} uses)"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode())
    print(f"\n  Top reactions:")
    for row in top10_reactions[:5]:
        line = f"    {row.get('display_name') or row['reaction_type']:<20} ({row['total_count']})"
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode())

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("  emoji_messages.parquet   per-message emoji extraction")
    print("  emoji_summary.parquet    per-emoji aggregate stats")
    print("  reaction_summary.parquet per-reaction-type stats")
    print("  figures/                 top_emoji_bar.html, emoji_by_self.html, emoji_by_others.html, emoji_over_time.html, emoji_density.html, emoji_density_by_partner.html")
    print("  emoji_report.json        run summary")
    print(f"\nNext: python message_miner/10_goemotions.py")


if __name__ == "__main__":
    main()
