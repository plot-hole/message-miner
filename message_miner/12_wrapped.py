"""
12_wrapped.py -- Wrapped Report: Compute Phase

Reads every prior pipeline output and computes all metrics for the
"Your Messages, Wrapped" report. Writes a single wrapped_data.json
containing every computed metric, organized by section.

No HTML rendering here. No LLM calls. Pure data aggregation.

Writes:
  outputs/12_wrapped/
  |-- wrapped_data.json      all computed metrics by section
  `-- wrapped_report.json    generation metadata (timing, row counts, errors)

Run from project root:
    python message_miner/12_wrapped.py [--since YYYY-MM-DD]
"""

import json
import os
import sys
import time
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

OUTPUT_DIR = Path("outputs/12_wrapped")

# Narrative (Section 10)
NARRATIVE_MODEL       = "gpt-5.4"
NARRATIVE_TEMPERATURE = 0.7
NARRATIVE_MAX_TOKENS  = 250
INPUT_COST_PER_MTOK   = 2.50
OUTPUT_COST_PER_MTOK  = 15.00


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _isodate(ts) -> str | None:
    """Timestamp -> ISO date string (YYYY-MM-DD), or None."""
    if ts is None:
        return None
    try:
        return pd.Timestamp(ts).date().isoformat()
    except Exception:
        return None


def _safe_float(v) -> float | None:
    """Convert to float, returning None for NaN/Inf/None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None


def assign_burst_ids(msgs: pd.DataFrame, bursts: pd.DataFrame) -> pd.DataFrame:
    """
    Range-join msgs to bursts via (chat_id, timestamp in [start_time, end_time]).
    Adds a 'burst_id' column to msgs. Rows outside any burst window get NaN.
    Requires msgs to have columns: message_id, chat_id, timestamp.
    """
    if msgs.empty or bursts.empty:
        return msgs.assign(burst_id=pd.NA)

    m = msgs[["message_id", "chat_id", "timestamp"]].copy()
    m = m[m["timestamp"].notna()].sort_values("timestamp")
    b = (
        bursts[["burst_id", "chat_id", "start_time", "end_time"]]
        .copy()
        .dropna(subset=["start_time", "end_time"])
        .sort_values("start_time")
        .rename(columns={"start_time": "ts_key"})
    )

    merged = pd.merge_asof(
        m,
        b,
        left_on="timestamp",
        right_on="ts_key",
        by="chat_id",
        direction="backward",
    )

    # Keep only messages that fall within the matched burst's end_time
    within = merged[
        merged["burst_id"].notna() & (merged["timestamp"] <= merged["end_time"])
    ][["message_id", "burst_id"]].copy()

    return msgs.merge(within, on="message_id", how="left")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data() -> dict:
    print("Loading pipeline outputs...")
    data: dict = {}

    def _parquet(key: str, path: str) -> None:
        p = Path(path)
        if not p.exists():
            print(f"  [WARN] Missing: {path}")
            data[key] = pd.DataFrame()
            return
        df = pd.read_parquet(p)
        print(f"  {path}  ({len(df):,} rows x {len(df.columns)} cols)")
        data[key] = df

    def _json(key: str, path: str) -> None:
        p = Path(path)
        if not p.exists():
            print(f"  [WARN] Missing: {path}")
            data[key] = None
            return
        obj = json.loads(p.read_text(encoding="utf-8"))
        print(f"  {path}  (loaded)")
        data[key] = obj

    _json(   "connection",        "outputs/01_connect/connection_report.json")
    _parquet("chats",             "outputs/02_extract/chats.parquet")
    _parquet("msgs",              "outputs/05_enrich/messages_enriched.parquet")
    _parquet("bursts",            "outputs/07_segment/bursts.parquet")
    _parquet("burst_topics",      "outputs/08b_topics_llm/burst_topics_llm.parquet")
    _json(   "topic_info",        "outputs/08b_topics_llm/topic_info_llm.json")
    _parquet("emoji_msgs",        "outputs/09_emoji/emoji_messages.parquet")
    _parquet("emoji_summary",     "outputs/09_emoji/emoji_summary.parquet")
    _parquet("reaction_summary",  "outputs/09_emoji/reaction_summary.parquet")
    _parquet("summaries",         "outputs/11_summarize/summaries.parquet")
    _parquet("profiles",          "outputs/11b_nlp_features/sender_profiles.parquet")
    _parquet("distinctive",       "outputs/11b_nlp_features/distinctive_words.parquet")
    _parquet("entities",          "outputs/11b_nlp_features/entity_summary.parquet")
    _parquet("questions",         "outputs/11b_nlp_features/questions.parquet")
    _json(   "nlp_report",        "outputs/11b_nlp_features/nlp_report.json")
    _parquet("classifications",   "outputs/11c_classify/classifications.parquet")
    _parquet("decisions",         "outputs/11c_classify/decisions.parquet")
    _parquet("action_items",      "outputs/11c_classify/action_items.parquet")

    # msgs_clean: canonical filtered view used throughout
    msgs = data["msgs"]
    if not msgs.empty:
        msgs_clean = msgs[
            (msgs["message_type"] == "message") &
            (msgs["is_deleted"] == False)
        ].copy()
    else:
        msgs_clean = pd.DataFrame()
    data["msgs_clean"] = msgs_clean
    print(f"\n  msgs_clean: {len(msgs_clean):,} rows after filter")

    # User identity from connection report
    conn = data["connection"] or {}
    user = conn.get("user", {})
    data["user_id"]   = user.get("id")
    data["user_name"] = user.get("name")

    # 1:1 chat lookup: chat_id -> other person's name
    chats = data["chats"]
    user_name = data["user_name"]
    oneone_map: dict[str, str] = {}
    if not chats.empty and user_name:
        oo = chats[chats["chat_type"] == "oneOnOne"]
        for _, row in oo.iterrows():
            members = row.get("member_names") or []
            if isinstance(members, str):
                try:
                    members = json.loads(members)
                except Exception:
                    members = [m.strip() for m in members.split(",")]
            others = [m for m in members if m != user_name]
            if others:
                oneone_map[row["chat_id"]] = others[0]
    data["oneone_map"] = oneone_map
    print(f"  1:1 chat map: {len(oneone_map)} chats")
    print()
    return data


# ---------------------------------------------------------------------------
# Date filter (applied in-memory after load — original parquets untouched)
# ---------------------------------------------------------------------------

def filter_data(data: dict, since: str) -> None:
    """
    Filter all in-memory dataframes to on/after *since* (YYYY-MM-DD).
    Modifies *data* in place. Original parquet files are never written to.
    """
    since_ts = pd.Timestamp(since, tz="UTC")
    print(f"Date filter: since {since}")

    # Messages
    for key in ("msgs", "msgs_clean"):
        df = data.get(key)
        if df is not None and not df.empty and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            before = len(df)
            data[key] = df[ts >= since_ts].copy()
            print(f"  {key}: {before:,} → {len(data[key]):,} rows")

    # Bursts by start_time
    bursts = data.get("bursts", pd.DataFrame())
    if not bursts.empty and "start_time" in bursts.columns:
        ts = pd.to_datetime(bursts["start_time"], utc=True, errors="coerce")
        before = len(bursts)
        data["bursts"] = bursts[ts >= since_ts].copy()
        print(f"  bursts: {before:,} → {len(data['bursts']):,} rows")

    # All burst-keyed tables — filter to surviving burst_ids
    valid_burst_ids = set(data["bursts"]["burst_id"]) if not data["bursts"].empty else set()
    for key in ("burst_topics", "summaries", "classifications", "decisions", "action_items"):
        df = data.get(key)
        if df is not None and not df.empty and "burst_id" in df.columns:
            before = len(df)
            data[key] = df[df["burst_id"].isin(valid_burst_ids)].copy()
            print(f"  {key}: {before:,} → {len(data[key]):,} rows")

    # Emoji messages by timestamp
    emoji_msgs = data.get("emoji_msgs")
    if emoji_msgs is not None and not emoji_msgs.empty and "timestamp" in emoji_msgs.columns:
        ts = pd.to_datetime(emoji_msgs["timestamp"], utc=True, errors="coerce")
        before = len(emoji_msgs)
        data["emoji_msgs"] = emoji_msgs[ts >= since_ts].copy()
        print(f"  emoji_msgs: {before:,} → {len(data['emoji_msgs']):,} rows")

    # Questions by timestamp
    questions = data.get("questions")
    if questions is not None and not questions.empty and "timestamp" in questions.columns:
        ts = pd.to_datetime(questions["timestamp"], utc=True, errors="coerce")
        before = len(questions)
        data["questions"] = questions[ts >= since_ts].copy()
        print(f"  questions: {before:,} → {len(data['questions']):,} rows")

    print()


# ---------------------------------------------------------------------------
# Section 0: The Headline
# ---------------------------------------------------------------------------

def compute_section_0(data: dict) -> dict:
    print("Section 0: The Headline...")
    msgs_clean = data["msgs_clean"]
    user_name  = data["user_name"]

    if msgs_clean.empty:
        return {"error": "missing msgs_clean"}

    self_msgs       = msgs_clean[msgs_clean["is_self"] == True]
    self_chat_ids   = set(self_msgs["chat_id"].unique())
    total_words     = int(self_msgs["word_count"].sum())

    other_senders = msgs_clean[
        msgs_clean["chat_id"].isin(self_chat_ids) &
        (msgs_clean["is_self"] == False)
    ]["sender_name"].nunique()

    return {
        "user_name":           user_name,
        "date_first":          _isodate(msgs_clean["timestamp"].min()),
        "date_last":           _isodate(msgs_clean["timestamp"].max()),
        "total_messages_sent": int(len(self_msgs)),
        "total_conversations": int(len(self_chat_ids)),
        "total_people":        int(other_senders),
        "total_words_written": total_words,
        "novel_equivalent":    round(total_words / 80_000, 1),
    }


# ---------------------------------------------------------------------------
# Section 1: Your Top People
# ---------------------------------------------------------------------------

def compute_section_1(data: dict) -> dict:
    print("Section 1: Your Top People...")
    msgs_clean = data["msgs_clean"]
    oneone_map = data["oneone_map"]

    if msgs_clean.empty:
        return {"error": "missing msgs_clean"}

    oo_ids = set(oneone_map.keys())
    oo     = msgs_clean[msgs_clean["chat_id"].isin(oo_ids)]

    counts = (
        oo.groupby(["chat_id", "is_self"])
        .size()
        .unstack(fill_value=0)
    )
    counts.columns.name = None
    if True  not in counts.columns: counts[True]  = 0
    if False not in counts.columns: counts[False] = 0
    counts = counts.rename(columns={True: "my", False: "their"})
    counts["total"] = counts["my"] + counts["their"]
    counts = counts.reset_index().sort_values("total", ascending=False).head(12)
    counts["name"] = counts["chat_id"].map(oneone_map)
    counts = counts.dropna(subset=["name"])

    top_10: list[dict] = []
    for _, row in counts.iterrows():
        my  = int(row["my"])
        thr = int(row["their"])
        top_10.append({
            "name":           row["name"],
            "my_messages":    my,
            "their_messages": thr,
            "total":          int(row["total"]),
            "ratio":          round(my / thr, 2) if thr > 0 else None,
        })

    # ride_or_die
    ride_or_die = None
    if top_10 and not counts.empty:
        best_cid  = counts.iloc[0]["chat_id"]
        chat_msgs = msgs_clean[msgs_clean["chat_id"] == best_cid]
        active    = chat_msgs["timestamp"].dt.date.nunique()
        total_m   = int(len(chat_msgs))
        ride_or_die = {
            "name":                    top_10[0]["name"],
            "total_messages":          total_m,
            "first_message_date":      _isodate(chat_msgs["timestamp"].min()),
            "last_message_date":       _isodate(chat_msgs["timestamp"].max()),
            "active_days":             active,
            "messages_per_active_day": round(total_m / active, 1) if active > 0 else None,
        }

    # new_connections — 1:1 chats only
    oo_chat_ids = set(oneone_map.keys())
    date_last   = msgs_clean["timestamp"].max()
    boundary    = date_last - pd.Timedelta(days=365)
    first_seen  = (
        msgs_clean[
            (msgs_clean["is_self"] == False) &
            (msgs_clean["chat_id"].isin(oo_chat_ids))
        ]
        .groupby("sender_name")["timestamp"]
        .min()
    )
    new = first_seen[first_seen > boundary].sort_values(ascending=False)
    new_connections = {"count": int(len(new)), "names": new.head(10).index.tolist()}

    return {
        "top_10_people":   top_10,
        "ride_or_die":     ride_or_die,
        "new_connections": new_connections,
    }


# ---------------------------------------------------------------------------
# Section 2: Your Calendar
# ---------------------------------------------------------------------------

def compute_section_2(data: dict) -> dict:
    print("Section 2: Your Calendar...")
    msgs_clean = data["msgs_clean"]
    summaries  = data["summaries"]
    bursts     = data["bursts"]

    if msgs_clean.empty:
        return {"error": "missing msgs_clean"}

    self_msgs = msgs_clean[msgs_clean["is_self"] == True].copy()

    # tz_offset_hours: sliding 8-hour window on weekday self messages
    wkday = self_msgs[self_msgs["timestamp"].dt.dayofweek < 5]
    hcnt  = wkday["timestamp"].dt.hour.value_counts().reindex(range(24), fill_value=0)
    best_h, best_s = 0, -1
    for h in range(24):
        s = sum(int(hcnt.iloc[(h + i) % 24]) for i in range(8))
        if s > best_s:
            best_s, best_h = s, h
    tz = 8 - best_h   # best_h is the UTC hour that maps to ~8am local
    if tz >  12: tz -= 24
    if tz < -12: tz += 24
    print(f"  tz_offset_hours: {tz}  (peak window starts at UTC {best_h:02d}:00)")
    data["tz_offset_hours"] = tz  # share with later sections

    self_msgs["local_ts"]   = self_msgs["timestamp"] + pd.Timedelta(hours=tz)
    self_msgs["local_date"] = self_msgs["local_ts"].dt.date
    self_msgs["dow"]        = self_msgs["local_ts"].dt.dayofweek
    self_msgs["hour"]       = self_msgs["local_ts"].dt.hour

    # heatmap_data (168 cells)
    hmap = (
        self_msgs.groupby(["dow", "hour"]).size()
        .reset_index(name="count")
    )
    hmap_dict = {(int(r["dow"]), int(r["hour"])): int(r["count"]) for _, r in hmap.iterrows()}
    heatmap_data = [
        {"day": d, "hour": h, "count": hmap_dict.get((d, h), 0)}
        for d in range(7) for h in range(24)
    ]

    # peak_date
    date_counts  = self_msgs.groupby("local_date").size()
    peak_date_val = date_counts.idxmax()
    peak_count    = int(date_counts.max())

    peak_summaries: list[str] = []
    if not summaries.empty and not bursts.empty:
        pk      = pd.Timestamp(peak_date_val, tz="UTC")
        pk_end  = pk + pd.Timedelta(days=1)
        day_bids = bursts[
            (bursts["start_time"] < pk_end) &
            (bursts["end_time"]   > pk) &
            (bursts["self_msg_count"] > 0)
        ]["burst_id"]
        peak_summaries = (
            summaries[
                summaries["burst_id"].isin(day_bids) &
                summaries["summary"].notna()
            ]["summary"]
            .head(5)
            .tolist()
        )

    # peak_hour
    hour_counts   = self_msgs["hour"].value_counts()
    peak_hour_val = int(hour_counts.idxmax())
    peak_hour_cnt = int(hour_counts.max())

    # after_hours_pct
    outside = self_msgs[
        (self_msgs["hour"] < 8) | (self_msgs["hour"] >= 18) | (self_msgs["dow"] >= 5)
    ]
    after_hours_pct = round(len(outside) / len(self_msgs) * 100, 1) if len(self_msgs) > 0 else None

    # monthly_volume
    self_msgs["ym"] = self_msgs["local_ts"].dt.strftime("%Y-%m")
    monthly = (
        self_msgs.groupby("ym").size()
        .reset_index(name="count")
        .sort_values("ym")
    )
    monthly_volume = [{"month": r["ym"], "count": int(r["count"])} for _, r in monthly.iterrows()]
    print(f"  heatmap_data: {len(heatmap_data)} cells  |  monthly_volume: {len(monthly_volume)} months")

    # most active season
    def _month_to_season(ym: str) -> str:
        mo = int(ym.split("-")[1])
        if mo in (12, 1, 2):  return "Winter"
        if mo in (3, 4, 5):   return "Spring"
        if mo in (6, 7, 8):   return "Summer"
        return "Fall"

    season_totals: dict[str, int] = {"Spring": 0, "Summer": 0, "Fall": 0, "Winter": 0}
    for row in monthly_volume:
        season_totals[_month_to_season(row["month"])] += row["count"]
    most_active_season = max(season_totals, key=lambda s: season_totals[s])
    season_emojis = {"Spring": "🌸", "Summer": "☀️", "Fall": "🍂", "Winter": "❄️"}

    return {
        "tz_offset_hours": tz,
        "heatmap_data":    heatmap_data,
        "peak_date": {
            "date":      peak_date_val.isoformat(),
            "count":     peak_count,
            "summaries": peak_summaries,
        },
        "peak_hour": {"hour": peak_hour_val, "count": peak_hour_cnt},
        "after_hours_pct":    after_hours_pct,
        "monthly_volume":     monthly_volume,
        "most_active_season": {"season": most_active_season, "emoji": season_emojis[most_active_season], "count": season_totals[most_active_season]},
    }


# ---------------------------------------------------------------------------
# Section 3: Your Topics
# ---------------------------------------------------------------------------

def compute_section_3(data: dict, wrapped: dict) -> dict:
    print("Section 3: Your Topics...")
    bursts       = data["bursts"]
    burst_topics = data["burst_topics"]   # burst_topics_llm.parquet
    topic_info   = data["topic_info"]     # topic_info_llm.json  {"topics": [...]}
    summaries    = data["summaries"]
    oneone_map   = data["oneone_map"]

    if burst_topics.empty:
        return {"error": "missing burst_topics"}

    # topic_info lookup: topic_id (int) -> {label, keywords}
    topic_lookup: dict[int, dict] = {}
    topics_list = topic_info.get("topics", []) if isinstance(topic_info, dict) else topic_info
    for t in topics_list:
        tid = int(t["id"])
        topic_lookup[tid] = {
            "label":    t.get("label", str(tid)),
            "keywords": t.get("keywords", []),
        }

    # User's burst_ids
    user_burst_ids = set(bursts[bursts["self_msg_count"] > 0]["burst_id"].unique())
    user_bt = burst_topics[burst_topics["burst_id"].isin(user_burst_ids)].copy()

    # Count each burst toward every topic it appears in (primary + secondary).
    # Stack primary and secondary into a single series, drop nulls.
    primary_series   = user_bt[["burst_id", "primary_topic_id"]].rename(
        columns={"primary_topic_id": "topic_id"})
    secondary_series = user_bt[["burst_id", "secondary_topic_id"]].rename(
        columns={"secondary_topic_id": "topic_id"})
    all_assignments = pd.concat([primary_series, secondary_series], ignore_index=True)
    all_assignments = all_assignments.dropna(subset=["topic_id"])
    all_assignments["topic_id"] = all_assignments["topic_id"].astype(int)
    # Each burst counts once per topic (drop dupes in case primary==secondary somehow)
    all_assignments = all_assignments.drop_duplicates()

    topic_counts = all_assignments.groupby("topic_id").size().sort_values(ascending=False)

    top_10_topics = []
    for tid, cnt in topic_counts.head(10).items():
        info = topic_lookup.get(int(tid), {"label": str(tid), "keywords": []})
        top_10_topics.append({
            "topic_id":    int(tid),
            "label":       info["label"],
            "keywords":    info["keywords"],
            "burst_count": int(cnt),
        })

    # topic_over_time: top 7 topics, computed from burst start_time
    top7_ids    = [t["topic_id"] for t in top_10_topics[:7]]
    top7_labels = {t["topic_id"]: t["label"] for t in top_10_topics[:7]}

    burst_times = bursts[["burst_id", "start_time"]].copy()
    burst_times["start_time"] = pd.to_datetime(burst_times["start_time"], utc=True, errors="coerce")
    burst_times = burst_times.dropna(subset=["start_time"])
    burst_times["month"] = burst_times["start_time"].dt.strftime("%Y-%m")

    tot_assignments = all_assignments[all_assignments["topic_id"].isin(top7_ids)]
    tot_merged = tot_assignments.merge(burst_times[["burst_id", "month"]], on="burst_id", how="left")
    tot_merged = tot_merged.dropna(subset=["month"])

    tot_months = sorted(tot_merged["month"].unique())
    tot_by_topic: dict[int, dict[str, int]] = {}
    for tid, grp in tot_merged.groupby("topic_id"):
        tot_by_topic[int(tid)] = grp.groupby("month").size().to_dict()

    topic_over_time = {
        "months": tot_months,
        "series": [
            {
                "topic_label": top7_labels[tid],
                "values": [int(tot_by_topic.get(tid, {}).get(m, 0)) for m in tot_months],
            }
            for tid in top7_ids
        ],
    }

    # topic_over_time_quarterly: same data aggregated to quarters for charting
    import math
    def _month_to_quarter(m: str) -> str:
        y, mo = m.split("-")
        return f"{y} Q{math.ceil(int(mo) / 3)}"

    tot_quarters = sorted(set(_month_to_quarter(m) for m in tot_months),
                          key=lambda q: tot_months[[_month_to_quarter(m) for m in tot_months].index(q)])
    q_by_topic: dict[int, dict[str, int]] = {tid: {} for tid in top7_ids}
    for m in tot_months:
        q = _month_to_quarter(m)
        for tid in top7_ids:
            q_by_topic[tid][q] = q_by_topic[tid].get(q, 0) + tot_by_topic.get(tid, {}).get(m, 0)

    topic_over_time_quarterly = {
        "quarters": tot_quarters,
        "series": [
            {
                "topic_label": top7_labels[tid],
                "values": [int(q_by_topic[tid].get(q, 0)) for q in tot_quarters],
            }
            for tid in top7_ids
        ],
    }

    # topic_by_person: top 5 people — count primary+secondary per chat
    top5_names    = [p["name"] for p in wrapped["section_1"]["top_10_people"][:5]]
    burst_to_chat = bursts.set_index("burst_id")["chat_id"].to_dict()
    bt_with_chat  = all_assignments.copy()
    bt_with_chat["chat_id"] = bt_with_chat["burst_id"].map(burst_to_chat)

    topic_by_person = []
    for person in top5_names:
        p_chats  = {cid for cid, n in oneone_map.items() if n == person}
        p_rows   = bt_with_chat[bt_with_chat["chat_id"].isin(p_chats)]
        p_counts = p_rows.groupby("topic_id").size().sort_values(ascending=False).head(3)
        topic_by_person.append({
            "person": person,
            "topics": [
                {"label": topic_lookup.get(int(tid), {"label": str(tid)})["label"], "count": int(cnt)}
                for tid, cnt in p_counts.items()
            ],
        })

    # longest_conversation
    user_bursts_df = bursts[bursts["self_msg_count"] > 0]
    longest_conversation = None
    if not user_bursts_df.empty:
        row = user_bursts_df.loc[user_bursts_df["msg_count"].idxmax()]
        parts = row.get("participants", "")
        if isinstance(parts, str):
            try:    parts = json.loads(parts)
            except: parts = [parts] if parts else []
        sum_row     = summaries[summaries["burst_id"] == row["burst_id"]]
        summary_txt = sum_row["summary"].iloc[0] if not sum_row.empty else None
        longest_conversation = {
            "burst_id":         row["burst_id"],
            "msg_count":        int(row["msg_count"]),
            "duration_minutes": round(float(row["duration_seconds"]) / 60, 1),
            "participants":     list(parts) if isinstance(parts, list) else [str(parts)],
            "summary":          summary_txt,
        }

    return {
        "top_10_topics":             top_10_topics,
        "topic_over_time":           topic_over_time,
        "topic_over_time_quarterly": topic_over_time_quarterly,
        "topic_by_person":           topic_by_person,
        "longest_conversation": longest_conversation,
    }


# ---------------------------------------------------------------------------
# Section 4: Your Conversations
# ---------------------------------------------------------------------------

def compute_section_4(data: dict, wrapped: dict) -> dict:
    print("Section 4: Your Conversations...")
    msgs_clean = data["msgs_clean"]
    bursts     = data["bursts"]
    user_id    = data["user_id"]
    tz         = wrapped["section_2"].get("tz_offset_hours", 0)

    if msgs_clean.empty or bursts.empty:
        return {"error": "missing data"}

    user_bursts    = bursts[bursts["self_msg_count"] > 0].copy()
    total_bursts   = int(len(user_bursts))

    self_msgs = msgs_clean[msgs_clean["is_self"] == True].copy()
    self_msgs["local_date"] = (
        self_msgs["timestamp"] + pd.Timedelta(hours=tz)
    ).dt.date
    active_dates = sorted(self_msgs["local_date"].dropna().unique())

    conversations_per_day = (
        round(total_bursts / len(active_dates), 1) if active_dates else None
    )

    # initiator_pct: is the first message in each user burst from the user?
    msgs_wb = assign_burst_ids(msgs_clean, user_bursts)
    msgs_wb = msgs_wb[msgs_wb["burst_id"].notna()]
    burst_first = (
        msgs_wb.sort_values("timestamp")
        .groupby("burst_id")
        .first()
        .reset_index()
    )
    initiated     = int((burst_first["sender_id"] == user_id).sum())
    initiator_pct = round(initiated / total_bursts * 100, 1) if total_bursts > 0 else None

    # longest_streak
    longest_streak = {"days": 0, "start_date": None, "end_date": None}
    if active_dates:
        best_len = run_len = 1
        best_start = run_start = active_dates[0]
        best_end = active_dates[0]
        for i in range(1, len(active_dates)):
            if (active_dates[i] - active_dates[i - 1]).days == 1:
                run_len += 1
                if run_len > best_len:
                    best_len, best_start, best_end = run_len, run_start, active_dates[i]
            else:
                run_len, run_start = 1, active_dates[i]
        longest_streak = {
            "days":       best_len,
            "start_date": best_start.isoformat(),
            "end_date":   best_end.isoformat(),
        }

    # longest_gap
    longest_gap = {"days": 0, "start_date": None, "end_date": None}
    if len(active_dates) > 1:
        gaps    = [(active_dates[i] - active_dates[i - 1], active_dates[i - 1], active_dates[i])
                   for i in range(1, len(active_dates))]
        max_gap = max(gaps, key=lambda x: x[0])
        longest_gap = {
            "days":       max_gap[0].days,
            "start_date": max_gap[1].isoformat(),
            "end_date":   max_gap[2].isoformat(),
        }

    return {
        "total_bursts":          total_bursts,
        "conversations_per_day": conversations_per_day,
        "initiator_pct":         initiator_pct,
        "longest_streak":        longest_streak,
        "longest_gap":           longest_gap,
    }


# ---------------------------------------------------------------------------
# Section 5: Your Response Patterns
# ---------------------------------------------------------------------------

def compute_section_5(data: dict, wrapped: dict) -> dict:
    print("Section 5: Your Response Patterns...")
    msgs_clean = data["msgs_clean"]
    oneone_map = data["oneone_map"]
    top10      = wrapped["section_1"]["top_10_people"]

    if msgs_clean.empty:
        return {"error": "missing msgs_clean"}

    oo_ids = set(oneone_map.keys())

    # My median response time
    my_resp = msgs_clean[
        (msgs_clean["is_self"] == True) &
        (msgs_clean["response_time_seconds"] > 0) &
        (msgs_clean["response_time_seconds"] <= 14400)
    ]["response_time_seconds"]
    my_median = round(float(my_resp.median()), 1) if len(my_resp) > 0 else None

    # Fastest/slowest responder (other people in 1:1 chats)
    oo_other = msgs_clean[
        msgs_clean["chat_id"].isin(oo_ids) &
        (msgs_clean["is_self"] == False) &
        (msgs_clean["response_time_seconds"] > 0) &
        (msgs_clean["response_time_seconds"] <= 14400)
    ].copy()
    oo_other["other_name"] = oo_other["chat_id"].map(oneone_map)
    person_resp = (
        oo_other.groupby("other_name")["response_time_seconds"]
        .agg(["median", "count"])
        .reset_index()
    )
    person_resp = person_resp[person_resp["count"] >= 10]

    fastest = slowest = None
    if not person_resp.empty:
        fi = person_resp["median"].idxmin()
        si = person_resp["median"].idxmax()
        fastest = {
            "name":                   person_resp.loc[fi, "other_name"],
            "median_response_time_s": round(float(person_resp.loc[fi, "median"]), 1),
        }
        slowest = {
            "name":                   person_resp.loc[si, "other_name"],
            "median_response_time_s": round(float(person_resp.loc[si, "median"]), 1),
        }

    # My response time by top 10 people
    my_resp_by_person = []
    for p in top10:
        p_chats = {cid for cid, n in oneone_map.items() if n == p["name"]}
        r = msgs_clean[
            msgs_clean["chat_id"].isin(p_chats) &
            (msgs_clean["is_self"] == True) &
            (msgs_clean["response_time_seconds"] > 0) &
            (msgs_clean["response_time_seconds"] <= 14400)
        ]["response_time_seconds"]
        if len(r) > 0:
            my_resp_by_person.append({
                "name":                   p["name"],
                "median_response_time_s": round(float(r.median()), 1),
            })
    my_resp_by_person.sort(key=lambda x: x["median_response_time_s"])

    # Response time trend by month
    trend = msgs_clean[
        (msgs_clean["is_self"] == True) &
        (msgs_clean["response_time_seconds"] > 0) &
        (msgs_clean["response_time_seconds"] <= 14400)
    ].copy()
    trend["ym"] = trend["timestamp"].dt.strftime("%Y-%m")
    trend_agg = (
        trend.groupby("ym")["response_time_seconds"]
        .median()
        .reset_index()
        .sort_values("ym")
    )
    response_time_trend = [
        {"month": r["ym"], "median_response_time_s": round(float(r["response_time_seconds"]), 1)}
        for _, r in trend_agg.iterrows()
    ]

    return {
        "my_median_response_time_s": my_median,
        "fastest_responder":         fastest,
        "slowest_responder":         slowest,
        "my_response_by_person":     my_resp_by_person,
        "response_time_trend":       response_time_trend,
    }


# ---------------------------------------------------------------------------
# Section 6: Your Emoji Game
# ---------------------------------------------------------------------------

def compute_section_6(data: dict, wrapped: dict) -> dict:
    print("Section 6: Your Emoji Game...")
    emoji_msgs    = data["emoji_msgs"]
    msgs_clean    = data["msgs_clean"]
    emoji_summary = data["emoji_summary"]
    oneone_map    = data["oneone_map"]
    top10         = wrapped["section_1"]["top_10_people"]

    if emoji_msgs.empty:
        return {"error": "missing emoji_msgs"}

    self_emoji = emoji_msgs[emoji_msgs["is_self"] == True].copy()

    # top_10_emoji: use emoji_summary (has correct Unicode characters already)
    top_counts = (
        emoji_summary[emoji_summary["self_uses"] > 0]
        .sort_values("self_uses", ascending=False)
        .head(10)
    )
    top_10_emoji = [
        {"emoji": row["emoji"], "count": int(row["self_uses"])}
        for _, row in top_counts.iterrows()
    ]

    # emoji_density
    total_self_emoji = int(self_emoji["emoji_count"].sum())
    total_self_words = int(msgs_clean[msgs_clean["is_self"] == True]["word_count"].sum())
    emoji_density    = round(total_self_emoji / total_self_words * 100, 3) if total_self_words > 0 else None

    total_all_emoji  = int(emoji_msgs["emoji_count"].sum())
    total_all_words  = int(msgs_clean["word_count"].sum())
    org_avg_density  = round(total_all_emoji / total_all_words * 100, 3) if total_all_words > 0 else None

    # signature_emoji
    signature_emoji = None
    if not emoji_summary.empty:
        user_total   = int(emoji_summary["self_uses"].sum())
        corpus_total = int(emoji_summary["total_uses"].sum())
        if user_total > 0 and corpus_total > 0:
            sig = emoji_summary[emoji_summary["self_uses"] >= 5].copy()
            sig["user_rate"]   = sig["self_uses"]  / user_total
            sig["corpus_rate"] = sig["total_uses"] / corpus_total
            sig["ratio"]       = sig["user_rate"] / sig["corpus_rate"].replace(0, np.nan)
            sig = sig.dropna(subset=["ratio"])
            if not sig.empty:
                best = sig.loc[sig["ratio"].idxmax()]
                signature_emoji = {
                    "emoji":            best["emoji"],
                    "user_count":       int(best["self_uses"]),
                    "ratio_vs_average": round(float(best["ratio"]), 2),
                }

    # emoji_partners: top 5 1:1 chats by user emoji density
    oo_ids = set(oneone_map.keys())
    se_oo  = self_emoji[self_emoji["chat_id"].isin(oo_ids)]
    sm_oo  = msgs_clean[msgs_clean["chat_id"].isin(oo_ids) & (msgs_clean["is_self"] == True)]

    ep = pd.DataFrame({
        "emoji_count": se_oo.groupby("chat_id")["emoji_count"].sum(),
        "word_count":  sm_oo.groupby("chat_id")["word_count"].sum(),
        "msg_count":   sm_oo.groupby("chat_id").size(),
    }).dropna()
    ep = ep[ep["msg_count"] >= 20].copy()
    ep["density"] = ep["emoji_count"] / ep["word_count"].replace(0, np.nan) * 100
    ep = ep.dropna(subset=["density"])
    ep["name"] = ep.index.map(oneone_map)
    ep = ep.sort_values("density", ascending=False).head(5)
    emoji_partners = [
        {"name": row["name"], "emoji_density": round(float(row["density"]), 3)}
        for _, row in ep.iterrows()
        if pd.notna(row["name"])
    ]

    # top_emoji_senders: top 10 non-self people by total emoji used
    import json as _json
    others_emoji = emoji_msgs[emoji_msgs["is_self"] == False].copy()

    # Per-sender totals (all messages, not just those with emoji)
    sender_totals = others_emoji.groupby("sender_name").agg(
        total_msgs  = ("message_id", "count"),
        total_emoji = ("emoji_count", "sum"),
    ).reset_index()
    sender_totals = sender_totals[sender_totals["total_emoji"] >= 10].copy()

    # Join word counts from msgs_clean for emoji-per-word ranking
    word_counts = (
        msgs_clean[msgs_clean["is_self"] == False]
        .groupby("sender_name")["word_count"]
        .sum()
        .reset_index()
        .rename(columns={"word_count": "total_words"})
    )
    sender_totals = sender_totals.merge(word_counts, on="sender_name", how="left")
    sender_totals["total_words"] = sender_totals["total_words"].fillna(1)
    sender_totals["emoji_per_100_words"] = (
        sender_totals["total_emoji"] / sender_totals["total_words"] * 100
    ).round(2)
    top10_senders = sender_totals.sort_values("emoji_per_100_words", ascending=False).head(8)

    # Build corpus-level emoji frequency (others only, for fair comparison)
    corpus_emoji_counts: dict[str, int] = {}
    for elist in others_emoji[others_emoji["emoji_count"] > 0]["emoji_list"].dropna():
        try:
            for e in _json.loads(elist):
                corpus_emoji_counts[e] = corpus_emoji_counts.get(e, 0) + 1
        except Exception:
            pass
    corpus_total = sum(corpus_emoji_counts.values())

    # Per-sender emoji breakdown for signature detection
    sender_emoji_counts: dict[str, dict[str, int]] = {}
    for _, row in others_emoji[others_emoji["emoji_count"] > 0].iterrows():
        name = row["sender_name"]
        try:
            for e in _json.loads(row["emoji_list"]):
                if name not in sender_emoji_counts:
                    sender_emoji_counts[name] = {}
                sender_emoji_counts[name][e] = sender_emoji_counts[name].get(e, 0) + 1
        except Exception:
            pass

    top_emoji_senders = []
    for _, row in top10_senders.iterrows():
        name = row["sender_name"]
        ec = sender_emoji_counts.get(name, {})
        sender_total = sum(ec.values())
        sig = None
        if sender_total > 0 and corpus_total > 0:
            best_ratio, best_e = 0.0, None
            for e, cnt in ec.items():
                if cnt < 3:
                    continue
                s_rate = cnt / sender_total
                c_rate = corpus_emoji_counts.get(e, 0) / corpus_total
                if c_rate == 0:
                    continue
                ratio = s_rate / c_rate
                if ratio > best_ratio:
                    best_ratio, best_e = ratio, e
            if best_e:
                sig = {"emoji": best_e, "ratio": round(best_ratio, 1)}
        top_emoji_senders.append({
            "name":                name,
            "total_emoji":         int(row["total_emoji"]),
            "emoji_per_100_words": float(row["emoji_per_100_words"]),
            "signature_emoji":     sig,
        })

    # most_reacted_message
    reacted = msgs_clean[
        (msgs_clean["is_self"] == True) & (msgs_clean["reaction_count"] > 0)
    ]
    most_reacted = None
    if not reacted.empty:
        best = reacted.loc[reacted["reaction_count"].idxmax()]
        most_reacted = {
            "text":           str(best["text"]) if pd.notna(best.get("text")) else None,
            "reaction_count": int(best["reaction_count"]),
            "date":           _isodate(best["timestamp"]),
        }

    return {
        "top_10_emoji":          top_10_emoji,
        "emoji_density":         emoji_density,
        "org_avg_emoji_density": org_avg_density,
        "signature_emoji":       signature_emoji,
        "emoji_partners":        emoji_partners,
        "most_reacted_message":  most_reacted,
        "top_emoji_senders":     top_emoji_senders,
    }


# ---------------------------------------------------------------------------
# Section 7: Your Emotional Landscape
# ---------------------------------------------------------------------------

def compute_section_7(data: dict, wrapped: dict) -> dict:
    print("Section 7: Your Emotional Landscape...")
    emotions   = data["emotions"]
    bursts     = data["bursts"]
    summaries  = data["summaries"]
    oneone_map = data["oneone_map"]
    top5_names = [p["name"] for p in wrapped["section_1"]["top_10_people"][:5]]

    if emotions.empty:
        return {"error": "missing emotions"}

    self_em      = emotions[emotions["is_self"] == True]
    nn_self      = self_em[self_em["top_emotion"] != "neutral"]
    nn_all       = emotions[emotions["top_emotion"] != "neutral"]

    # emotion_profile
    em_counts = nn_self["top_emotion"].value_counts()
    em_total  = int(em_counts.sum())
    emotion_profile = [
        {"emotion": em, "proportion": round(int(cnt) / em_total, 4)}
        for em, cnt in em_counts.head(8).items()
    ] if em_total > 0 else []

    # vibe_vs_average
    corp_counts = nn_all["top_emotion"].value_counts()
    corp_total  = int(corp_counts.sum())
    user_props  = em_counts   / em_total   * 100 if em_total   > 0 else pd.Series(dtype=float)
    corp_props  = corp_counts / corp_total * 100 if corp_total > 0 else pd.Series(dtype=float)

    all_ems = set(user_props.index) | set(corp_props.index)
    diffs = sorted(
        [
            {
                "emotion": em,
                "user_pct": round(float(user_props.get(em, 0)), 2),
                "avg_pct":  round(float(corp_props.get(em, 0)), 2),
                "diff":     round(float(user_props.get(em, 0)) - float(corp_props.get(em, 0)), 2),
            }
            for em in all_ems
        ],
        key=lambda x: x["diff"],
        reverse=True,
    )
    vibe_vs_average = {
        "more_than_average": diffs[:3],
        "less_than_average": diffs[-3:][::-1],
    }

    # emotion_by_person
    emotion_by_person = []
    for person in top5_names:
        p_chats = {cid for cid, n in oneone_map.items() if n == person}
        p_nn    = emotions[
            emotions["chat_id"].isin(p_chats) &
            (emotions["is_self"] == True) &
            (emotions["top_emotion"] != "neutral")
        ]
        p_counts = p_nn["top_emotion"].value_counts()
        p_total  = int(p_counts.sum())
        emotion_by_person.append({
            "person": person,
            "emotions": [
                {"emotion": em, "proportion": round(int(cnt) / p_total, 4)}
                for em, cnt in p_counts.head(5).items()
            ] if p_total > 0 else [],
        })

    # most_intense_burst: range-join emotions to user bursts (min 15 self messages)
    user_bursts = bursts[bursts["self_msg_count"] >= 15].copy()
    em_wb       = assign_burst_ids(emotions, user_bursts)
    em_wb       = em_wb[em_wb["burst_id"].notna() & (em_wb["is_self"] == True)]

    most_intense = None
    if not em_wb.empty:
        b_stats = (
            em_wb.groupby("burst_id")
            .agg(total=("top_emotion", "count"), non_neutral=("is_neutral", lambda x: (~x).sum()))
            .reset_index()
        )
        b_stats = b_stats[b_stats["total"] >= 15].copy()
        b_stats["nn_pct"] = b_stats["non_neutral"] / b_stats["total"]
        if not b_stats.empty:
            best    = b_stats.loc[b_stats["nn_pct"].idxmax()]
            bid     = best["burst_id"]
            brow    = bursts[bursts["burst_id"] == bid]
            parts   = []
            if not brow.empty:
                p = brow.iloc[0].get("participants", "")
                if isinstance(p, str):
                    try:    parts = json.loads(p)
                    except: parts = [p] if p else []
                else:
                    parts = list(p) if p else []
            srow    = summaries[summaries["burst_id"] == bid]
            sum_txt = srow["summary"].iloc[0] if not srow.empty else None
            most_intense = {
                "burst_id":        bid,
                "non_neutral_pct": round(float(best["nn_pct"]) * 100, 1),
                "msg_count":       int(best["total"]),
                "participants":    parts,
                "summary":         sum_txt,
            }

    # mood_over_time: user's top 3 non-neutral emotions
    top3 = em_counts.head(3).index.tolist()
    m3   = nn_self[nn_self["top_emotion"].isin(top3) & nn_self["timestamp"].notna()].copy()
    m3["ym"] = m3["timestamp"].dt.strftime("%Y-%m")
    nn_self_copy = nn_self[nn_self["timestamp"].notna()].copy()
    nn_self_copy["ym"] = nn_self_copy["timestamp"].dt.strftime("%Y-%m")
    month_totals = nn_self_copy.groupby("ym").size()

    mood_months = sorted(m3["ym"].dropna().unique())
    series: dict[str, list] = {em: [] for em in top3}
    for mo in mood_months:
        mo_total = int(month_totals.get(mo, 0))
        for em in top3:
            cnt = int(m3[(m3["ym"] == mo) & (m3["top_emotion"] == em)].shape[0])
            series[em].append(round(cnt / mo_total, 4) if mo_total > 0 else 0.0)

    return {
        "emotion_profile":    emotion_profile,
        "vibe_vs_average":    vibe_vs_average,
        "emotion_by_person":  emotion_by_person,
        "most_intense_burst": most_intense,
        "mood_over_time": {
            "emotions": top3,
            "months":   mood_months,
            "series":   series,
        },
    }


# ---------------------------------------------------------------------------
# Section 8: Your Words
# ---------------------------------------------------------------------------

def compute_section_8(data: dict, wrapped: dict) -> dict:
    print("Section 8: Your Words...")
    profiles    = data["profiles"]
    distinctive = data["distinctive"]
    questions   = data["questions"]
    entities    = data["entities"]
    nlp_report  = data["nlp_report"]
    msgs_clean  = data["msgs_clean"]
    oneone_map  = data["oneone_map"]

    result: dict = {}

    # vocabulary_size, type_token_ratio, reading_level
    user_grade: float | None = None
    if not profiles.empty and "is_self" in profiles.columns:
        sp = profiles[profiles["is_self"] == True]
        if not sp.empty:
            row = sp.iloc[0]
            result["vocabulary_size"]  = int(row["unique_words"])
            result["type_token_ratio"] = round(float(row["type_token_ratio"]), 4)
            user_grade = _safe_float(row["flesch_kincaid_grade"])
        else:
            result["vocabulary_size"]  = None
            result["type_token_ratio"] = None
    else:
        result["vocabulary_size"]  = None
        result["type_token_ratio"] = None

    org_grade: float | None = None
    if nlp_report and "corpus_averages" in nlp_report:
        org_grade = _safe_float(nlp_report["corpus_averages"].get("flesch_kincaid_grade"))

    result["reading_level"] = {
        "user_grade":    round(user_grade, 2) if user_grade is not None else None,
        "org_avg_grade": round(org_grade,  2) if org_grade  is not None else None,
    }

    # distinctive_words: top 10 by rank
    if not distinctive.empty and "is_self" in distinctive.columns:
        dw = distinctive[distinctive["is_self"] == True].nsmallest(10, "rank")
        result["distinctive_words"] = [
            {"word": row["word"], "tfidf_score": round(float(row["tfidf_score"]), 4)}
            for _, row in dw.iterrows()
        ]
    else:
        result["distinctive_words"] = []

    # questions
    if not questions.empty and "is_self" in questions.columns:
        self_q     = questions[questions["is_self"] == True]
        total_q    = int(len(self_q))
        answered   = int(self_q["was_answered"].sum()) if "was_answered" in self_q.columns else 0
        answer_rate = round(answered / total_q * 100, 1) if total_q > 0 else None

        oo_q = self_q[self_q["chat_id"].isin(oneone_map.keys())].copy()
        oo_q["recipient"] = oo_q["chat_id"].map(oneone_map)
        recip = oo_q.groupby("recipient").size().sort_values(ascending=False).head(5)
        top_recipients = [
            {"name": name, "question_count": int(cnt)} for name, cnt in recip.items()
        ]
        result["questions"] = {
            "total_asked":    total_q,
            "answer_rate":    answer_rate,
            "top_recipients": top_recipients,
        }
    else:
        result["questions"] = {"total_asked": 0, "answer_rate": None, "top_recipients": []}

    # top_entities
    if not entities.empty and "self_mentions" in entities.columns:
        top_ents = (
            entities[entities["self_mentions"] > 0]
            .sort_values("self_mentions", ascending=False)
            .head(15)
        )
        result["top_entities"] = [
            {
                "entity":   row["entity_text"],
                "type":     row.get("entity_label", ""),
                "mentions": int(row["self_mentions"]),
            }
            for _, row in top_ents.iterrows()
        ]
    else:
        result["top_entities"] = []

    return result


# ---------------------------------------------------------------------------
# Section 9: Your Conversations, Classified
# ---------------------------------------------------------------------------

def compute_section_9(data: dict, wrapped: dict) -> dict:
    print("Section 9: Conversations, Classified...")
    classifications = data["classifications"]
    bursts          = data["bursts"]
    decisions       = data["decisions"]
    action_items    = data["action_items"]
    summaries       = data["summaries"]
    oneone_map      = data["oneone_map"]
    top5_names      = [p["name"] for p in wrapped["section_1"]["top_10_people"][:12]]

    if classifications.empty or bursts.empty:
        return {"error": "missing classifications or bursts"}

    user_burst_ids = set(bursts[bursts["self_msg_count"] > 0]["burst_id"].unique())
    user_cls       = classifications[classifications["burst_id"].isin(user_burst_ids)].copy()

    # function_distribution
    fc = user_cls.groupby("function", observed=True).size().reset_index(name="count")
    fc = fc.sort_values("count", ascending=False)
    ft = int(fc["count"].sum())
    function_distribution = [
        {
            "function": str(r["function"]),
            "count":    int(r["count"]),
            "pct":      round(int(r["count"]) / ft * 100, 1),
        }
        for _, r in fc.iterrows()
    ]

    # function_by_person
    burst_to_chat = bursts.set_index("burst_id")["chat_id"].to_dict()
    function_by_person = []
    for person in top5_names:
        p_chats  = {cid for cid, n in oneone_map.items() if n == person}
        p_cls    = user_cls[user_cls["burst_id"].map(burst_to_chat).isin(p_chats)]
        p_counts = p_cls.groupby("function", observed=True).size().sort_values(ascending=False).head(3)
        function_by_person.append({
            "person":    person,
            "functions": [{"function": str(f), "count": int(c)} for f, c in p_counts.items()],
        })

    # function_over_time: top 5 functions
    top5_funcs     = [r["function"] for r in function_distribution[:5]]
    burst_times    = bursts[["burst_id", "start_time"]].copy()
    cls_time       = user_cls.merge(burst_times, on="burst_id", how="left")
    cls_time       = cls_time[cls_time["function"].isin(top5_funcs)]
    cls_time["ym"] = cls_time["start_time"].dt.strftime("%Y-%m")

    fot_months = sorted(cls_time["ym"].dropna().unique())
    fot_series = [
        {
            "function": f,
            "values":   [
                int(cls_time[(cls_time["function"] == f) & (cls_time["ym"] == m)].shape[0])
                for m in fot_months
            ],
        }
        for f in top5_funcs
    ]
    function_over_time = {"months": fot_months, "series": fot_series}

    # function_over_time_quarterly: same data aggregated to quarters
    import math
    def _m2q(m: str) -> str:
        y, mo = m.split("-")
        return f"{y} Q{math.ceil(int(mo) / 3)}"

    fot_quarters = sorted(set(_m2q(m) for m in fot_months),
                          key=lambda q: fot_months[[_m2q(m) for m in fot_months].index(q)])
    fq_by_func: dict[str, dict[str, int]] = {f: {} for f in top5_funcs}
    for m in fot_months:
        q = _m2q(m)
        for s in fot_series:
            f = s["function"]
            fq_by_func[f][q] = fq_by_func[f].get(q, 0) + int(
                cls_time[(cls_time["function"] == f) & (cls_time["ym"] == m)].shape[0]
            )
    function_over_time_quarterly = {
        "quarters": fot_quarters,
        "series": [
            {"function": f, "values": [int(fq_by_func[f].get(q, 0)) for q in fot_quarters]}
            for f in top5_funcs
        ],
    }

    # depth metrics
    avg_depth = round(float(user_cls["depth"].dropna().mean()), 2) if not user_cls.empty else None

    user_cls_chat       = user_cls.copy()
    user_cls_chat["chat_id"] = user_cls_chat["burst_id"].map(burst_to_chat)
    oo_ids              = set(oneone_map.keys())
    oo_cls              = user_cls_chat[user_cls_chat["chat_id"].isin(oo_ids)].copy()
    oo_cls["other_name"] = oo_cls["chat_id"].map(oneone_map)
    p_depth             = (
        oo_cls.groupby("other_name")["depth"]
        .agg(["mean", "count"])
        .reset_index()
    )
    p_depth = p_depth[p_depth["count"] >= 10]
    deepest_person = None
    if not p_depth.empty:
        best = p_depth.loc[p_depth["mean"].idxmax()]
        deepest_person = {
            "name":      best["other_name"],
            "avg_depth": round(float(best["mean"]), 2),
        }
        # Debug: show selected deepest-person burst details
        dp_chat = {cid for cid, n in oneone_map.items() if n == best["other_name"]}
        dp_cls  = user_cls[user_cls["burst_id"].map(burst_to_chat).isin(dp_chat)]
        dp_deep = dp_cls[dp_cls["depth"].notna()].sort_values("depth", ascending=False).head(1)
        if not dp_deep.empty:
            bid      = dp_deep.iloc[0]["burst_id"]
            brow     = bursts[bursts["burst_id"] == bid]
            srow     = summaries[summaries["burst_id"] == bid]
            print(f"  [debug §9 deepest] person={best['other_name']}  avg_depth={round(float(best['mean']),2)}  convo_count={int(best['count'])}")
            if not brow.empty:
                print(f"    top burst: depth={dp_deep.iloc[0]['depth']}  msg_count={brow.iloc[0]['msg_count']}  participants={brow.iloc[0]['participants']}")
            if not srow.empty:
                summary_text = srow.iloc[0]["summary"]
                print(f"    summary: {str(summary_text)[:200]}")

    # recent_decisions (10 most recent)
    recent_decisions: list[dict] = []
    if not decisions.empty:
        for _, r in decisions.sort_values("timestamp", ascending=False).head(10).iterrows():
            recent_decisions.append({
                "decision_text": r["decision_text"],
                "participants":  r.get("participants", ""),
                "date":          _isodate(r["timestamp"]),
            })

    # recent_action_items (last 30 days relative to corpus max timestamp)
    recent_action_items: list[dict] = []
    if not action_items.empty:
        ts_series = [action_items["timestamp"].max()]
        if not decisions.empty:
            ts_series.append(decisions["timestamp"].max())
        max_ts  = max(ts_series)
        cutoff  = max_ts - pd.Timedelta(days=30)
        recent  = action_items[action_items["timestamp"] >= cutoff].sort_values("timestamp")
        raw_items: list[dict] = []
        for _, r in recent.iterrows():
            raw_items.append({
                "owner":        r["owner"],
                "task":         r["task"],
                "participants": r.get("participants", ""),
                "date":         _isodate(r["timestamp"]),
            })

        # Fix 3: deduplicate — same owner + similar task within the same week
        def _task_similarity(a: str, b: str) -> float:
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        deduped: list[dict] = []
        for item in raw_items:
            is_dup = False
            for kept in deduped:
                if kept["owner"] == item["owner"]:
                    # Within 7 days?
                    d1 = date.fromisoformat(item["date"])
                    d2 = date.fromisoformat(kept["date"])
                    if abs((d1 - d2).days) <= 7 and _task_similarity(item["task"], kept["task"]) >= 0.60:
                        is_dup = True
                        break
            if not is_dup:
                deduped.append(item)

        recent_action_items = deduped

    # ── Questions ────────────────────────────────────────────────────────────
    # Extract from msgs_clean: messages containing "?" that aren't just URLs
    msgs_clean   = data["msgs_clean"]
    my_questions:     list[dict] = []
    other_questions:  list[dict] = []
    questions_stats:  dict       = {}

    if not msgs_clean.empty and "text" in msgs_clean.columns:
        # Only messages in conversations the user participated in
        self_chat_ids = set(msgs_clean[msgs_clean["is_self"] == True]["chat_id"].unique())
        conv_msgs = msgs_clean[msgs_clean["chat_id"].isin(self_chat_ids)].copy()

        # Filter to messages containing "?" — exclude if "?" only appears in a URL
        def _has_question(text: str) -> bool:
            import re
            # Remove URLs, then check for "?"
            stripped = re.sub(r'https?://\S+', '', str(text))
            return '?' in stripped

        q_mask  = conv_msgs["text"].apply(_has_question)
        q_msgs  = conv_msgs[q_mask].copy()

        self_q  = q_msgs[q_msgs["is_self"] == True]
        other_q = q_msgs[q_msgs["is_self"] == False]

        questions_stats = {
            "you_asked":    int(len(self_q)),
            "others_asked": int(len(other_q)),
        }

        # Top questioners (others only)
        top_questioners = (
            other_q.groupby("sender_name")
            .size()
            .sort_values(ascending=False)
            .head(3)
        )
        questions_stats["top_questioners"] = [
            {"name": name, "count": int(count)}
            for name, count in top_questioners.items()
        ]

        # Recent samples (last 30 days of corpus)
        if not q_msgs.empty:
            max_ts  = q_msgs["timestamp"].max()
            cutoff  = max_ts - pd.Timedelta(days=30)
            recent_q = q_msgs[q_msgs["timestamp"] >= cutoff]

            recent_self_q  = recent_q[recent_q["is_self"] == True]
            recent_other_q = recent_q[recent_q["is_self"] == False]

            import re as _re
            def _clean_q(text: str) -> str:
                # Strip URLs, collapse whitespace
                t = _re.sub(r'https?://\S+', '', str(text)).strip()
                return ' '.join(t.split())

            for _, r in recent_self_q.tail(15).iterrows():
                cleaned = _clean_q(r["text"])
                if cleaned and '?' in cleaned:
                    my_questions.append({
                        "text": cleaned[:200],
                        "date": _isodate(r["timestamp"]),
                    })

            for _, r in recent_other_q.tail(15).iterrows():
                cleaned = _clean_q(r["text"])
                if cleaned and '?' in cleaned:
                    other_questions.append({
                        "sender": r["sender_name"],
                        "text":   cleaned[:200],
                        "date":   _isodate(r["timestamp"]),
                    })

    return {
        "function_distribution": function_distribution,
        "function_by_person":    function_by_person,
        "function_over_time":             function_over_time,
        "function_over_time_quarterly":   function_over_time_quarterly,
        "avg_depth":             avg_depth,
        "deepest_person":        deepest_person,
        "recent_decisions":      recent_decisions,
        "recent_action_items":   recent_action_items,
        "questions_stats":       questions_stats,
        "my_questions":          my_questions,
        "other_questions":       other_questions,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Section 10: Personalized Narrative
# ---------------------------------------------------------------------------

def _build_narrative_context(wrapped: dict) -> dict:
    """Extract the most narratively interesting metrics for the LLM."""
    ctx: dict = {}

    s0 = wrapped.get("section_0", {})
    ctx["user_name"]           = s0.get("user_name")
    ctx["date_range"]          = f"{s0.get('date_first')} to {s0.get('date_last')}"
    ctx["total_messages_sent"] = s0.get("total_messages_sent")
    ctx["total_words_written"] = s0.get("total_words_written")
    ctx["total_people"]        = s0.get("total_people")
    ctx["novel_equivalent"]    = s0.get("novel_equivalent")

    s1 = wrapped.get("section_1", {})
    ctx["ride_or_die"]          = (s1.get("ride_or_die") or {}).get("name")
    ctx["ride_or_die_messages"] = (s1.get("ride_or_die") or {}).get("total_messages")
    ctx["top_5_people"]         = [p["name"] for p in s1.get("top_10_people", [])[:5]]
    ctx["new_connections_count"] = (s1.get("new_connections") or {}).get("count")

    s2 = wrapped.get("section_2", {})
    ctx["peak_hour"]       = s2.get("peak_hour")
    ctx["after_hours_pct"] = s2.get("after_hours_pct")
    ctx["peak_date"]       = (s2.get("peak_date") or {}).get("date")
    ctx["peak_date_count"] = (s2.get("peak_date") or {}).get("count")

    s3 = wrapped.get("section_3", {})
    ctx["top_3_topics"] = [t["label"] for t in s3.get("top_10_topics", [])[:3]]

    s4 = wrapped.get("section_4", {})
    ctx["initiator_pct"]       = s4.get("initiator_pct")
    ctx["longest_streak_days"] = (s4.get("longest_streak") or {}).get("days")
    ctx["longest_gap_days"]    = (s4.get("longest_gap") or {}).get("days")

    s5 = wrapped.get("section_5", {})
    ctx["median_response_time_s"] = s5.get("my_median_response_time_s")
    ctx["fastest_responder"]      = (s5.get("fastest_responder") or {}).get("name")

    s6 = wrapped.get("section_6", {})
    top_emoji = s6.get("top_10_emoji", [])
    ctx["top_emoji"]      = top_emoji[0]["emoji"] if top_emoji else None
    sig = s6.get("signature_emoji")
    ctx["signature_emoji"] = sig.get("emoji") if sig else None

    s8 = wrapped.get("section_8", {})
    ctx["vocabulary_size"]   = s8.get("vocabulary_size")
    ctx["reading_level"]     = (s8.get("reading_level") or {}).get("user_grade")
    ctx["distinctive_words"] = [w["word"] for w in s8.get("distinctive_words", [])[:5]]
    ctx["top_entities"]      = [e["entity"] for e in s8.get("top_entities", [])[:5]]

    s9 = wrapped.get("section_9", {})
    ctx["top_3_functions"] = [f["function"] for f in s9.get("function_distribution", [])[:3]]
    ctx["avg_depth"]       = s9.get("avg_depth")
    ctx["deepest_person"]  = (s9.get("deepest_person") or {}).get("name")

    return ctx


def _pick_sample_summaries(data: dict, n: int = 50) -> list[dict]:
    """
    Select n random conversation summaries from high-depth (4-5) bursts
    the user participated in.
    """
    import random

    summaries       = data.get("summaries", pd.DataFrame())
    classifications = data.get("classifications", pd.DataFrame())
    bursts          = data.get("bursts", pd.DataFrame())

    if summaries.empty or classifications.empty or bursts.empty:
        return []

    merged = (
        summaries[summaries["summary"].notna() & summaries["skip_reason"].isna()]
        .merge(classifications[["burst_id", "depth"]], on="burst_id", how="inner")
        .merge(bursts[["burst_id", "participants", "start_time", "self_msg_count"]], on="burst_id", how="inner")
    )

    pool = merged[
        (merged["self_msg_count"] > 0) &
        (merged["depth"] >= 4)
    ].copy()

    if pool.empty:
        pool = merged[merged["self_msg_count"] > 0].copy()

    if pool.empty:
        return []

    pool["start_time"] = pd.to_datetime(pool["start_time"], utc=True, errors="coerce")
    pool = pool.dropna(subset=["start_time"])

    sample_rows = pool.sample(n=min(n, len(pool)), random_state=None)

    return [
        {
            "date":         str(row["start_time"].date()),
            "participants": row["participants"],
            "summary":      row["summary"][:500],
        }
        for _, row in sample_rows.iterrows()
    ]


def compute_section_10(wrapped: dict, data: dict | None = None) -> dict:
    """Generate the personalized Year in Messages narrative via one LLM call."""
    print("Section 10: Generating narrative...")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("  [WARN] OPENAI_API_KEY not set — skipping narrative.")
        return {"narrative": None, "error": "OPENAI_API_KEY not set"}

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    ctx     = _build_narrative_context(wrapped)
    samples = _pick_sample_summaries(data) if data else []

    samples_text = ""
    if samples:
        lines = ["\n\nSample conversation summaries:"]
        for i, s in enumerate(samples, 1):
            lines.append(f"\n[Sample {i} — {s['date']} — {s['participants']}]\n{s['summary']}")
        samples_text = "\n".join(lines)

    prompt = (
        "Below is messaging data and conversation summaries for a workplace chat user.\n\n"
        "Write a single short paragraph — 4 to 6 sentences — that reads like a sharp character observation. "
        "Say something specific and non-obvious about how this person actually communicates. "
        "Be direct and opinionated. Do not recap the data or list statistics. "
        "Do not use filler phrases. Do not invent anything not grounded in the data.\n\n"
        f"Data:\n{json.dumps(ctx, indent=2, default=str)}"
        f"{samples_text}"
    )

    try:
        response = client.chat.completions.create(
            model=NARRATIVE_MODEL,
            temperature=NARRATIVE_TEMPERATURE,
            max_completion_tokens=NARRATIVE_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        narrative_text = response.choices[0].message.content.strip()
        in_tok  = response.usage.prompt_tokens
        out_tok = response.usage.completion_tokens
        cost    = (in_tok * INPUT_COST_PER_MTOK + out_tok * OUTPUT_COST_PER_MTOK) / 1_000_000

        print(f"  Tokens: {in_tok} in / {out_tok} out  |  Cost: ${cost:.4f}")

        return {
            "narrative":      narrative_text,
            "model":          NARRATIVE_MODEL,
            "input_tokens":   in_tok,
            "output_tokens":  out_tok,
            "cost_usd":       round(cost, 4),
        }

    except Exception as exc:
        print(f"  [WARN] Narrative generation failed: {exc}")
        print("  The report will be generated without a personalized narrative.")
        return {"narrative": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate the Wrapped report")
    parser.add_argument(
        "--since", type=str, default=None,
        help="Only include data on or after this date (YYYY-MM-DD). "
             "Original parquet files are not modified."
    )
    args = parser.parse_args()

    t0 = time.time()

    print("=" * 60)
    print("WRAPPED REPORT -- COMPUTE + NARRATE PHASE")
    print("=" * 60)
    print()
    print("NOTE: The generated report will contain personal messaging data")
    print("including names, message statistics, conversation summaries,")
    print("and quoted messages. Review before sharing.")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    if args.since:
        filter_data(data, args.since)

    section_defs = [
        ("section_0", compute_section_0, False),
        ("section_1", compute_section_1, False),
        ("section_2", compute_section_2, False),
        ("section_3", compute_section_3, True),
        ("section_4", compute_section_4, True),
        ("section_5", compute_section_5, True),
        ("section_6", compute_section_6, True),
        ("section_8", compute_section_8, True),
        ("section_9", compute_section_9, True),
    ]

    wrapped: dict = {}
    section_errors: list[str] = []

    for name, fn, needs_wrapped in section_defs:
        try:
            wrapped[name] = fn(data, wrapped) if needs_wrapped else fn(data)
            if "error" in wrapped[name]:
                section_errors.append(f"{name}: {wrapped[name]['error']}")
        except Exception as exc:
            import traceback
            section_errors.append(f"{name}: {exc}")
            print(f"  [ERROR] {name}: {exc}")
            traceback.print_exc()
            wrapped[name] = {"error": str(exc)}

    # Narrate phase (Section 10)
    try:
        wrapped["section_10"] = compute_section_10(wrapped, data)
        if wrapped["section_10"].get("error"):
            section_errors.append(f"section_10: {wrapped['section_10']['error']}")
    except Exception as exc:
        import traceback
        section_errors.append(f"section_10: {exc}")
        print(f"  [ERROR] section_10: {exc}")
        traceback.print_exc()
        wrapped["section_10"] = {"narrative": None, "error": str(exc)}

    # Write outputs
    print("\nWriting outputs...")
    out_path = OUTPUT_DIR / "wrapped_data.json"
    out_path.write_text(
        json.dumps(wrapped, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  {out_path}")

    elapsed = round(time.time() - t0, 1)
    s10 = wrapped.get("section_10", {})
    report = {
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds":   elapsed,
        "sections_computed": list(wrapped.keys()),
        "section_errors":    section_errors,
        "narrative_model":   s10.get("model"),
        "narrative_cost_usd": s10.get("cost_usd"),
        "row_counts": {
            "msgs_clean":      len(data["msgs_clean"]),
            "bursts":          len(data["bursts"]),
            "classifications": len(data["classifications"]),
        },
    }
    rep_path = OUTPUT_DIR / "wrapped_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  {rep_path}")

    # Render phase
    render_html(wrapped)

    print()
    print("-" * 60)
    print("  COMPUTE + NARRATE + RENDER COMPLETE")
    print("-" * 60)
    print(f"  Sections:  {len(wrapped)}")
    print(f"  Errors:    {len(section_errors)}")
    print(f"  Elapsed:   {elapsed}s")
    if section_errors:
        print("  Section errors:")
        for e in section_errors:
            print(f"    {e}")
    print()
    print(f"Data:  {out_path}")
    print(f"HTML:  {OUTPUT_DIR / 'wrapped.html'}")


# ---------------------------------------------------------------------------
# Render phase
# ---------------------------------------------------------------------------

def render_html(wrapped: dict) -> None:
    """Render wrapped_data into a self-contained HTML file via Jinja2."""
    import plotly
    from jinja2 import Environment, FileSystemLoader

    plotly_js_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    plotly_js = plotly_js_path.read_text(encoding="utf-8")

    def format_duration(seconds) -> str:
        if seconds is None:
            return "N/A"
        s = float(seconds)
        if s < 60:    return f"{s:.0f}s"
        if s < 3600:  return f"{s / 60:.0f}m"
        if s < 86400: return f"{s / 3600:.1f}h"
        return f"{s / 86400:.1f}d"

    def format_number(n) -> str:
        if n is None:
            return "N/A"
        try:
            return f"{int(n):,}"
        except (ValueError, TypeError):
            return str(n)

    env = Environment(
        loader=FileSystemLoader(Path(__file__).parent / "templates"),
        autoescape=False,
    )
    env.filters["duration"] = format_duration
    env.filters["number"]   = format_number

    template = env.get_template("wrapped_template.html")
    html = template.render(
        data=wrapped,
        s0=wrapped.get("section_0", {}),
        s1=wrapped.get("section_1", {}),
        s2=wrapped.get("section_2", {}),
        s3=wrapped.get("section_3", {}),
        s4=wrapped.get("section_4", {}),
        s5=wrapped.get("section_5", {}),
        s6=wrapped.get("section_6", {}),
        s8=wrapped.get("section_8", {}),
        s9=wrapped.get("section_9", {}),
        s10=wrapped.get("section_10", {}),
        generated_at=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        plotly_js=plotly_js,
    )

    output_path = OUTPUT_DIR / "wrapped.html"
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"  {output_path}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
