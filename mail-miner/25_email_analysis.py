"""
25_email_analysis.py — Email-Specific Analysis

Computes email-specific analytics and writes structured JSON outputs.
No LLM calls. Pure pandas/numpy computation on enriched email data.

Reads:
  outputs/23_enrich_mail/messages_mail_enriched.parquet
  outputs/22_edge_cases_mail/messages_mail_flagged.parquet
  outputs/20_extract_mail/extraction_report.json

Writes:
  outputs/25_email_analysis/
    network_graph.json
    response_patterns.json
    thread_analysis.json
    internal_external.json
    temporal_patterns.json
    email_analysis_report.json

Run from project root:
    python mail-miner/25_email_analysis.py [--force]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

ENRICHED_PATH = Path("outputs/23_enrich_mail/messages_mail_enriched.parquet")
FLAGGED_PATH = Path("outputs/22_edge_cases_mail/messages_mail_flagged.parquet")
EXTRACTION_REPORT_PATH = Path("outputs/20_extract_mail/extraction_report.json")
OUTPUT_DIR = Path("outputs/25_email_analysis")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _parse_recipients(val) -> list[dict]:
    """Parse a JSON-string recipient list into a list of dicts."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = json.loads(val)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _user_domain(user_email: str) -> str:
    if "@" in user_email:
        return user_email.split("@", 1)[1].lower()
    return ""


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} minutes"
    hours = minutes / 60
    if hours < 24:
        return f"{hours:.1f} hours"
    return f"{hours/24:.1f} days"


# ── Analysis 1: Network Graph ─────────────────────────────────────────────────

def build_network_graph(df: pd.DataFrame, user_email: str, user_domain: str) -> dict:
    print("  Building network graph...")
    ue = user_email.lower()

    # Explode recipients so each (message, recipient) is one row
    print("    Exploding recipient lists...")
    records = []
    for _, row in df.iterrows():
        sender = (row.get("sender_email") or "").lower().strip()
        sender_name = (row.get("sender_name") or "").strip()
        ts = row.get("received_datetime")
        is_self = row.get("is_self", False)

        to_list = _parse_recipients(row.get("to_recipients"))
        cc_list = _parse_recipients(row.get("cc_recipients"))
        all_recips = to_list + cc_list

        for recip in all_recips:
            recip_email = (recip.get("email") or "").lower().strip()
            recip_name = (recip.get("name") or "").strip()
            if recip_email:
                records.append({
                    "sender": sender,
                    "sender_name": sender_name,
                    "recipient": recip_email,
                    "recipient_name": recip_name,
                    "ts": ts,
                    "is_self": is_self,
                })

    recip_df = pd.DataFrame(records)

    # Node computation: collect all contacts (people who exchanged with user)
    contact_info: dict[str, dict] = {}

    def _update_contact(email: str, name: str, ts, sent_to_user: int = 0, recv_from_user: int = 0):
        if not email or email == ue:
            return
        if email not in contact_info:
            contact_info[email] = {
                "email": email,
                "name": name or email,
                "sent_to_user": 0,
                "received_from_user": 0,
                "is_internal": email.endswith(f"@{user_domain}") if user_domain else False,
                "first_contact": ts,
                "last_contact": ts,
            }
        else:
            if name and not contact_info[email]["name"]:
                contact_info[email]["name"] = name
            if ts is not None:
                existing_first = contact_info[email]["first_contact"]
                existing_last = contact_info[email]["last_contact"]
                if existing_first is None or (ts is not None and ts < existing_first):
                    contact_info[email]["first_contact"] = ts
                if existing_last is None or (ts is not None and ts > existing_last):
                    contact_info[email]["last_contact"] = ts
        contact_info[email]["sent_to_user"] += sent_to_user
        contact_info[email]["received_from_user"] += recv_from_user

    if not recip_df.empty:
        # Emails where non-user sender sent to user (or user was a recipient)
        # sent_to_user: sender != user, recipient == user (or user in recipients when sender != user)
        sent_to_user_df = recip_df[(recip_df["sender"] != ue) & (recip_df["recipient"] == ue)]
        for _, row in sent_to_user_df.iterrows():
            _update_contact(row["sender"], row["sender_name"], row["ts"], sent_to_user=1)

        # received_from_user: sender == user, recipient != user
        recv_from_user_df = recip_df[(recip_df["sender"] == ue) & (recip_df["recipient"] != ue)]
        for _, row in recv_from_user_df.iterrows():
            _update_contact(row["recipient"], row["recipient_name"], row["ts"], recv_from_user=1)

    # Also catch senders who didn't appear in recipient lists
    inbound = df[~df["is_self"]]
    for _, row in inbound.iterrows():
        sender = (row.get("sender_email") or "").lower().strip()
        sender_name = (row.get("sender_name") or "").strip()
        ts = row.get("received_datetime")
        if sender and sender != ue and sender not in contact_info:
            contact_info[sender] = {
                "email": sender,
                "name": sender_name or sender,
                "sent_to_user": 1,
                "received_from_user": 0,
                "is_internal": sender.endswith(f"@{user_domain}") if user_domain else False,
                "first_contact": ts,
                "last_contact": ts,
            }

    # Compute total_exchange and format timestamps
    for c in contact_info.values():
        c["total_exchange"] = c["sent_to_user"] + c["received_from_user"]
        for tf in ("first_contact", "last_contact"):
            v = c[tf]
            if v is not None and hasattr(v, "isoformat"):
                c[tf] = v.isoformat()
            elif v is None:
                c[tf] = None

    sorted_contacts = sorted(contact_info.values(), key=lambda x: x["total_exchange"], reverse=True)
    top_20 = sorted_contacts[:20]
    top_30_emails = {c["email"] for c in sorted_contacts[:30]}

    # Edge computation (involving user, top 30 contacts, weight >= 3)
    edges_dict: dict[tuple, int] = {}
    if not recip_df.empty:
        # user -> contact
        user_sent = recip_df[(recip_df["sender"] == ue) & (recip_df["recipient"].isin(top_30_emails))]
        for _, row in user_sent.iterrows():
            key = (ue, row["recipient"])
            edges_dict[key] = edges_dict.get(key, 0) + 1

        # contact -> user
        contact_recv = recip_df[(recip_df["sender"].isin(top_30_emails)) & (recip_df["recipient"] == ue)]
        for _, row in contact_recv.iterrows():
            key = (row["sender"], ue)
            edges_dict[key] = edges_dict.get(key, 0) + 1

    edges = [
        {"source": src, "target": tgt, "weight": w}
        for (src, tgt), w in edges_dict.items()
        if w >= 3
    ]
    edges.sort(key=lambda e: e["weight"], reverse=True)

    internal_contacts = sum(1 for c in contact_info.values() if c["is_internal"])
    external_contacts = len(contact_info) - internal_contacts

    print(f"    Found {len(contact_info):,} contacts, {len(edges)} edges.")
    return {
        "user_email": user_email,
        "total_contacts": len(contact_info),
        "internal_contacts": internal_contacts,
        "external_contacts": external_contacts,
        "top_20_contacts": top_20,
        "edges": edges,
    }


# ── Analysis 2: Response Patterns ────────────────────────────────────────────

def build_response_patterns(df: pd.DataFrame, top_contacts: list[dict]) -> dict:
    print("  Computing response patterns...")

    # User's overall response behavior
    user_replies = df[
        df["is_self"].fillna(False) &
        df["is_reply"].fillna(False) &
        df["response_time_seconds"].notna() &
        (df["response_time_seconds"] > 0) &
        (df["response_time_seconds"] <= 604800)
    ]["response_time_seconds"]

    def _pct(series, threshold_s):
        if series.empty:
            return 0.0
        return round(100 * (series <= threshold_s).sum() / len(series), 1)

    if not user_replies.empty:
        # same-day: compare calendar date of reply vs date of the original inbound
        # Approximate: response_time < 24h and sent during same day — use < 86400 as proxy
        overall = {
            "reply_count": int(len(user_replies)),
            "median_response_time_s": round(float(user_replies.median()), 0),
            "mean_response_time_s": round(float(user_replies.mean()), 0),
            "p25_response_time_s": round(float(user_replies.quantile(0.25)), 0),
            "p75_response_time_s": round(float(user_replies.quantile(0.75)), 0),
            "under_1hr_pct": _pct(user_replies, 3600),
            "under_4hr_pct": _pct(user_replies, 14400),
            "under_24hr_pct": _pct(user_replies, 86400),
            "same_day_pct": _pct(user_replies, 86400),  # approximation
        }
    else:
        overall = {
            "reply_count": 0,
            "median_response_time_s": None,
            "mean_response_time_s": None,
            "p25_response_time_s": None,
            "p75_response_time_s": None,
            "under_1hr_pct": 0.0,
            "under_4hr_pct": 0.0,
            "under_24hr_pct": 0.0,
            "same_day_pct": 0.0,
        }

    # Response time by contact
    my_response_time_to = []
    their_response_time_to_me = []

    # Build thread sequences for "their response to me"
    # Sort all messages by conversation + time
    thread_df = df.sort_values(["conversation_id", "received_datetime"]).reset_index(drop=True)

    for contact in top_contacts:
        c_email = contact["email"].lower()

        # My response time to them: user replies in threads involving this contact
        user_reply_rows = df[
            df["is_self"].fillna(False) &
            df["is_reply"].fillna(False) &
            df["response_time_seconds"].notna() &
            (df["response_time_seconds"] > 0) &
            (df["response_time_seconds"] <= 604800)
        ]
        # Filter to threads where this contact is involved
        contact_thread_ids = df[
            df["sender_email"].fillna("").str.lower() == c_email
        ]["conversation_id"].unique()
        my_rt = user_reply_rows[user_reply_rows["conversation_id"].isin(contact_thread_ids)]["response_time_seconds"]
        if len(my_rt) >= 5:
            my_response_time_to.append({
                "name": contact["name"],
                "email": contact["email"],
                "median_s": round(float(my_rt.median()), 0),
                "count": int(len(my_rt)),
            })

        # Their response time to me: find contact replies after user's messages in shared threads
        gap_times = []
        for conv_id in contact_thread_ids:
            conv = thread_df[thread_df["conversation_id"] == conv_id].copy()
            for i in range(len(conv) - 1):
                cur = conv.iloc[i]
                nxt = conv.iloc[i + 1]
                if (
                    cur.get("is_self", False) and
                    (nxt.get("sender_email") or "").lower() == c_email
                ):
                    cur_dt = cur.get("sent_datetime") or cur.get("received_datetime")
                    nxt_dt = nxt.get("received_datetime")
                    if pd.notna(cur_dt) and pd.notna(nxt_dt):
                        gap = (nxt_dt - cur_dt).total_seconds()
                        if 0 < gap <= 604800:
                            gap_times.append(gap)
        if len(gap_times) >= 5:
            their_response_time_to_me.append({
                "name": contact["name"],
                "email": contact["email"],
                "median_s": round(float(np.median(gap_times)), 0),
                "count": int(len(gap_times)),
            })

    my_response_time_to.sort(key=lambda x: x["median_s"])
    their_response_time_to_me.sort(key=lambda x: x["median_s"])

    # Response time by hour of day
    hourly_rt = []
    if "hour_of_day" in df.columns:
        hr_group = df[
            df["is_self"].fillna(False) &
            df["is_reply"].fillna(False) &
            df["response_time_seconds"].notna() &
            (df["response_time_seconds"] > 0) &
            (df["response_time_seconds"] <= 604800)
        ].groupby("hour_of_day")["response_time_seconds"]
        for hour in range(24):
            if hour in hr_group.groups:
                grp = hr_group.get_group(hour)
                hourly_rt.append({"hour": int(hour), "median_s": round(float(grp.median()), 0), "count": int(len(grp))})
            else:
                hourly_rt.append({"hour": int(hour), "median_s": None, "count": 0})

    # Monthly trend
    monthly_trend = []
    if "received_datetime" in df.columns:
        rt_df = df[
            df["is_self"].fillna(False) &
            df["is_reply"].fillna(False) &
            df["response_time_seconds"].notna() &
            (df["response_time_seconds"] > 0) &
            (df["response_time_seconds"] <= 604800)
        ].copy()
        rt_df["month"] = rt_df["received_datetime"].dt.to_period("M").astype(str)
        for month, grp in rt_df.groupby("month"):
            monthly_trend.append({
                "month": month,
                "median_s": round(float(grp["response_time_seconds"].median()), 0),
                "reply_count": int(len(grp)),
            })

    print(f"    Response patterns complete.")
    return {
        "overall": overall,
        "my_response_time_to": my_response_time_to,
        "their_response_time_to_me": their_response_time_to_me,
        "response_time_by_hour": hourly_rt,
        "monthly_trend": monthly_trend,
    }


# ── Analysis 3: Thread Analysis ───────────────────────────────────────────────

def build_thread_analysis(df: pd.DataFrame, user_email: str) -> dict:
    print("  Computing thread analysis...")
    ue = user_email.lower()

    threads = df.groupby("conversation_id")

    total_threads = df["conversation_id"].nunique()
    thread_lengths = threads["message_id"].transform("count") if "message_id" in df.columns else threads.size()

    # Thread-level aggregation
    tl_series = df.groupby("conversation_id")["thread_length"].first() if "thread_length" in df.columns else threads.size()
    single_msg = int((tl_series == 1).sum())
    avg_tl = round(float(tl_series.mean()), 1) if not tl_series.empty else 0.0
    median_tl = round(float(tl_series.median()), 1) if not tl_series.empty else 0.0
    max_tl = int(tl_series.max()) if not tl_series.empty else 0
    p90_tl = round(float(tl_series.quantile(0.90)), 1) if not tl_series.empty else 0.0

    stats = {
        "total_threads": total_threads,
        "single_message_threads": single_msg,
        "single_message_pct": round(100 * single_msg / total_threads, 1) if total_threads else 0.0,
        "avg_thread_length": avg_tl,
        "median_thread_length": median_tl,
        "max_thread_length": max_tl,
        "p90_thread_length": p90_tl,
        "threads_over_5": int((tl_series > 5).sum()),
        "threads_over_10": int((tl_series > 10).sum()),
        "threads_over_20": int((tl_series > 20).sum()),
    }

    # Longest threads top 10
    thread_meta = df.groupby("conversation_id").agg(
        subject=("subject_clean", "first"),
        message_count=("message_id", "count") if "message_id" in df.columns else ("conversation_id", "count"),
        first_message=("received_datetime", "min"),
        last_message=("received_datetime", "max"),
    ).reset_index()
    thread_meta["duration_days"] = (
        (thread_meta["last_message"] - thread_meta["first_message"]).dt.total_seconds() / 86400
    ).round(0).astype(int)

    # Participants per thread
    def _participant_names(grp):
        names = grp["sender_name"].fillna(grp["sender_email"]).dropna().unique()
        return list(names[:10])

    participant_info = df.groupby("conversation_id").apply(
        lambda g: pd.Series({
            "participant_count": g["sender_email"].nunique(),
            "participants": list(g["sender_name"].fillna(g["sender_email"]).dropna().unique()[:5]),
        })
    ).reset_index()

    thread_meta = thread_meta.merge(participant_info, on="conversation_id", how="left")
    top10 = thread_meta.nlargest(10, "message_count")

    longest_threads = []
    for _, row in top10.iterrows():
        longest_threads.append({
            "conversation_id": str(row["conversation_id"]),
            "subject": str(row["subject"]) if pd.notna(row["subject"]) else None,
            "message_count": int(row["message_count"]),
            "participant_count": int(row.get("participant_count", 0)),
            "participants": row.get("participants", []),
            "first_message": row["first_message"].isoformat() if pd.notna(row["first_message"]) else None,
            "last_message": row["last_message"].isoformat() if pd.notna(row["last_message"]) else None,
            "duration_days": int(row["duration_days"]),
        })

    # Thread participant distribution (3+ message threads)
    multi_threads = thread_meta[thread_meta["message_count"] >= 3]
    avg_participants = round(float(multi_threads["participant_count"].mean()), 1) if not multi_threads.empty else 0.0
    max_participants = int(multi_threads["participant_count"].max()) if not multi_threads.empty else 0

    participant_dist = {}
    if not multi_threads.empty:
        for n in [2, 3, 4]:
            participant_dist[str(n)] = int((multi_threads["participant_count"] == n).sum())
        participant_dist["5+"] = int((multi_threads["participant_count"] >= 5).sum())

    # Thread initiation
    starter_df = df[df["is_thread_starter"].fillna(False)]
    threads_started_by_user = int(starter_df[starter_df["is_self"].fillna(False)]["conversation_id"].nunique())
    threads_started_by_others = int(starter_df[~starter_df["is_self"].fillna(False)]["conversation_id"].nunique())
    total_started = threads_started_by_user + threads_started_by_others
    initiation_pct = round(100 * threads_started_by_user / total_started, 1) if total_started else 0.0

    # Thread reply rate
    # Threads started by others where user was a recipient
    others_started_ids = set(starter_df[~starter_df["is_self"].fillna(False)]["conversation_id"])
    user_replied_ids = set(df[df["is_self"].fillna(False) & ~df["is_thread_starter"].fillna(False)]["conversation_id"])

    threads_user_replied_to = len(others_started_ids & user_replied_ids)
    threads_without_email_reply = len(others_started_ids - user_replied_ids)
    total_reply_denominator = threads_user_replied_to + threads_without_email_reply
    reply_rate_pct = round(100 * threads_user_replied_to / total_reply_denominator, 1) if total_reply_denominator else 0.0

    print(f"    Thread analysis complete. {total_threads:,} threads.")
    return {
        "statistics": stats,
        "longest_threads": longest_threads,
        "participant_analysis": {
            "avg_participants_per_multi_thread": avg_participants,
            "max_participants": max_participants,
            "distribution": participant_dist,
        },
        "thread_initiation": {
            "threads_started_by_user": threads_started_by_user,
            "threads_started_by_others": threads_started_by_others,
            "initiation_pct": initiation_pct,
        },
        "thread_reply_rate": {
            "threads_user_replied_to": threads_user_replied_to,
            "threads_without_email_reply": threads_without_email_reply,
            "reply_rate_pct": reply_rate_pct,
        },
    }


# ── Analysis 4: Internal vs. External ────────────────────────────────────────

def build_internal_external(df: pd.DataFrame, user_email: str, user_domain: str) -> dict:
    print("  Computing internal/external breakdown...")
    ue = user_email.lower()

    total_human = len(df)

    # Classify each message as internal or external
    # For received: sender domain check
    # For sent: any recipient has different domain
    def _is_external_message(row):
        if row.get("is_self", False):
            # Sent: check all recipients
            to_list = _parse_recipients(row.get("to_recipients"))
            cc_list = _parse_recipients(row.get("cc_recipients"))
            for recip in to_list + cc_list:
                addr = (recip.get("email") or "").lower()
                if addr and not addr.endswith(f"@{user_domain}"):
                    return True
            return False
        else:
            sender = (row.get("sender_email") or "").lower()
            return not sender.endswith(f"@{user_domain}") if user_domain else False

    if "is_external" in df.columns:
        # Use existing flag for received; recompute for sent
        sent_mask = df["is_self"].fillna(False)
        received_mask = ~sent_mask

        sent_df = df[sent_mask].copy()
        recv_df = df[received_mask].copy()

        # Recompute external for sent messages
        sent_external = sent_df.apply(_is_external_message, axis=1)
        recv_external = recv_df["is_external"].fillna(False)

        int_sent = int((~sent_external).sum())
        ext_sent = int(sent_external.sum())
        int_recv = int((~recv_external).sum())
        ext_recv = int(recv_external.sum())
    else:
        sent_mask = df["is_self"].fillna(False)
        is_ext = df.apply(_is_external_message, axis=1)
        int_sent = int((sent_mask & ~is_ext).sum())
        ext_sent = int((sent_mask & is_ext).sum())
        int_recv = int((~sent_mask & ~is_ext).sum())
        ext_recv = int((~sent_mask & is_ext).sum())

    internal_count = int_sent + int_recv
    external_count = ext_sent + ext_recv

    volume_split = {
        "total_human_mail": total_human,
        "internal": {
            "count": internal_count,
            "pct": round(100 * internal_count / total_human, 1) if total_human else 0.0,
            "sent": int_sent,
            "received": int_recv,
        },
        "external": {
            "count": external_count,
            "pct": round(100 * external_count / total_human, 1) if total_human else 0.0,
            "sent": ext_sent,
            "received": ext_recv,
        },
    }

    # Top external domains (top 15)
    ext_df = df[df["is_external"].fillna(False)] if "is_external" in df.columns else pd.DataFrame()
    ext_domain_counts: dict[str, dict] = {}
    if not ext_df.empty:
        for _, row in ext_df.iterrows():
            is_sent = row.get("is_self", False)
            if is_sent:
                to_list = _parse_recipients(row.get("to_recipients"))
                cc_list = _parse_recipients(row.get("cc_recipients"))
                for recip in to_list + cc_list:
                    addr = (recip.get("email") or "").lower()
                    if addr and "@" in addr:
                        dom = addr.split("@", 1)[1]
                        if dom != user_domain:
                            if dom not in ext_domain_counts:
                                ext_domain_counts[dom] = {"received": 0, "sent": 0}
                            ext_domain_counts[dom]["sent"] += 1
            else:
                sender = (row.get("sender_email") or "").lower()
                if "@" in sender:
                    dom = sender.split("@", 1)[1]
                    if dom != user_domain:
                        if dom not in ext_domain_counts:
                            ext_domain_counts[dom] = {"received": 0, "sent": 0}
                        ext_domain_counts[dom]["received"] += 1

    top_ext_domains = []
    for dom, counts in sorted(
        ext_domain_counts.items(), key=lambda x: x[1]["sent"] + x[1]["received"], reverse=True
    )[:15]:
        total = counts["sent"] + counts["received"]
        recv_pct = counts["received"] / total if total else 0
        sent_pct = counts["sent"] / total if total else 0
        if recv_pct > 0.7:
            direction = "mostly_received"
        elif sent_pct > 0.7:
            direction = "mostly_sent"
        else:
            direction = "balanced"
        top_ext_domains.append({
            "domain": dom,
            "count": total,
            "sent": counts["sent"],
            "received": counts["received"],
            "direction": direction,
        })

    # Top external contacts (top 10)
    ext_contacts: dict[str, dict] = {}
    if not ext_df.empty:
        for _, row in ext_df.iterrows():
            is_sent = row.get("is_self", False)
            if is_sent:
                to_list = _parse_recipients(row.get("to_recipients"))
                cc_list = _parse_recipients(row.get("cc_recipients"))
                for recip in to_list + cc_list:
                    addr = (recip.get("email") or "").lower()
                    name = (recip.get("name") or "").strip()
                    if addr and "@" in addr:
                        dom = addr.split("@", 1)[1]
                        if dom != user_domain:
                            if addr not in ext_contacts:
                                ext_contacts[addr] = {"name": name, "email": addr, "domain": dom, "total_exchange": 0}
                            ext_contacts[addr]["total_exchange"] += 1
            else:
                sender = (row.get("sender_email") or "").lower()
                sender_name = (row.get("sender_name") or "").strip()
                if sender and "@" in sender:
                    dom = sender.split("@", 1)[1]
                    if dom != user_domain:
                        if sender not in ext_contacts:
                            ext_contacts[sender] = {"name": sender_name, "email": sender, "domain": dom, "total_exchange": 0}
                        ext_contacts[sender]["total_exchange"] += 1

    top_ext_contacts = sorted(ext_contacts.values(), key=lambda x: x["total_exchange"], reverse=True)[:10]

    # Internal vs external response time
    user_replies = df[
        df["is_self"].fillna(False) &
        df["is_reply"].fillna(False) &
        df["response_time_seconds"].notna() &
        (df["response_time_seconds"] > 0) &
        (df["response_time_seconds"] <= 604800)
    ].copy()

    rt_comparison = {}
    if "is_external" in df.columns and not user_replies.empty:
        # For user replies, check if the thread has external senders
        ext_thread_ids = set(df[df["is_external"].fillna(False)]["conversation_id"])
        int_rt = user_replies[~user_replies["conversation_id"].isin(ext_thread_ids)]["response_time_seconds"]
        ext_rt = user_replies[user_replies["conversation_id"].isin(ext_thread_ids)]["response_time_seconds"]
        rt_comparison = {
            "internal_median_s": round(float(int_rt.median()), 0) if not int_rt.empty else None,
            "external_median_s": round(float(ext_rt.median()), 0) if not ext_rt.empty else None,
            "internal_count": int(len(int_rt)),
            "external_count": int(len(ext_rt)),
        }

    # Monthly internal/external trend
    monthly_trend = []
    if "received_datetime" in df.columns and "is_external" in df.columns:
        trend_df = df.copy()
        trend_df["month"] = trend_df["received_datetime"].dt.to_period("M").astype(str)
        trend_df["is_ext_msg"] = trend_df["is_external"].fillna(False)
        monthly = trend_df.groupby("month").agg(
            internal=("is_ext_msg", lambda x: (~x).sum()),
            external=("is_ext_msg", "sum"),
        ).reset_index()
        for _, row in monthly.iterrows():
            monthly_trend.append({
                "month": row["month"],
                "internal": int(row["internal"]),
                "external": int(row["external"]),
            })

    print(f"    Internal/external complete.")
    return {
        "volume_split": volume_split,
        "top_external_domains": top_ext_domains,
        "top_external_contacts": top_ext_contacts,
        "response_time_comparison": rt_comparison,
        "monthly_trend": monthly_trend,
    }


# ── Analysis 5: Temporal Patterns ────────────────────────────────────────────

def build_temporal_patterns(df: pd.DataFrame) -> dict:
    print("  Computing temporal patterns...")

    sent_df = df[df["is_self"].fillna(False)]
    recv_df = df[~df["is_self"].fillna(False)]

    # Hourly distribution
    hourly_sent = [0] * 24
    hourly_recv = [0] * 24
    if "hour_of_day" in df.columns:
        for h, cnt in sent_df["hour_of_day"].dropna().value_counts().items():
            hourly_sent[int(h)] = int(cnt)
        for h, cnt in recv_df["hour_of_day"].dropna().value_counts().items():
            hourly_recv[int(h)] = int(cnt)

    # Day of week distribution
    dow_sent = [0] * 7
    dow_recv = [0] * 7
    if "day_of_week" in df.columns:
        for d, cnt in sent_df["day_of_week"].dropna().value_counts().items():
            dow_sent[int(d)] = int(cnt)
        for d, cnt in recv_df["day_of_week"].dropna().value_counts().items():
            dow_recv[int(d)] = int(cnt)

    # Heatmap (day × hour for sent)
    heatmap_sent = []
    if "day_of_week" in df.columns and "hour_of_day" in df.columns:
        hm = sent_df.groupby(["day_of_week", "hour_of_day"]).size().reset_index(name="count")
        hm_dict = {(int(r["day_of_week"]), int(r["hour_of_day"])): int(r["count"]) for _, r in hm.iterrows()}
        for day in range(7):
            for hour in range(24):
                heatmap_sent.append({"day": day, "hour": hour, "count": hm_dict.get((day, hour), 0)})

    # Peak day
    peak_day_info = {}
    if "received_datetime" in df.columns and not sent_df.empty:
        sent_df = sent_df.copy()
        sent_df["date"] = sent_df["received_datetime"].dt.date
        daily_counts = sent_df.groupby("date").size()
        if not daily_counts.empty:
            peak_date = daily_counts.idxmax()
            peak_count = int(daily_counts.max())
            peak_subjects = (
                sent_df[sent_df["date"] == peak_date]["subject_clean"]
                .dropna()
                .head(3)
                .tolist()
            ) if "subject_clean" in sent_df.columns else []
            peak_day_info = {
                "date": str(peak_date),
                "sent_count": peak_count,
                "top_subjects": peak_subjects,
            }

    # Monthly volume
    monthly_volume = []
    if "received_datetime" in df.columns:
        vol_df = df.copy()
        vol_df["month"] = vol_df["received_datetime"].dt.to_period("M").astype(str)
        monthly = vol_df.groupby("month").apply(
            lambda g: pd.Series({
                "sent": int(g["is_self"].fillna(False).sum()),
                "received": int((~g["is_self"].fillna(False)).sum()),
            })
        ).reset_index()
        for _, row in monthly.iterrows():
            monthly_volume.append({
                "month": row["month"],
                "sent": int(row["sent"]),
                "received": int(row["received"]),
            })

    # After hours (before 8am, after 6pm local time, or weekends).
    # Local time = LOCAL_TIMEZONE (IANA name, e.g. America/New_York) if set,
    # otherwise this computer's current UTC offset.
    after_hours_pct = 0.0
    if "received_datetime" in df.columns and not sent_df.empty:
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo
        tz_name = os.environ.get("LOCAL_TIMEZONE")
        local_tz = ZoneInfo(tz_name) if tz_name else datetime.now().astimezone().tzinfo
        local_dt = sent_df["received_datetime"].dt.tz_convert(local_tz)
        local_hour = local_dt.dt.hour
        local_dow = local_dt.dt.dayofweek
        after_hours_mask = (
            (local_dow >= 5) |       # weekend
            (local_hour < 8) |       # before 8am local
            (local_hour >= 18)       # after 6pm local
        )
        after_hours_pct = round(100 * after_hours_mask.sum() / len(sent_df), 1)

    # Busiest day of week
    busiest_dow = None
    if dow_sent:
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        busiest_dow_idx = int(np.argmax(dow_sent))
        busiest_dow = {"day": days[busiest_dow_idx], "sent_count": dow_sent[busiest_dow_idx]}

    # Peak send hour
    peak_send_hour = int(np.argmax(hourly_sent)) if any(hourly_sent) else None

    print(f"    Temporal patterns complete.")
    return {
        "hourly": {
            "sent": hourly_sent,
            "received": hourly_recv,
        },
        "day_of_week": {
            "sent": dow_sent,
            "received": dow_recv,
        },
        "heatmap_sent": heatmap_sent,
        "peak_day": peak_day_info,
        "monthly_volume": monthly_volume,
        "after_hours_pct": after_hours_pct,
        "busiest_day_of_week": busiest_dow,
        "peak_send_hour_utc": peak_send_hour,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Email-specific analysis")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output without prompting")
    args = parser.parse_args()

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not args.force:
        answer = input("outputs/25_email_analysis/ already has output. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== message-miner: 25_email_analysis ===\n")

    for path in (ENRICHED_PATH, FLAGGED_PATH, EXTRACTION_REPORT_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    extraction_report = json.loads(EXTRACTION_REPORT_PATH.read_text(encoding="utf-8"))
    user_email = extraction_report.get("user_email", "")
    user_name = extraction_report.get("user_name", "")
    user_domain = _user_domain(user_email)

    print(f"User: {user_name} <{user_email}>")
    print(f"Domain: {user_domain}\n")

    print("Loading enriched data...")
    enriched = pd.read_parquet(ENRICHED_PATH)

    print("Loading flagged data...")
    flagged = pd.read_parquet(FLAGGED_PATH)

    # Join: enriched has computed features; flagged has is_human_mail and other flags
    flag_cols = [c for c in flagged.columns if c not in enriched.columns or c == "message_id"]
    enriched = enriched.merge(flagged[flag_cols], on="message_id", how="left")

    total_loaded = len(enriched)
    human_mask = enriched["is_human_mail"].fillna(False)
    df = enriched[human_mask].copy()

    excluded = total_loaded - len(df)
    print(f"Total messages loaded: {total_loaded:,}")
    print(f"  Included (is_human_mail=True): {len(df):,}")
    print(f"  Excluded:                      {excluded:,}")
    print()

    # Ensure tz-aware datetimes
    for col in ("received_datetime", "sent_datetime"):
        if col in df.columns and df[col].dt.tz is None:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # ── Run analyses ──────────────────────────────────────────────────────────

    print("[1/5] Network graph...")
    network_graph = build_network_graph(df, user_email, user_domain)
    (OUTPUT_DIR / "network_graph.json").write_text(
        json.dumps(network_graph, indent=2, default=str), encoding="utf-8"
    )
    print(f"      -> network_graph.json written.\n")

    top_contacts = network_graph.get("top_20_contacts", [])

    print("[2/5] Response patterns...")
    response_patterns = build_response_patterns(df, top_contacts)
    (OUTPUT_DIR / "response_patterns.json").write_text(
        json.dumps(response_patterns, indent=2, default=str), encoding="utf-8"
    )
    print(f"      -> response_patterns.json written.\n")

    print("[3/5] Thread analysis...")
    thread_analysis = build_thread_analysis(df, user_email)
    (OUTPUT_DIR / "thread_analysis.json").write_text(
        json.dumps(thread_analysis, indent=2, default=str), encoding="utf-8"
    )
    print(f"      -> thread_analysis.json written.\n")

    print("[4/5] Internal/external breakdown...")
    internal_external = build_internal_external(df, user_email, user_domain)
    (OUTPUT_DIR / "internal_external.json").write_text(
        json.dumps(internal_external, indent=2, default=str), encoding="utf-8"
    )
    print(f"      -> internal_external.json written.\n")

    print("[5/5] Temporal patterns...")
    temporal_patterns = build_temporal_patterns(df)
    (OUTPUT_DIR / "temporal_patterns.json").write_text(
        json.dumps(temporal_patterns, indent=2, default=str), encoding="utf-8"
    )
    print(f"      -> temporal_patterns.json written.\n")

    # ── Summary report ────────────────────────────────────────────────────────
    print("Writing summary report...")

    overall_rt = response_patterns.get("overall", {})
    median_rt_s = overall_rt.get("median_response_time_s")
    top_contact = top_contacts[0] if top_contacts else {}
    top_ext_domains = internal_external.get("top_external_domains", [])
    longest_threads = thread_analysis.get("longest_threads", [])
    thread_stats = thread_analysis.get("statistics", {})
    thread_init = thread_analysis.get("thread_initiation", {})
    reply_rate = thread_analysis.get("thread_reply_rate", {})
    vol_split = internal_external.get("volume_split", {})
    temporal = temporal_patterns

    # Date range
    date_first = None
    date_last = None
    if "received_datetime" in df.columns:
        valid_dt = df["received_datetime"].dropna()
        if not valid_dt.empty:
            date_first = valid_dt.min().date().isoformat()
            date_last = valid_dt.max().date().isoformat()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_human_mail_analyzed": len(df),
        "date_range": f"{date_first} to {date_last}" if date_first else "unknown",
        "highlights": {
            "total_contacts": network_graph.get("total_contacts", 0),
            "top_contact": (
                f"{top_contact.get('name')} ({top_contact.get('total_exchange')} emails)"
                if top_contact else "N/A"
            ),
            "median_response_time": _fmt_duration(median_rt_s),
            "internal_pct": vol_split.get("internal", {}).get("pct", 0),
            "top_external_domain": top_ext_domains[0]["domain"] if top_ext_domains else "N/A",
            "longest_thread": (
                f"{longest_threads[0]['message_count']} messages — {longest_threads[0]['subject']}"
                if longest_threads else "N/A"
            ),
            "thread_initiation_pct": thread_init.get("initiation_pct", 0),
            "peak_send_hour_utc": temporal.get("peak_send_hour_utc"),
            "after_hours_pct": temporal.get("after_hours_pct", 0),
            "reply_rate_pct": reply_rate.get("reply_rate_pct", 0),
        },
    }
    (OUTPUT_DIR / "email_analysis_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print("\n=== Summary ===")
    h = report["highlights"]
    print(f"  Human mail analyzed:     {len(df):,}")
    print(f"  Date range:              {report['date_range']}")
    print(f"  Total contacts:          {h['total_contacts']:,}")
    print(f"  Top contact:             {h['top_contact']}")
    print(f"  Median response time:    {h['median_response_time']}")
    print(f"  Internal mail:           {h['internal_pct']}%")
    print(f"  Top external domain:     {h['top_external_domain']}")
    print(f"  Longest thread:          {h['longest_thread']}")
    print(f"  Thread initiation:       {h['thread_initiation_pct']}%")
    print(f"  Peak send hour (UTC):    {h['peak_send_hour_utc']:02d}:00" if h['peak_send_hour_utc'] is not None else "  Peak send hour (UTC):    N/A")
    print(f"  After-hours sent:        {h['after_hours_pct']}%")
    print(f"  Thread reply rate:       {h['reply_rate_pct']}%")
    print(f"\nAll outputs written to {OUTPUT_DIR}/")
    print("  network_graph.json")
    print("  response_patterns.json")
    print("  thread_analysis.json")
    print("  internal_external.json")
    print("  temporal_patterns.json")
    print("  email_analysis_report.json")


if __name__ == "__main__":
    main()
