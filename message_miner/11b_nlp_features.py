"""
11b_nlp_features.py — Classic NLP Feature Extraction

Zero API cost. Pure local computation using spaCy, scikit-learn, and pandas.

Reads:
  outputs/05_enrich/messages_enriched.parquet
  outputs/09_emoji/emoji_messages.parquet
  outputs/01_connect/connection_report.json

Writes:
  outputs/11b_nlp_features/
  ├-- sender_profiles.parquet          one row per sender: all computed features
  ├-- distinctive_words.parquet        per-sender distinctive vocabulary
  ├-- entities.parquet                 per-message named entities
  ├-- entity_summary.parquet           aggregate entity stats
  ├-- questions.parquet                messages identified as questions
  └-- nlp_report.json

Run from project root:
    python message_miner/11b_nlp_features.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

# -- Paths ----------------------------------------------------------------------

MESSAGES_PATH   = Path("outputs/05_enrich/messages_enriched.parquet")
EMOJI_PATH      = Path("outputs/09_emoji/emoji_messages.parquet")
CONNECTION_PATH = Path("outputs/01_connect/connection_report.json")
OUTPUT_DIR      = Path("outputs/11b_nlp_features")
NER_CACHE_PATH  = OUTPUT_DIR / "cache" / "ner_pass.json"

# -- Constants ------------------------------------------------------------------

MIN_SENDER_MESSAGES       = 50
MIN_WORDS_FOR_NER         = 5
MIN_WORDS_FOR_READABILITY = 10
NER_BATCH_SIZE            = 256
KEEP_ENTITY_LABELS        = frozenset({"ORG", "PRODUCT", "GPE", "PERSON", "WORK_OF_ART"})
TFIDF_MAX_FEATURES        = 10_000
TFIDF_TOP_N               = 20

_VOWEL_RE      = re.compile(r"[aeiouAEIOU]+")
_TOKEN_RE      = re.compile(r"\b[a-z]+\b")
_ENC_ID_RE     = re.compile(r"^[a-z0-9]{15,}$", re.IGNORECASE)

# Communication-style regex patterns (applied via Series.str.contains)
_GREETING_PAT    = r"(?i)^\s*(hi\s|hey\s|hello|good morning|good afternoon|good evening|morning\s)"
_HEDGE_PAT       = r"(?i)(i think|maybe|not sure|probably|might be|could be|i guess|possibly|it seems)"
_DIRECTIVE_PAT   = r"(?i)(please\s|can you|could you|need you|make sure|don't forget|let me know|lmk)"
_CONTRACTION_PAT = r"(?i)(n't|'m|'re|'ve|'ll|'d|i'm|we're|don't|can't|won't|it's)"


# -- spaCy ----------------------------------------------------------------------

def _load_spacy():
    """Prefer en_core_web_trf, fall back to en_core_web_sm."""
    try:
        import spacy
    except ImportError:
        print("ERROR: spaCy not installed. Run: pip install spacy")
        sys.exit(1)

    for model_name in ("en_core_web_trf", "en_core_web_sm"):
        try:
            nlp = spacy.load(model_name)
            print(f"  spaCy model: {model_name}")
            return nlp, model_name
        except Exception:
            continue

    print("ERROR: No spaCy model found.")
    print("  Run: python -m spacy download en_core_web_sm")
    sys.exit(1)


# -- Helpers --------------------------------------------------------------------

def _syllables(word: str) -> int:
    """Count vowel groups per word, minimum 1."""
    return max(len(_VOWEL_RE.findall(word)), 1)


def _style_pcts(texts: pd.Series) -> dict:
    """Return communication-style feature percentages for a Series of texts."""
    n = len(texts)
    if n == 0:
        return {k: 0.0 for k in [
            "question_pct", "exclamation_pct", "greeting_pct",
            "hedge_pct", "directive_pct", "contraction_pct",
        ]}
    s = texts.fillna("").astype(str)
    return {
        "question_pct":     float(s.str.contains(r"\?", regex=True, na=False).sum() / n * 100),
        "exclamation_pct":  float(s.str.contains("!", regex=False, na=False).sum() / n * 100),
        "greeting_pct":     float(s.str.contains(_GREETING_PAT, na=False).sum() / n * 100),
        "hedge_pct":        float(s.str.contains(_HEDGE_PAT, na=False).sum() / n * 100),
        "directive_pct":    float(s.str.contains(_DIRECTIVE_PAT, na=False).sum() / n * 100),
        "contraction_pct":  float(s.str.contains(_CONTRACTION_PAT, na=False).sum() / n * 100),
    }


# -- Main -----------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="NLP feature extraction")
    parser.add_argument("--force", action="store_true", help="Re-run spaCy NER pass, ignoring cache")
    args = parser.parse_args()

    print("=== message-miner: 11b_nlp_features ===\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    NER_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for path in (MESSAGES_PATH, EMOJI_PATH, CONNECTION_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run earlier pipeline steps first.")
            sys.exit(1)

    # -- Load inputs -------------------------------------------------------------
    print("Loading data...")
    messages   = pd.read_parquet(MESSAGES_PATH)
    emoji_msgs = pd.read_parquet(EMOJI_PATH)
    connection = json.loads(CONNECTION_PATH.read_text(encoding="utf-8"))
    print(f"  {len(messages):,} total messages")
    print(f"  {len(emoji_msgs):,} emoji-message rows")

    # -- Core filter -------------------------------------------------------------
    df = messages[
        (messages["message_type"] == "message") &
        (messages["is_deleted"] == False)
    ].copy()
    df["text"] = df["text"].fillna("").astype(str)
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0)
    print(f"  {len(df):,} messages after filter (type=message, not deleted)\n")

    # -- Qualified senders (>= 50 messages) --------------------------------------
    sender_msg_counts = df.groupby("sender_id").size()
    qualified_ids     = set(sender_msg_counts[sender_msg_counts >= MIN_SENDER_MESSAGES].index)
    senders_below     = len(sender_msg_counts) - len(qualified_ids)
    df_q = df[df["sender_id"].isin(qualified_ids)].copy()

    print(f"  Senders with >= {MIN_SENDER_MESSAGES} messages: {len(qualified_ids):,}")
    print(f"  Senders below threshold:              {senders_below:,}\n")

    # ========================================================================
    # STEP 1: TF-IDF distinctive words
    # ========================================================================
    print("Computing TF-IDF distinctive words...")

    sender_docs: dict[str, str]  = {}
    sender_meta: dict[str, dict] = {}

    for sender_id, grp in df_q.groupby("sender_id"):
        sender_docs[sender_id] = " ".join(grp["text"].tolist())
        row0 = grp.iloc[0]
        sender_meta[sender_id] = {
            "sender_name": str(row0.get("sender_name") or ""),
            "is_self":     bool(row0.get("is_self", False)),
        }

    sender_ids_ordered = list(sender_docs.keys())
    corpus             = [sender_docs[sid] for sid in sender_ids_ordered]

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=2,
        stop_words="english",
        token_pattern=r"\b[a-zA-Z]{3,}\b",
    )
    tfidf_matrix  = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    distinctive_rows = []
    for i, sender_id in enumerate(sender_ids_ordered):
        row_vec    = tfidf_matrix[i].toarray().flatten()
        top_idxs   = row_vec.argsort()[-TFIDF_TOP_N:][::-1]
        meta       = sender_meta[sender_id]
        for rank, idx in enumerate(top_idxs, 1):
            if row_vec[idx] > 0:
                distinctive_rows.append({
                    "sender_name": meta["sender_name"],
                    "is_self":     meta["is_self"],
                    "word":        feature_names[idx],
                    "tfidf_score": float(row_vec[idx]),
                    "rank":        rank,
                })

    distinctive_df         = pd.DataFrame(distinctive_rows)
    distinctive_df["rank"] = distinctive_df["rank"].astype("int32")
    print(f"  {len(distinctive_rows):,} entries for {len(sender_ids_ordered):,} senders\n")

    # ========================================================================
    # STEP 2: spaCy NER + readability (single pipe pass, with cache)
    # ========================================================================
    ner_df = df[df["word_count"] >= MIN_WORDS_FOR_NER].copy()

    if NER_CACHE_PATH.exists() and not args.force:
        print(f"spaCy NER cache found — loading from {NER_CACHE_PATH}")
        cached = json.loads(NER_CACHE_PATH.read_text(encoding="utf-8"))
        entity_rows = cached["entity_rows"]
        read_acc    = defaultdict(
            lambda: {"n_sents": 0, "n_alpha_words": 0, "n_syllables": 0, "n_char": 0},
            {k: v for k, v in cached["read_acc"].items()},
        )
        spacy_model_name = cached.get("spacy_model", "unknown")
        print(f"  {len(entity_rows):,} entity rows, {len(read_acc):,} sender accumulators\n")
    else:
        if args.force and NER_CACHE_PATH.exists():
            NER_CACHE_PATH.unlink()

        # -- spaCy model -------------------------------------------------------
        nlp, spacy_model_name = _load_spacy()
        print()

        print(f"spaCy NER pass on {len(ner_df):,} messages (word_count >= {MIN_WORDS_FOR_NER})...")

        # Per-sender readability accumulators keyed by sender_id
        read_acc: dict[str, dict] = defaultdict(lambda: {
            "n_sents": 0, "n_alpha_words": 0,
            "n_syllables": 0, "n_char": 0,
        })

        entity_rows  = []
        meta_records = ner_df[[
            "message_id", "chat_id", "sender_id", "sender_name", "is_self",
            "timestamp", "word_count",
        ]].to_dict("records")
        texts = ner_df["text"].tolist()

        for doc, row in tqdm(
            zip(nlp.pipe(texts, batch_size=NER_BATCH_SIZE), meta_records),
            total=len(meta_records),
            desc="  NER",
        ):
            sender_id   = row["sender_id"]
            sender_name = row["sender_name"] or "Unknown"

            # -- Named entities ----------------------------------------------------
            for ent in doc.ents:
                if ent.label_ not in KEEP_ENTITY_LABELS:
                    continue
                text = ent.text.strip()
                if len(text) <= 1:
                    continue
                if re.fullmatch(r"\d+", text):
                    continue
                if _ENC_ID_RE.match(text):
                    continue
                entity_rows.append({
                    "message_id":   row["message_id"],
                    "chat_id":      row["chat_id"],
                    "sender_name":  sender_name,
                    "is_self":      row["is_self"],
                    "entity_text":  text,
                    "entity_label": ent.label_,
                    "timestamp":    str(row["timestamp"]),
                })

            # -- Readability (word_count >= 10 only) ------------------------------
            if row["word_count"] >= MIN_WORDS_FOR_READABILITY:
                sents = list(doc.sents)
                if not sents:
                    continue
                acc = read_acc[sender_id]
                acc["n_sents"] += len(sents)
                for token in doc:
                    if token.is_alpha:
                        acc["n_alpha_words"] += 1
                        acc["n_syllables"]   += _syllables(token.text)
                        acc["n_char"]        += len(token.text)

        print(f"  Extracted {len(entity_rows):,} raw entity mentions")
        print(f"  Saving NER cache to {NER_CACHE_PATH}...")
        NER_CACHE_PATH.write_text(
            json.dumps({
                "spacy_model": spacy_model_name,
                "entity_rows": entity_rows,
                "read_acc":    dict(read_acc),
            }, default=str),
            encoding="utf-8",
        )
        print()

    # ========================================================================
    # STEP 3: Vocabulary per sender (Counter-based)
    # ========================================================================
    print("Computing vocabulary features...")
    vocab_stats: dict[str, dict] = {}
    for sender_id, grp in tqdm(
        df_q.groupby("sender_id"),
        desc="  Vocabulary",
        total=len(qualified_ids),
    ):
        all_tokens = []
        for text in grp["text"]:
            all_tokens.extend(_TOKEN_RE.findall(text.lower()))
        filtered = [t for t in all_tokens if t not in ENGLISH_STOP_WORDS]
        if not filtered:
            vocab_stats[sender_id] = {"unique_words": 0, "hapax_pct": 0.0}
            continue
        freq   = Counter(filtered)
        unique = len(freq)
        hapax  = sum(1 for c in freq.values() if c == 1)
        vocab_stats[sender_id] = {
            "unique_words": unique,
            "hapax_pct":    float(hapax / unique * 100),
        }
    print()

    # ========================================================================
    # STEP 4: Emoji features per sender (keyed by sender_name)
    # ========================================================================
    emoji_total_by_name: dict[str, int]   = {}
    emoji_only_by_name:  dict[str, float] = {}
    top5_by_name:        dict[str, str]   = {}

    for sender_name, grp in emoji_msgs.groupby("sender_name"):
        n = len(grp)
        emoji_total_by_name[sender_name] = int(grp["emoji_count"].sum())
        emoji_only_by_name[sender_name]  = float(grp["is_emoji_only"].sum() / n * 100) if n > 0 else 0.0
        all_emojis: list[str] = []
        for lst_str in grp.loc[grp["emoji_count"] > 0, "emoji_list"]:
            try:
                all_emojis.extend(json.loads(lst_str))
            except Exception:
                pass
        top5 = [e for e, _ in Counter(all_emojis).most_common(5)]
        top5_by_name[sender_name] = json.dumps(top5)

    # ========================================================================
    # STEP 5: Build sender_profiles
    # ========================================================================
    print("Building sender profiles...")
    profile_rows = []

    for sender_id, grp in tqdm(
        df_q.groupby("sender_id"),
        desc="  Profiles",
        total=len(qualified_ids),
    ):
        row0        = grp.iloc[0]
        sender_name = str(row0.get("sender_name") or "")
        is_self     = bool(row0.get("is_self", False))
        wc          = grp["word_count"]

        # -- Volume & length ---------------------------------------------------
        total_msgs  = len(grp)
        total_words = int(wc.sum())
        avg_words   = float(wc.mean())
        med_words   = float(wc.median())
        p95_words   = float(wc.quantile(0.95))
        sw_pct      = float((wc == 1).sum() / total_msgs * 100)

        # -- Vocabulary --------------------------------------------------------
        vs           = vocab_stats.get(sender_id, {"unique_words": 0, "hapax_pct": 0.0})
        unique_words = vs["unique_words"]
        denom        = min(total_words, 5000)
        ttr          = float(unique_words / denom) if denom > 0 else 0.0
        hapax_pct    = float(vs["hapax_pct"])

        # -- Readability -------------------------------------------------------
        acc = read_acc.get(sender_id)
        if acc and acc["n_sents"] > 0 and acc["n_alpha_words"] > 0:
            fk_grade     = float(
                0.39 * (acc["n_alpha_words"] / acc["n_sents"]) +
                11.8 * (acc["n_syllables"]   / acc["n_alpha_words"]) -
                15.59
            )
            avg_sent_len = float(acc["n_alpha_words"] / acc["n_sents"])
            avg_word_len = float(acc["n_char"]        / acc["n_alpha_words"])
        else:
            fk_grade = avg_sent_len = avg_word_len = float("nan")

        # -- Temporal ----------------------------------------------------------
        ts = pd.to_datetime(grp["timestamp"], utc=True, errors="coerce")
        if ts.notna().any():
            peak_hour   = int(ts.dt.hour.value_counts().idxmax())
            weekend_pct = float((ts.dt.dayofweek >= 5).sum() / total_msgs * 100)
        else:
            peak_hour   = -1
            weekend_pct = 0.0

        rt = grp["response_time_seconds"].dropna() if "response_time_seconds" in grp.columns else pd.Series(dtype=float)
        avg_rt = float(rt.mean())    if len(rt) > 0 else float("nan")
        med_rt = float(rt.median())  if len(rt) > 0 else float("nan")

        # -- Communication style -----------------------------------------------
        style = _style_pcts(grp["text"])

        # -- Emoji -------------------------------------------------------------
        total_emoji    = emoji_total_by_name.get(sender_name, 0)
        emoji_per_100  = float(total_emoji / total_words * 100) if total_words > 0 else 0.0
        emoji_only_pct = float(emoji_only_by_name.get(sender_name, 0.0))
        top5_emoji     = top5_by_name.get(sender_name, "[]")

        profile_rows.append({
            "sender_id":                sender_id,
            "sender_name":              sender_name,
            "is_self":                  is_self,
            "total_messages":           total_msgs,
            "total_words":              total_words,
            "avg_words_per_message":    avg_words,
            "median_words_per_message": med_words,
            "p95_words_per_message":    p95_words,
            "single_word_pct":          sw_pct,
            "unique_words":             unique_words,
            "type_token_ratio":         ttr,
            "hapax_pct":                hapax_pct,
            "flesch_kincaid_grade":     fk_grade,
            "avg_sentence_length":      avg_sent_len,
            "avg_word_length":          avg_word_len,
            "avg_response_time_s":      avg_rt,
            "median_response_time_s":   med_rt,
            "peak_hour_utc":            peak_hour,
            "weekend_pct":              weekend_pct,
            **style,
            "emoji_per_100_words":      emoji_per_100,
            "emoji_only_pct":           emoji_only_pct,
            "top_5_emoji":              top5_emoji,
        })

    profiles_df = pd.DataFrame(profile_rows)

    # -- Cast dtypes -----------------------------------------------------------
    for col in ("total_messages", "total_words", "unique_words"):
        profiles_df[col] = profiles_df[col].astype("int32")
    profiles_df["peak_hour_utc"] = profiles_df["peak_hour_utc"].astype("int8")

    float32_cols = [
        "avg_words_per_message", "median_words_per_message", "p95_words_per_message",
        "single_word_pct", "type_token_ratio", "hapax_pct",
        "flesch_kincaid_grade", "avg_sentence_length", "avg_word_length",
        "weekend_pct", "question_pct", "exclamation_pct", "greeting_pct",
        "hedge_pct", "directive_pct", "contraction_pct",
        "emoji_per_100_words", "emoji_only_pct",
    ]
    for col in float32_cols:
        profiles_df[col] = profiles_df[col].astype("float32")

    print()

    # ========================================================================
    # STEP 6: entities.parquet + entity_summary.parquet
    # ========================================================================
    print("Building entity tables...")

    entities_df = pd.DataFrame(entity_rows)
    if not entities_df.empty:
        entities_df["timestamp"] = pd.to_datetime(entities_df["timestamp"], utc=True, errors="coerce")

    if not entities_df.empty:
        lower_text = entities_df["entity_text"].str.lower().str.strip()

        def _agg_entity(grp: pd.DataFrame) -> pd.Series:
            return pd.Series({
                "entity_label":   grp["entity_label"].value_counts().idxmax(),
                "mention_count":  len(grp),
                "unique_senders": int(grp["sender_name"].nunique()),
                "self_mentions":  int(grp["is_self"].sum()),
                "unique_chats":   int(grp["chat_id"].nunique()),
            })

        entity_summary = (
            entities_df.assign(entity_key=lower_text)
            .groupby("entity_key", sort=False)
            .apply(_agg_entity)
            .reset_index()
            .rename(columns={"entity_key": "entity_text"})
        )
        for col in ("mention_count", "unique_senders", "self_mentions", "unique_chats"):
            entity_summary[col] = entity_summary[col].astype("int32")
    else:
        entity_summary = pd.DataFrame(columns=[
            "entity_text", "entity_label", "mention_count",
            "unique_senders", "self_mentions", "unique_chats",
        ])

    print(f"  {len(entities_df):,} entity rows, {len(entity_summary):,} unique entities\n")

    # ========================================================================
    # STEP 7: questions.parquet
    # ========================================================================
    print("Detecting questions...")

    questions_raw = df[df["text"].str.strip().str.endswith("?")].copy()

    # Attach next-message info (same chat, next by timestamp)
    df_sorted = (
        df.sort_values(["chat_id", "timestamp"])
        .reset_index(drop=True)
    )
    df_sorted["_next_sender_id"] = df_sorted.groupby("chat_id")["sender_id"].shift(-1)
    df_sorted["_next_timestamp"] = df_sorted.groupby("chat_id")["timestamp"].shift(-1)

    questions_raw = questions_raw.merge(
        df_sorted[["message_id", "_next_sender_id", "_next_timestamp"]],
        on="message_id",
        how="left",
    )

    ts_q    = pd.to_datetime(questions_raw["timestamp"],      utc=True, errors="coerce")
    ts_next = pd.to_datetime(questions_raw["_next_timestamp"], utc=True, errors="coerce")
    delta   = ts_next - ts_q

    answered = (
        questions_raw["_next_sender_id"].notna() &
        (questions_raw["_next_sender_id"] != questions_raw["sender_id"]) &
        (delta <= pd.Timedelta("24h"))
    )

    questions_df = pd.DataFrame({
        "message_id":          questions_raw["message_id"].values,
        "chat_id":             questions_raw["chat_id"].values,
        "sender_name":         questions_raw["sender_name"].values,
        "is_self":             questions_raw["is_self"].values,
        "timestamp":           ts_q.values,
        "text":                questions_raw["text"].values,
        "was_answered":        answered.values,
        "answer_time_seconds": delta.dt.total_seconds().where(answered).values,
    })

    answered_count = int(questions_df["was_answered"].sum())
    q_total        = len(questions_df)
    q_answer_rate  = round(answered_count / q_total * 100, 1) if q_total > 0 else 0.0
    print(f"  {q_total:,} questions detected  ({q_answer_rate:.1f}% answered)\n")

    # ========================================================================
    # STEP 8: Write outputs
    # ========================================================================
    print("Writing outputs...")

    profiles_df.to_parquet(OUTPUT_DIR / "sender_profiles.parquet",       index=False)
    distinctive_df.to_parquet(OUTPUT_DIR / "distinctive_words.parquet",  index=False)
    entities_df.to_parquet(OUTPUT_DIR / "entities.parquet",              index=False)
    entity_summary.to_parquet(OUTPUT_DIR / "entity_summary.parquet",     index=False)
    questions_df.to_parquet(OUTPUT_DIR / "questions.parquet",            index=False)

    print(f"  sender_profiles.parquet      ({len(profiles_df):,} senders)")
    print(f"  distinctive_words.parquet    ({len(distinctive_df):,} rows)")
    print(f"  entities.parquet             ({len(entities_df):,} rows)")
    print(f"  entity_summary.parquet       ({len(entity_summary):,} unique entities)")
    print(f"  questions.parquet            ({len(questions_df):,} rows)")

    # ========================================================================
    # STEP 9: nlp_report.json
    # ========================================================================
    numeric_avg_cols = [
        "avg_words_per_message", "flesch_kincaid_grade", "type_token_ratio",
        "question_pct", "exclamation_pct", "greeting_pct", "hedge_pct",
        "directive_pct", "contraction_pct", "hapax_pct", "single_word_pct",
        "avg_sentence_length", "avg_word_length", "avg_response_time_s",
        "median_response_time_s", "weekend_pct", "emoji_per_100_words",
        "emoji_only_pct",
    ]
    corpus_averages = {}
    for col in numeric_avg_cols:
        if col in profiles_df.columns:
            val = float(profiles_df[col].astype(float).mean(skipna=True))
            corpus_averages[col] = round(val, 4) if not np.isnan(val) else None

    report = {
        "generated_at":             datetime.now(timezone.utc).isoformat(),
        "spacy_model":              spacy_model_name,
        "senders_profiled":         len(profiles_df),
        "senders_below_threshold":  senders_below,
        "corpus_averages":          corpus_averages,
        "total_entities_extracted": len(entities_df),
        "unique_entities":          len(entity_summary),
        "total_questions_detected": q_total,
        "question_answer_rate":     q_answer_rate,
    }
    (OUTPUT_DIR / "nlp_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("  nlp_report.json\n")

    # ========================================================================
    # STEP 10: Console summary table
    # ========================================================================
    print("-" * 54)
    print("  11b_nlp_features — complete")
    print("-" * 54)
    print(f"  Senders profiled:            {len(profiles_df):>8,}")
    print(f"  Senders below threshold:     {senders_below:>8,}")
    print(f"  Entity mentions extracted:   {len(entities_df):>8,}")
    print(f"  Unique entities:             {len(entity_summary):>8,}")
    print(f"  Questions detected:          {q_total:>8,}")
    print(f"  Question answer rate:        {q_answer_rate:>7.1f}%")
    print(f"  spaCy model:                 {spacy_model_name}")
    print()

    if len(profiles_df) > 0:
        top10 = profiles_df.nlargest(10, "total_messages")[
            ["sender_name", "total_messages", "avg_words_per_message",
             "type_token_ratio", "flesch_kincaid_grade"]
        ]
        hdr = f"  {'Sender':<30} {'Msgs':>7} {'Avg Wds':>8} {'TTR':>6} {'FK Grd':>7}"
        print(hdr)
        print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*6} {'-'*7}")
        for _, r in top10.iterrows():
            fk_val = r["flesch_kincaid_grade"]
            fk_str = f"{fk_val:>7.1f}" if not np.isnan(float(fk_val)) else "      —"
            print(
                f"  {str(r['sender_name'])[:30]:<30} "
                f"{int(r['total_messages']):>7,} "
                f"{float(r['avg_words_per_message']):>8.1f} "
                f"{float(r['type_token_ratio']):>6.3f} "
                f"{fk_str}"
            )

    print(f"\nOutputs written to {OUTPUT_DIR}/")
    print("Next: python message_miner/11c_classify.py")


if __name__ == "__main__":
    main()
