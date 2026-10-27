"""
label_boundaries.py — Interactive tool for building a golden set of
human-labeled conversation boundaries.

Recommended workflow (point-and-click):
  1. web         — generate a single HTML file containing all labelable chats
  2. [you click] — open the HTML in a browser, click messages to toggle boundaries
  3. [export]    — click "Download golden_set.json" in the page; drop it into
                   outputs/07_segment/golden_set/golden_set.json

Legacy text-file workflow:
  1. list        — show chats available for labeling
  2. export      — write a chat's messages to a .txt file for review
  3. [you edit]  — add `>>>` at the start of any line that begins a new conversation
  4. import      — parse edited files and merge into golden_set.json

The resulting golden_set.json can be passed to validation.py via --golden.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

MESSAGES_PATH = Path("outputs/05_enrich/messages_enriched.parquet")
CHATS_PATH = Path("outputs/02_extract/chats.parquet")
BURSTS_PATH = Path("outputs/07_segment/bursts.parquet")
BURST_MAP_PATH = Path("outputs/07_segment/message_burst_map.parquet")
FLAGGED_PATH = Path("outputs/07_segment/flagged_bursts.json")
GOLDEN_DIR = Path("outputs/07_segment/golden_set")
GOLDEN_JSON = GOLDEN_DIR / "golden_set.json"

BOUNDARY_MARKER = ">>>"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_name(display_name: str | None) -> str:
    if not display_name:
        return "Unknown"
    return display_name.split()[0]


def _format_gap(seconds) -> str:
    if pd.isna(seconds) or seconds <= 0:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def _load_messages() -> pd.DataFrame:
    for path in (MESSAGES_PATH, CHATS_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found. Run the pipeline through 05_enrich first.")
            sys.exit(1)

    df = pd.read_parquet(MESSAGES_PATH)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    mask = (
        (df["message_type"] == "message")
        & (df["is_deleted"] == False)
        & (df["is_empty"] == False)
    )
    df = df[mask].sort_values(["chat_id", "timestamp"]).reset_index(drop=True)
    return df


def _load_chats() -> pd.DataFrame:
    return pd.read_parquet(CHATS_PATH)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(args) -> None:
    """Show chats available for labeling, sorted by suitability."""
    df = _load_messages()
    chats_df = _load_chats()

    counts = df.groupby("chat_id").size().rename("msg_count")
    chat_type_map = chats_df.set_index("chat_id")["chat_type"].astype(str).to_dict()

    # Prefer chats with 30–300 messages — enough variety, not overwhelming
    summary = counts.reset_index()
    summary["chat_type"] = summary["chat_id"].map(chat_type_map).fillna("unknown")
    summary["suitable"] = (summary["msg_count"] >= 30) & (summary["msg_count"] <= 300)
    summary = summary.sort_values(["suitable", "msg_count"], ascending=[False, True])

    labeled = set()
    if GOLDEN_JSON.exists():
        labeled = set(json.loads(GOLDEN_JSON.read_text(encoding="utf-8")).keys())

    print(f"Chats available for labeling (* = suitable: 30-300 msgs, X = already labeled):\n")
    print(f"  {'*':1}  {'X':1}  {'type':<15}  {'msgs':>6}  chat_id")
    print(f"  {'-':1}  {'-':1}  {'-' * 15}  {'-' * 6}  {'-' * 40}")
    for _, row in summary.head(args.limit).iterrows():
        star = "*" if row["suitable"] else " "
        done = "X" if row["chat_id"] in labeled else " "
        print(f"  {star}  {done}  {row['chat_type']:<15}  {row['msg_count']:>6}  {row['chat_id']}")

    print(f"\nSuggestion: pick 2-3 * unlabeled chats totaling ~50-100 boundaries worth of material.")
    print(f"Next: python message_miner/label_boundaries.py export --chat CHAT_ID")


def cmd_export(args) -> None:
    """Write a chat's messages to a text file for manual boundary labeling."""
    df = _load_messages()
    chats_df = _load_chats()

    chat_id = args.chat
    chat_msgs = df[df["chat_id"] == chat_id]
    if chat_msgs.empty:
        print(f"ERROR: No messages found for chat_id={chat_id}")
        sys.exit(1)

    chat_type_map = chats_df.set_index("chat_id")["chat_type"].astype(str).to_dict()
    chat_type = chat_type_map.get(chat_id, "unknown")

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = chat_id.replace(":", "_").replace("/", "_").replace("\\", "_")
    output_path = Path(args.output) if args.output else GOLDEN_DIR / f"{safe_id}.txt"

    lines = [
        f"# chat_id: {chat_id}",
        f"# chat_type: {chat_type}",
        f"# message_count: {len(chat_msgs)}",
        f"#",
        f"# INSTRUCTIONS:",
        f"#   Put '{BOUNDARY_MARKER}' at the START of any line that begins a NEW conversation.",
        f"#   Line [0] is always a boundary — the marker is pre-placed.",
        f"#   Save the file and run:",
        f"#     python message_miner/label_boundaries.py import \"{output_path}\"",
        f"#",
        f"# Format: [index] (time_gap) sender: text",
        "",
    ]

    for i, (_, row) in enumerate(chat_msgs.iterrows()):
        gap = _format_gap(row.get("time_delta_seconds"))
        gap_part = f" ({gap})" if gap and i > 0 else ""
        name = _first_name(row.get("sender_name"))
        text = str(row.get("text") or "").strip()[:500]
        # Pre-mark index 0 so the user doesn't forget it's always a boundary
        prefix = f"{BOUNDARY_MARKER} " if i == 0 else ""
        lines.append(f"{prefix}[{i}]{gap_part} {name}: {text}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(chat_msgs)} messages to {output_path}")
    print(f"Edit the file, then run: python message_miner/label_boundaries.py import \"{output_path}\"")


def cmd_import(args) -> None:
    """Parse an edited label file and merge its boundaries into golden_set.json."""
    input_path = Path(args.file)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    content = input_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Pull chat_id from header
    chat_id = None
    for line in lines[:20]:
        m = re.match(r"#\s*chat_id:\s*(.+)", line)
        if m:
            chat_id = m.group(1).strip()
            break
    if not chat_id:
        print(f"ERROR: Could not find '# chat_id: ...' header in {input_path}.")
        sys.exit(1)

    # Find boundaries — lines starting with the marker, extract the [N] index
    boundaries = []
    pattern = re.compile(rf"^\s*{re.escape(BOUNDARY_MARKER)}\s*\[(\d+)\]")
    for line in lines:
        m = pattern.match(line)
        if m:
            boundaries.append(int(m.group(1)))

    if not boundaries:
        print(f"ERROR: No boundary markers ('{BOUNDARY_MARKER}') found in {input_path}.")
        sys.exit(1)
    if 0 not in boundaries:
        print(f"WARNING: Index 0 is not marked as a boundary — adding it (it's always the start of the first conversation).")
        boundaries = [0] + boundaries

    boundaries = sorted(set(boundaries))

    # Merge into golden_set.json
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if GOLDEN_JSON.exists():
        existing = json.loads(GOLDEN_JSON.read_text(encoding="utf-8"))

    was_present = chat_id in existing
    existing[chat_id] = {"boundaries": boundaries}
    GOLDEN_JSON.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    verb = "updated" if was_present else "added"
    print(f"{verb.capitalize()} {chat_id}: {len(boundaries)} boundaries in {GOLDEN_JSON}")
    total = sum(len(v["boundaries"]) for v in existing.values())
    print(f"Golden set now contains {len(existing)} chat(s), {total} boundaries total.")
    print(f"\nRun validation with: python message_miner/validation.py --golden \"{GOLDEN_JSON}\"")


# ── Web labeler ───────────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Boundary Labeler</title>
<style>
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: #222; }
  #app { display: grid; grid-template-columns: 340px 1fr; height: 100vh; }
  #sidebar { border-right: 1px solid #ddd; overflow-y: hidden; display: flex; flex-direction: column; background: #fafafa; }
  #sidebar h1 { font-size: 15px; margin: 0; padding: 12px 14px; border-bottom: 1px solid #ddd; background: #fff; }
  .tabs { display: flex; border-bottom: 1px solid #ddd; background: #fff; }
  .tab { flex: 1; padding: 10px; border: none; background: #fff; cursor: pointer; font-size: 13px; color: #555; border-bottom: 2px solid transparent; }
  .tab.active { color: #2563eb; border-bottom-color: #2563eb; font-weight: 600; }
  #sidebar .toolbar { padding: 10px 14px; border-bottom: 1px solid #eee; background: #fff; display: flex; flex-direction: column; gap: 6px; }
  #sidebar .toolbar button { padding: 8px 10px; font-size: 13px; cursor: pointer; border: 1px solid #bbb; border-radius: 4px; background: #fff; }
  #sidebar .toolbar button.primary { background: #2563eb; color: #fff; border-color: #2563eb; }
  #sidebar .toolbar .stats { font-size: 12px; color: #555; }
  #listContainer { flex: 1; overflow-y: auto; }
  .chat-item { padding: 10px 14px; border-bottom: 1px solid #eee; cursor: pointer; font-size: 13px; }
  .chat-item:hover { background: #f0f0f0; }
  .chat-item.active { background: #e0e7ff; }
  .chat-item .type { color: #888; font-size: 11px; text-transform: uppercase; }
  .chat-item .id { font-family: monospace; font-size: 11px; color: #555; word-break: break-all; }
  .chat-item .meta { font-size: 12px; margin-top: 3px; }
  .chat-item .done { color: #16a34a; font-weight: 600; }
  .chat-item .reason { font-size: 12px; color: #555; margin-top: 4px; font-style: italic; }
  .chat-item .seen { color: #888; font-size: 11px; }
  #main { overflow-y: auto; padding: 20px 32px; }
  #main .header { font-size: 13px; color: #555; margin-bottom: 14px; padding-bottom: 10px; border-bottom: 1px solid #eee; }
  #main .header .count { font-weight: 600; color: #222; }
  .reason-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 4px; padding: 8px 10px; font-size: 13px; color: #92400e; margin-bottom: 12px; }
  .msg { padding: 6px 10px; border-radius: 4px; cursor: pointer; user-select: none; margin-bottom: 2px; border: 1px solid transparent; }
  .msg:hover { background: #f5f5f5; border-color: #e5e5e5; }
  .msg .sender { font-weight: 600; margin-right: 6px; }
  .msg .gap { color: #888; font-size: 11px; margin-right: 6px; }
  .msg .gap.big { color: #dc2626; font-weight: 600; font-size: 12px; }
  .msg .text { white-space: pre-wrap; word-break: break-word; }
  .msg.is-gap-boundary { border-left: 3px solid #9ca3af; background: #f3f4f6; }
  .msg.is-boundary { border-left: 3px solid #dc2626; background: #fef2f2; }
  .msg.is-suggested { box-shadow: inset 0 0 0 1px #f59e0b; }
  .msg.is-first { opacity: 0.8; font-style: italic; }
  .boundary-divider { display: flex; align-items: center; gap: 8px; margin: 12px 0 4px; padding: 0; color: #dc2626; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }
  .boundary-divider::before, .boundary-divider::after { content: ""; flex: 1; border-top: 2px solid #dc2626; }
  .boundary-divider.gap { color: #6b7280; }
  .boundary-divider.gap::before, .boundary-divider.gap::after { border-top-color: #9ca3af; border-top-style: dashed; }
  .hint { font-size: 12px; color: #666; margin-bottom: 12px; }
  .legend { font-size: 11px; color: #555; margin-top: 8px; display: flex; gap: 12px; flex-wrap: wrap; }
  .legend span { padding: 2px 6px; border-radius: 3px; border-left-width: 3px; border-left-style: solid; }
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <h1>Boundary Labeler</h1>
    <div class="tabs" id="tabs"></div>
    <div class="toolbar">
      <button class="primary" onclick="downloadJSON()">Download golden_set.json</button>
      <button onclick="clearCurrent()">Clear current</button>
      <button onclick="clearAll()">Clear ALL progress</button>
      <div class="stats" id="stats"></div>
    </div>
    <div id="listContainer"></div>
  </div>
  <div id="main">
    <div class="hint" id="hint">Click any message to mark/unmark it as the start of a new conversation. Progress is saved automatically in your browser.</div>
    <div class="legend">
      <span style="border-left-color:#9ca3af; background:#f3f4f6;">gap boundary (auto)</span>
      <span style="border-left-color:#dc2626; background:#fef2f2;">your boundary</span>
      <span style="box-shadow: inset 0 0 0 1px #f59e0b; padding:2px 6px;">LLM-suggested</span>
    </div>
    <div id="messages"></div>
  </div>
</div>
<script>
const DATA = __DATA__;
const STORAGE_KEY = "boundary_labels_v1";
const GAP_BOUNDS = DATA.gap_boundaries || {};

let state = loadState();
let mode = "chats";  // "chats" or "flagged"
let currentChatId = DATA.chats.length > 0 ? DATA.chats[0].chat_id : null;
let currentBurstId = (DATA.flagged && DATA.flagged.length > 0) ? DATA.flagged[0].burst_id : null;

function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
    if (!s.__meta) s.__meta = { seen_flagged: [] };
    return s;
  } catch (e) { return { __meta: { seen_flagged: [] } }; }
}
function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  renderStats();
  renderList();
}
function getBoundaries(chatId) {
  if (!state[chatId]) state[chatId] = { boundaries: [0] };
  if (!state[chatId].boundaries.includes(0)) state[chatId].boundaries.unshift(0);
  return state[chatId].boundaries;
}
function toggleBoundary(chatId, idx) {
  if (idx === 0) return;
  const b = getBoundaries(chatId);
  const i = b.indexOf(idx);
  if (i === -1) b.push(idx); else b.splice(i, 1);
  b.sort((a, c) => a - c);
  saveState();
  renderMain();
}
function clearCurrent() {
  if (mode === "chats" && currentChatId) {
    if (!confirm("Clear your boundaries for this chat?")) return;
    state[currentChatId] = { boundaries: [0] };
  } else if (mode === "flagged" && currentBurstId) {
    const burst = DATA.flagged.find(b => b.burst_id === currentBurstId);
    if (!burst) return;
    if (!confirm("Clear boundaries inside this burst (keeps others in the same chat)?")) return;
    const burstChatIndices = new Set((burst.messages || []).map(m => m.chat_msg_idx));
    const keep = getBoundaries(burst.chat_id).filter(i => !burstChatIndices.has(i) || i === 0);
    state[burst.chat_id] = { boundaries: keep };
    // also un-mark as seen so re-opening re-applies suggestions
    state.__meta.seen_flagged = (state.__meta.seen_flagged || []).filter(bid => bid !== burst.burst_id);
  }
  saveState();
  renderMain();
}
function clearAll() {
  if (!confirm("Clear ALL labeling progress for every chat? This cannot be undone.")) return;
  state = { __meta: { seen_flagged: [] } };
  saveState();
  renderMain();
}
function downloadJSON() {
  const out = {};
  const chatsWithState = Object.keys(state).filter(k => k !== "__meta");
  for (const chatId of chatsWithState) {
    const s = new Set(state[chatId].boundaries || []);
    s.add(0);
    for (const i of (GAP_BOUNDS[chatId] || [])) s.add(i);
    out[chatId] = { boundaries: [...s].sort((a, c) => a - c) };
  }
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "golden_set.json";
  a.click();
  URL.revokeObjectURL(url);
}
function renderStats() {
  const chatsTouched = Object.keys(state).filter(k => k !== "__meta").length;
  const boundaryCount = Object.entries(state)
    .filter(([k]) => k !== "__meta")
    .reduce((s, [, v]) => s + (v.boundaries || []).length, 0);
  document.getElementById("stats").textContent =
    `${chatsTouched} chat${chatsTouched === 1 ? "" : "s"} touched · ${boundaryCount} user boundaries`;
}
function setMode(m) {
  mode = m;
  renderTabs();
  renderList();
  renderMain();
}
function renderTabs() {
  const tabs = document.getElementById("tabs");
  const flaggedCount = (DATA.flagged || []).length;
  tabs.innerHTML = `
    <button class="tab ${mode === "chats" ? "active" : ""}" onclick="setMode('chats')">Chats (${DATA.chats.length})</button>
    <button class="tab ${mode === "flagged" ? "active" : ""}" onclick="setMode('flagged')">Flagged (${flaggedCount})</button>
  `;
}
function renderList() {
  const list = document.getElementById("listContainer");
  list.innerHTML = "";
  if (mode === "chats") {
    for (const chat of DATA.chats) {
      const div = document.createElement("div");
      div.className = "chat-item" + (chat.chat_id === currentChatId ? " active" : "");
      div.onclick = () => { currentChatId = chat.chat_id; renderList(); renderMain(); };
      const bCount = (state[chat.chat_id]?.boundaries || []).length;
      div.innerHTML = `
        <div class="type">${chat.chat_type}</div>
        <div class="id">${chat.chat_id}</div>
        <div class="meta">
          ${chat.messages.length} msgs
          ${bCount > 1 ? `<span class="done">· ${bCount} boundaries</span>` : ""}
        </div>`;
      list.appendChild(div);
    }
  } else {
    const flagged = DATA.flagged || [];
    if (flagged.length === 0) {
      list.innerHTML = `<div style="padding: 14px; color: #666; font-size: 13px;">
        No flagged bursts available. Run <code>07c_flag_bursts.py</code> to generate them.</div>`;
      return;
    }
    const seen = new Set(state.__meta?.seen_flagged || []);
    for (const burst of flagged) {
      const div = document.createElement("div");
      div.className = "chat-item" + (burst.burst_id === currentBurstId ? " active" : "");
      div.onclick = () => { currentBurstId = burst.burst_id; renderList(); renderMain(); };
      const splits = (burst.suggested_chat_indices || []).length;
      div.innerHTML = `
        <div class="type">${burst.chat_type} · ${burst.msg_count} msgs</div>
        <div class="id">${burst.burst_id}</div>
        <div class="meta">
          ${splits} suggested split${splits === 1 ? "" : "s"}
          ${seen.has(burst.burst_id) ? `<span class="seen">· reviewed</span>` : ""}
        </div>
        <div class="reason">${escapeHtml(burst.reason || "")}</div>`;
      list.appendChild(div);
    }
  }
}
function renderMain() {
  if (mode === "chats") renderChatView(); else renderBurstView();
}
function renderChatView() {
  const chat = DATA.chats.find(c => c.chat_id === currentChatId);
  const container = document.getElementById("messages");
  container.innerHTML = "";
  if (!chat) { container.textContent = "No chat selected."; return; }
  const userBoundaries = new Set(getBoundaries(chat.chat_id));
  const gapSet = new Set(GAP_BOUNDS[chat.chat_id] || []);

  const header = document.createElement("div");
  header.className = "header";
  header.innerHTML = `<div><span class="count">${chat.chat_type}</span> · ${chat.messages.length} messages · <span class="count">${userBoundaries.size}</span> your boundaries · <span class="count">${gapSet.size}</span> gap boundaries</div>
    <div style="font-family: monospace; font-size: 11px; margin-top: 4px;">${chat.chat_id}</div>`;
  container.appendChild(header);

  for (let i = 0; i < chat.messages.length; i++) {
    const m = chat.messages[i];
    const isUser = userBoundaries.has(i);
    const isGap = gapSet.has(i);
    if ((isUser || isGap) && i > 0) {
      const div = document.createElement("div");
      div.className = "boundary-divider" + (isUser ? "" : " gap");
      div.textContent = isUser ? "new conversation" : "gap boundary";
      container.appendChild(div);
    }
    const div = document.createElement("div");
    let cls = "msg";
    if (isUser) cls += " is-boundary";
    else if (isGap) cls += " is-gap-boundary";
    if (i === 0) cls += " is-first";
    div.className = cls;
    div.onclick = () => toggleBoundary(chat.chat_id, i);
    const gapCls = (m.gap && (m.gap.includes("h") || m.gap.includes("d"))) ? "gap big" : "gap";
    const gap = m.gap ? `<span class="${gapCls}">(${m.gap})</span>` : "";
    div.innerHTML = `<span class="sender">${escapeHtml(m.sender)}:</span>${gap}<span class="text">${escapeHtml(m.text)}</span>`;
    container.appendChild(div);
  }
}
function renderBurstView() {
  const burst = (DATA.flagged || []).find(b => b.burst_id === currentBurstId);
  const container = document.getElementById("messages");
  container.innerHTML = "";
  if (!burst) { container.textContent = "No burst selected (or none flagged)."; return; }

  // First-view auto-apply of LLM suggestions
  if (!state.__meta.seen_flagged.includes(burst.burst_id)) {
    state.__meta.seen_flagged.push(burst.burst_id);
    const b = getBoundaries(burst.chat_id);
    for (const idx of (burst.suggested_chat_indices || [])) {
      if (!b.includes(idx)) b.push(idx);
    }
    b.sort((a, c) => a - c);
    saveState();
  }

  const userBoundaries = new Set(getBoundaries(burst.chat_id));
  const suggestedSet = new Set(burst.suggested_chat_indices || []);

  const header = document.createElement("div");
  header.className = "header";
  const kept = [...suggestedSet].filter(i => userBoundaries.has(i)).length;
  header.innerHTML = `<div><span class="count">${burst.chat_type}</span> · ${burst.msg_count} msgs · <span class="count">${suggestedSet.size}</span> LLM splits (${kept} kept)</div>
    <div style="font-family: monospace; font-size: 11px; margin-top: 4px;">${burst.burst_id}</div>`;
  container.appendChild(header);

  const reason = document.createElement("div");
  reason.className = "reason-box";
  reason.textContent = "LLM reason: " + (burst.reason || "(none)");
  container.appendChild(reason);

  for (let i = 0; i < burst.messages.length; i++) {
    const m = burst.messages[i];
    const chatIdx = m.chat_msg_idx;
    const isUser = userBoundaries.has(chatIdx);
    const isSuggested = suggestedSet.has(chatIdx);
    if (isUser && i > 0) {
      const div = document.createElement("div");
      div.className = "boundary-divider";
      div.textContent = isSuggested ? "new conversation (LLM suggested)" : "new conversation";
      container.appendChild(div);
    }
    const div = document.createElement("div");
    let cls = "msg";
    if (isUser) cls += " is-boundary";
    if (isSuggested) cls += " is-suggested";
    if (i === 0) cls += " is-first";
    div.className = cls;
    div.onclick = () => toggleBoundary(burst.chat_id, chatIdx);
    const gapCls = (m.gap && (m.gap.includes("h") || m.gap.includes("d"))) ? "gap big" : "gap";
    const gap = m.gap ? `<span class="${gapCls}">(${m.gap})</span>` : "";
    div.innerHTML = `<span class="sender">${escapeHtml(m.sender)}:</span>${gap}<span class="text">${escapeHtml(m.text)}</span>`;
    container.appendChild(div);
  }
}
function escapeHtml(s) {
  return String(s || "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

renderTabs();
renderList();
renderMain();
renderStats();
</script>
</body>
</html>
"""


def cmd_web(args) -> None:
    """Generate a self-contained HTML page for point-and-click labeling.

    Embeds:
      - All suitable chats (or all if --all) for full-chat labeling.
      - Gap-detected boundaries from bursts.parquet (auto-unioned on download).
      - Flagged bursts from flagged_bursts.json if it exists, for focused review.
    """
    df = _load_messages()
    chats_df = _load_chats()

    counts = df.groupby("chat_id").size().rename("msg_count")
    chat_type_map = chats_df.set_index("chat_id")["chat_type"].astype(str).to_dict()

    # Burst counts per chat drive the "meaningful segmentation" filter.
    bursts_per_chat: dict[str, int] = {}
    if BURSTS_PATH.exists():
        bursts_df = pd.read_parquet(BURSTS_PATH)
        bursts_per_chat = bursts_df.groupby("chat_id").size().astype(int).to_dict()

    if args.all:
        chat_ids = counts.index.tolist()
    else:
        # Good golden-set candidates:
        #   - labelable in one sitting (min_msgs..max_msgs)
        #   - enough bursts that gap segmentation is doing non-trivial work
        #     (otherwise there's nothing to agree/disagree with)
        #   - stratified across chat_type so the sample isn't dominated by one category
        candidates: list[tuple[str, int, str]] = []
        for cid in counts.index:
            mc = int(counts[cid])
            bc = int(bursts_per_chat.get(cid, 0))
            if args.min_msgs <= mc <= args.max_msgs and bc >= args.min_bursts:
                ct = chat_type_map.get(cid, "unknown")
                candidates.append((cid, mc, ct))

        by_type: dict[str, list[tuple[str, int]]] = {}
        for cid, mc, ct in candidates:
            by_type.setdefault(ct, []).append((cid, mc))

        rng = random.Random(42)
        target = args.sample_size
        total = len(candidates)
        chat_ids = []
        if total > 0:
            # Proportional allocation, at least 1 per type that has candidates.
            for ct, items in by_type.items():
                rng.shuffle(items)
                quota = max(1, round(target * len(items) / total))
                chat_ids.extend(c for c, _ in items[:quota])
            chat_ids = chat_ids[:target]

    chat_ids = sorted(chat_ids, key=lambda c: counts[c])
    chat_ids_set = set(chat_ids)

    # Assign chat_msg_idx = row position within each chat's filtered+sorted messages
    df = df.sort_values(["chat_id", "timestamp"]).reset_index(drop=True)
    df["chat_msg_idx"] = df.groupby("chat_id").cumcount()

    chats_payload = []
    for chat_id in chat_ids:
        chat_msgs = df[df["chat_id"] == chat_id]
        msgs = []
        for i, (_, row) in enumerate(chat_msgs.iterrows()):
            msgs.append({
                "sender": _first_name(row.get("sender_name")),
                "gap": _format_gap(row.get("time_delta_seconds")) if i > 0 else "",
                "text": str(row.get("text") or "").strip()[:800],
            })
        chats_payload.append({
            "chat_id": chat_id,
            "chat_type": chat_type_map.get(chat_id, "unknown"),
            "messages": msgs,
        })

    # Gap-detected boundaries (from bursts.parquet if available)
    gap_bounds: dict[str, list[int]] = {}
    if BURSTS_PATH.exists() and BURST_MAP_PATH.exists():
        burst_map = pd.read_parquet(BURST_MAP_PATH)
        # Find first message_id per burst, map back to chat_msg_idx
        msg_to_idx = df.set_index("message_id")[["chat_id", "chat_msg_idx"]]
        enriched = burst_map.join(msg_to_idx, on="message_id", how="inner")
        # First chat_msg_idx per burst = start of that burst in its chat
        first_idx_per_burst = (
            enriched.sort_values("chat_msg_idx")
            .groupby("burst_id")
            .first()
            .reset_index()
        )
        for chat_id, g in first_idx_per_burst.groupby("chat_id"):
            if chat_id in chat_ids_set or args.all:
                gap_bounds[chat_id] = sorted(int(x) for x in g["chat_msg_idx"].tolist())

    # Flagged bursts (from 07c_flag_bursts output) — only include where multiple=true
    flagged_payload: list[dict] = []
    flagged_src = Path(args.flagged) if args.flagged else FLAGGED_PATH
    if flagged_src.exists():
        raw = json.loads(flagged_src.read_text(encoding="utf-8"))
        for b in raw.get("bursts", []):
            if not b.get("multiple"):
                continue
            burst_msgs_embedded = b.get("messages") or []
            # Keep only fields the UI needs
            msgs_for_ui = [
                {
                    "sender": m.get("sender"),
                    "gap": m.get("gap") or "",
                    "text": str(m.get("text") or "")[:800],
                    "chat_msg_idx": int(m.get("chat_msg_idx")),
                }
                for m in burst_msgs_embedded
            ]
            # Map burst-relative split indices → chat-level
            splits_in_burst = b.get("splits_in_burst") or []
            suggested_chat_indices = []
            for s in splits_in_burst:
                if 0 <= s < len(msgs_for_ui):
                    suggested_chat_indices.append(msgs_for_ui[s]["chat_msg_idx"])
            flagged_payload.append({
                "burst_id": b["burst_id"],
                "chat_id": b["chat_id"],
                "chat_type": b.get("chat_type", ""),
                "msg_count": b.get("msg_count", len(msgs_for_ui)),
                "reason": b.get("reason", ""),
                "messages": msgs_for_ui,
                "suggested_chat_indices": sorted(set(suggested_chat_indices)),
            })
        # Sort flagged by msg_count desc (longest first)
        flagged_payload.sort(key=lambda b: -b["msg_count"])

    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else GOLDEN_DIR / "label.html"
    payload = {
        "chats": chats_payload,
        "gap_boundaries": gap_bounds,
        "flagged": flagged_payload,
    }
    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    output_path.write_text(html, encoding="utf-8")

    total_msgs = sum(len(c["messages"]) for c in chats_payload)
    print(f"Wrote {len(chats_payload)} chats ({total_msgs} messages) to {output_path}")
    # Breakdown by chat_type so you can see the stratification.
    type_breakdown: dict[str, int] = {}
    for c in chats_payload:
        type_breakdown[c["chat_type"]] = type_breakdown.get(c["chat_type"], 0) + 1
    if type_breakdown:
        parts = ", ".join(f"{k}:{v}" for k, v in sorted(type_breakdown.items()))
        print(f"  by chat_type: {parts}")
    print(f"Gap-detected boundaries embedded for {len(gap_bounds)} chats")
    print(f"Flagged bursts embedded: {len(flagged_payload)}")
    print(f"\nOpen it in your browser:")
    print(f"  start \"\" \"{output_path.resolve()}\"")
    print(f"\nUse the 'Flagged' tab to review LLM-suggested splits point-and-click.")
    print(f"When done, click 'Download golden_set.json' and move the file to:")
    print(f"  {GOLDEN_JSON}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build a golden set of labeled boundaries.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Show chats available for labeling")
    p_list.add_argument("--limit", type=int, default=30, help="Rows to show (default: 30)")
    p_list.set_defaults(func=cmd_list)

    p_export = sub.add_parser("export", help="Export a chat's messages to a text file")
    p_export.add_argument("--chat", required=True, help="chat_id to export")
    p_export.add_argument("--output", default=None, help="Output path (default: golden_set/<chat_id>.txt)")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import", help="Parse an edited label file into golden_set.json")
    p_import.add_argument("file", help="Path to edited label file")
    p_import.set_defaults(func=cmd_import)

    p_web = sub.add_parser("web", help="Generate point-and-click HTML labeler")
    p_web.add_argument("--output", default=None, help="Output path (default: golden_set/label.html)")
    p_web.add_argument("--all", action="store_true", help="Include all chats, skip the golden-set filter")
    p_web.add_argument("--min-msgs", type=int, default=30, help="Min messages to include (default: 30)")
    p_web.add_argument("--max-msgs", type=int, default=200, help="Max messages to include (default: 200)")
    p_web.add_argument("--min-bursts", type=int, default=3, help="Require at least this many gap-detected bursts (default: 3)")
    p_web.add_argument("--sample-size", type=int, default=30, help="Target number of chats to include, stratified by chat_type (default: 30)")
    p_web.add_argument("--flagged", default=None, help="Path to flagged_bursts.json (default: outputs/07_segment/flagged_bursts.json)")
    p_web.set_defaults(func=cmd_web)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
