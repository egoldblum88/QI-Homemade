// The chat page's behavior.
// It talks ONLY to our own local server, at addresses starting with /api/.
// All model text is inserted with textContent (never as HTML), so it can't break the page.
"use strict";

const messagesEl = document.getElementById("messages");
const form = document.getElementById("composer");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const themeBtn = document.getElementById("theme-btn");
const infoEl = document.getElementById("model-info");
const memoryBtn = document.getElementById("memory-btn");
const memoryPanel = document.getElementById("memory-panel");
const memoryList = document.getElementById("memory-list");
const memoryCount = document.getElementById("memory-count");
const historyInfo = document.getElementById("history-info");
const knowledgeBtn = document.getElementById("knowledge-btn");
const knowledgePanel = document.getElementById("knowledge-panel");
const knowledgeList = document.getElementById("knowledge-list");
const knowledgeCount = document.getElementById("knowledge-count");
const worldBtn = document.getElementById("world-btn");
const worldPanel = document.getElementById("world-panel");
const worldList = document.getElementById("world-list");
const worldCount = document.getElementById("world-count");
const selfBtn = document.getElementById("self-btn");
const selfPanel = document.getElementById("self-panel");
const selfList = document.getElementById("self-list");
const selfCount = document.getElementById("self-count");
const stateBtn = document.getElementById("state-btn");
const statePanel = document.getElementById("state-panel");
const stateList = document.getElementById("state-list");
const ARROW = " " + String.fromCharCode(0x2192) + " ";
const RETURN_ARROW = String.fromCharCode(0x21b5);
const DOT = String.fromCharCode(0xb7);
const MARKERS = /(<user>|<ai>|<end>)/;
let busy = false;
let info = null;

// ---------------------------------------------------------------- server calls
async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
  let response;
  try {
    response = await fetch("/api/" + path, options);
  } catch (e) {
    throw new Error("Cannot reach the local server. Is it still running in Terminal?");
  }
  let data = {};
  try { data = await response.json(); } catch (e) { /* not JSON */ }
  if (!response.ok) throw new Error(data.error || "The server answered with error " + response.status + ".");
  return data;
}

// ---------------------------------------------------------------- building the page
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function showWelcome() {
  const box = el("div", "notice welcome");
  box.append(el("strong", "", "What this brain can do. "),
    info ? info.capability : "Loading the model information...",
    el("br"),
    el("span", "", "Under each reply, a short line shows how it was made; \"Details\" shows everything."));
  if (info && info.memory) {
    box.append(el("br"), el("span", "", "Long-term memory is on: lasting things you tell it (like your name) are " +
      "saved on this Mac and shown under \"Memory\", where you can delete them. \"Remember that ...\" and " +
      "\"Forget that ...\" also work."));
  }
  if (info && info.learning) {
    box.append(el("br"), el("span", "", "Learning is on: facts you state or teach (\"Learn that ...\") become knowledge, " +
      "shown under \"Knowledge\" with where each came from and how sure it is. \"Correction: ...\" updates them."));
  }
  if (info && info.reasoning) {
    box.append(el("br"), el("span", "", "The world model is on: questions like \"What breed is my dog?\" are answered " +
      "by reasoning over what it knows (shown under \"World\" and in \"Details\"); the brain then writes the reply, " +
      "and whether the reply said the answer is measured."));
  }
  messagesEl.append(box);
}

function removeWelcome() {
  messagesEl.querySelectorAll(".welcome").forEach((n) => n.remove());
}

function addMessage(role, text, details) {
  const row = el("div", "row " + role);
  row.append(el("div", "who", role === "user" ? "You" : "AI"));
  const bubble = el("div", "bubble");
  if (role === "ai" && !text) {
    bubble.classList.add("empty");
    bubble.textContent = "(the model produced no letters this time)";
  } else {
    bubble.textContent = text;
  }
  row.append(bubble);
  if (role === "ai" && details) row.append(renderMeta(details));
  messagesEl.append(row);
  scrollToBottom();
}

function addError(message) {
  messagesEl.append(el("div", "notice error", message));
  scrollToBottom();
}

function addTyping() {
  const row = el("div", "row ai typing");
  row.append(el("div", "who", "AI"));
  const bubble = el("div", "bubble");
  bubble.setAttribute("aria-label", "The model is writing");
  bubble.append(el("span", "dot"), el("span", "dot"), el("span", "dot"));
  row.append(bubble);
  messagesEl.append(row);
  scrollToBottom();
  return row;
}

// ---------------------------------------------------------------- how a reply was made
function charLabel(ch) {
  if (ch === " ") return "space";
  if (ch === "\n") return RETURN_ARROW + " new line";
  return ch;
}

// Show invisible characters: newline as a return arrow, and (optionally) spaces as dots.
function visible(text, showSpaces) {
  const box = el("pre", "mono");
  for (const part of text.split(MARKERS)) {
    if (MARKERS.test(part)) {
      box.append(el("span", "tag", part));
      continue;
    }
    for (const ch of part) {
      if (ch === "\n") box.append(el("span", "ws", RETURN_ARROW), "\n");
      else if (ch === " " && showSpaces) box.append(el("span", "ws", DOT));
      else box.append(ch);
    }
  }
  return box;
}

// Your message, each word shaded by how much attention the reply paid to it.
function attentionView(d) {
  const box = el("div", "attn");
  const shares = d.attention_by_word || [];
  let last = 0, i = 0;
  const wordPattern = /[A-Za-z0-9']+/g;
  let match;
  while ((match = wordPattern.exec(d.message_seen)) !== null) {
    box.append(d.message_seen.slice(last, match.index));
    const share = shares[i] ? shares[i][1] : 0;
    const word = el("span", "w", match[0]);
    word.style.backgroundColor = "color-mix(in srgb, var(--accent) " + Math.min(85, Math.round(share * 220)) + "%, transparent)";
    word.title = Math.round(share * 100) + "% of the attention weight that fell on your message";
    box.append(word);
    last = wordPattern.lastIndex;
    i += 1;
  }
  box.append(d.message_seen.slice(last));
  const wrap = el("div");
  wrap.append(box, el("p", "plain", "Share of the total attention weight that fell on your message (the rest fell " +
    "on its own reply so far, the markers and earlier turns): " + Math.round(d.attention_on_message * 100) + "%. " +
    "Attention weights show where the computation drew information from. They are not a measure of understanding."));
  return wrap;
}

function isReplyStage(d) {
  return Boolean(d.stage) && d.stage !== "1";      // every brain after Stage 1 (the engine's own rule), not a fixed list
}

function section(title, content) {
  const wrap = el("div");
  wrap.append(el("h4", "", title), content);
  return wrap;
}

function renderDetails(d) {
  const body = el("div", "how-body");
  body.append(section("What this brain is", el("p", "plain", d.capability)));
  if (isReplyStage(d)) {
    const received = el("div");
    received.append(visible(d.prompt, false));
    if (d.turns_available > d.turns_in_prompt && !d.answer_alone) {
      received.append(el("p", "plain warn", "Only the last " + d.turns_in_prompt + " of " + d.turns_available +
        " turns fit into the model's context; the older ones were not given to it."));
    }
    if (d.message_cut) received.append(el("p", "plain warn", "Your message was too long: only its end was given to the model."));
    if (d.answer_alone && d.statement) {
      received.append(el("p", "plain", d.statement.status === "known" ?
        "Your message is a statement, so it was read BEFORE the reply (on a throwaway copy: nothing was stored yet) and " +
        "the brain was given only what it told, as a structured line, and your message. It is stored after the reply." :
        "Nothing could read this unfinished message, so the brain was given only \"unknown: what you mean.\" and " +
        "your message."));
    } else if (d.answer_alone) {
      received.append(el("p", "plain", "The reasoning layer answered this question, so the brain was given only " +
        "the structured answer (the reasoning result first, then the facts it rests on) and your message. " +
        "Earlier turns are left out on purpose: the brain copied names from them. Nothing else was added."));
    } else if (d.memory_turns_in_prompt && d.reasoning && d.reasoning.used) {
      received.append(el("p", "plain", "The first " + d.memory_turns_in_prompt + " turns are the facts the reasoning " +
        "layer's answer rests on, written as plain statements from the stored facts, each followed by the brain's own " +
        "short reply to that sentence alone. Nothing else was added: no answer, no instruction."));
    } else if (d.memory_turns_in_prompt) {
      received.append(el("p", "plain", "The first " + d.memory_turns_in_prompt + " turns are recalled memories or " +
        "learned facts: your own earlier words, each followed by the brain's own short reply to that sentence alone " +
        "(generated now, so it cannot bring back an old value). Nothing else was added."));
    }
    body.append(section("1. Exactly what the model received (the markers are single tokens you cannot type)", received));
    body.append(section("2. Attention distribution over your message (all layers and heads averaged)", attentionView(d)));
  } else {
    body.append(section("1. The conversation, written as book dialogue", visible(d.prompt, false)));
    body.append(section("2. The model can only see the last " + d.context_size + " characters",
      visible(d.context_window, true)));
  }
  body.append(section("3. What it actually receives: one number per token", el("pre", "mono", d.context_ids.join(" "))));
  if (d.reasoning) body.append(section("World model + reasoning: each step apart (interpretation, retrieval, " +
    "reasoning, what the brain was shown, and whether the reply said it)", reasoningView(d.reasoning, d.statement,
    d.reference)));
  if (d.memory) body.append(section("Long-term memory: what was FOUND (retrieval), measured apart from the reply", memoryView(d.memory)));
  if (d.knowledge) body.append(section("Learned knowledge: what was RECALLED and what was LEARNED", knowledgeView(d.knowledge)));
  if (d.state) body.append(section("Internal state: this message's appraisal and the change it caused (computed after " +
    "the reply; not shown to the brain)", stateChangeView(d.state)));
  if (d.self && d.self.question) body.append(section("Self-model: a question about the system itself (this report was " +
    "NOT shown to the brain; the reply above was written by the brain alone)", selfReportView(d.self.question, d.self.brain_reply)));

  const steps = el("div");
  d.steps.forEach((s, i) => {
    const line = el("div", "step");
    line.append(el("span", "step-n", String(i + 1)));
    s.top.forEach(([ch, p]) => {
      const chip = el("span", "chip" + (ch === s.chose ? " chosen" : ""));
      chip.append(el("span", "", charLabel(ch)), el("span", "pct", Math.round(p * 100) + "%"));
      line.append(chip);
    });
    if (!s.top.some(([ch]) => ch === s.chose)) line.append(el("span", "chip chosen", charLabel(s.chose)));
    steps.append(line);
  });
  body.append(section("4. Next-token probabilities: the model's top 3 for each of the first tokens (the one the " +
    "decoder picked is outlined)", steps));

  const facts = el("div", "facts");
  const add = (t) => facts.append(el("span", "", t));
  add(d.chars_generated + " characters written in " + d.elapsed_ms + " ms");
  add("stopped because " + d.stopped_by);
  add("mean next-token probability of the chosen tokens: " + Math.round(d.mean_token_probability * 100) +
    "% (a property of the model's probability distribution, not confidence and not correctness)");
  add("decoding: " + d.decoding);
  if (d.attempts > 1) add("needed " + d.attempts + " tries to produce letters");
  if (d.removed_characters && d.removed_characters.length) add("removed unknown characters: " + d.removed_characters.join(" "));
  body.append(section("5. Facts", facts));
  return body;
}

// Retrieval (which memories were found, and their scores) is shown apart from the reply (generation).
function memoryView(m) {
  const box = el("div");
  if (!m.retrieved.length) {
    box.append(el("p", "plain", "No stored memory was relevant enough to this message, so none was given to the model."));
  }
  m.retrieved.forEach((r) => {
    const line = el("div", "mem-found");
    line.append(el("strong", "", r.summary), " ",
      el("span", r.in_prompt ? "" : "off", (r.in_prompt ? "given to the model" : "found, but did not fit") +
        " " + DOT + " relevance " + Math.round(r.relevance * 100) + "% (words " + Math.round(r.parts.words * 100) +
        "%, names " + Math.round(r.parts.names * 100) + "%, kind " + Math.round(r.parts.kind * 100) + "%)"));
    box.append(line);
  });
  if (m.in_prompt) {
    box.append(el("p", "plain", m.reply_mentions.length
      ? "The reply uses: " + m.reply_mentions.join(", ") + " (from the recalled memories)."
      : "The reply does not contain the recalled memories' names or words. Retrieval can succeed while " +
        "generation fails: that is a limit of the brain, not of memory."));
  }
  const changes = [];
  m.formed.forEach((f) => changes.push("stored: " + f.summary + " (" + f.reason + ")"));
  m.forgotten.forEach((f) => changes.push("forgotten: " + f.summary));
  if (m.replaced.length) changes.push("replaced " + m.replaced.length + " older memory (a newer one about the same thing)");
  if (!changes.length && m.note) changes.push("nothing stored: " + m.note);
  changes.forEach((c) => box.append(el("p", "plain", c)));
  return box;
}

// Learning: recalled facts (given to the brain) and what this message taught, apart from the reply.
function knowledgeView(k) {
  const box = el("div");
  if (!k.recalled.length) box.append(el("p", "plain", "No learned fact was relevant enough to this message."));
  k.recalled.forEach((f) => {
    const line = el("div", "mem-found");
    line.append(el("strong", "", f.about + " " + f.relation + ": " + f.value), " ",
      el("span", f.in_prompt ? "" : "off", (f.in_prompt ? "given to the model" : "found, but did not fit") + " " + DOT +
        " relevance " + Math.round(f.relevance * 100) + "% " + DOT + " confidence " + Math.round(f.confidence * 100) + "%"));
    box.append(line);
  });
  const changes = [];
  k.learned.forEach((f) => changes.push("learned: " + f.about + " " + f.relation + " " + (f.negated ? "NOT " : "") + f.value +
    " (" + f.source_label + ")"));
  k.confirmed.forEach((f) => changes.push("confirmed again: " + f.about + " " + f.relation + " " + f.value +
    " (" + f.confirmations + "x)"));
  k.corrected.forEach((c) => changes.push("corrected: " + c.from + " -> " + c.to + " (the old value is kept as history)"));
  k.changed.forEach((c) => changes.push("changed without a correction: " + c.from + " -> " + c.to + " (flagged, less certain)"));
  k.connected.forEach((c) => changes.push("connected: " + c));
  k.forgotten.forEach((f) => changes.push("removed: " + f.about + " " + f.relation + " " + f.value));
  if (!changes.length && k.note) changes.push(k.note);
  changes.forEach((c) => box.append(el("p", "plain", c)));
  return box;
}

// The reasoning pipeline, step by step. Only short records of operations: there is no hidden text to show.
function reasoningView(r, statement, reference) {
  const box = el("div");
  const i = r.interpretation;
  const row = (label, text, cls) => {
    const line = el("div", "r-step" + (cls ? " " + cls : ""));
    line.append(el("span", "r-label", label), el("span", "", text));
    box.append(line);
  };
  if (reference) {                                // a follow-up's it / its / she ... (D-081)
    row("0 Reference", reference.substitutions.map((s) => "\"" + s[0] + "\" " + ARROW + " \"" + s[1] + "\"").join(", ") +
      " (the one fitting thing of the previous turn). Read as: \"" + reference.resolved + "\". Your message is kept " +
      "as you wrote it.");
  }
  if (!i.understood) {
    box.append(el("p", "plain", "Interpretation: " + i.note + "."));
    if (statement) {                              // a statement read before the reply (D-079)
      row("Read before the reply", statement.note);
      row("  shown to the brain", statement.shown_to_brain.join("  "), "r-" + statement.status);
      const e = statement.expression;
      row("The brain's reply", (e.said ? "said it" : "did NOT say it") + " (" + e.check + "; expected: " + e.expected +
        "). This is measured only; the reply is never changed.", e.said ? "" : "r-miss");
    }
    return box;
  }
  const what = i.task ? "how to " + i.task : "path: " + i.path.split(" -> ").join(ARROW) +
    (i.other ? " and " + i.other.split(" -> ").join(ARROW) : "") + (i.value ? " (checking: " + i.value + ")" : "");
  row("1 Interpretation", i.type + "  " + DOT + "  " + what + "  " + DOT + "  rule: " + i.form);
  row("2 Retrieval", r.retrieval.count + (r.retrieval.count === 1 ? " fact" : " facts") + " found, starting from " +
    (r.retrieval.started_from.map((s) => s.name + (s.found.length ? " = " + s.found.join(", ") : " (unknown)")).join("; ") || "-"));
  const steps = el("div", "r-steps");
  r.reasoning.steps.forEach((t) => steps.append(el("div", /^(Conflict|Uncertain)/.test(t) ? "r-warn" : "", t.split(" -> ").join(ARROW))));
  row("3 Reasoning", r.reasoning.operations.join(", ") || "-");
  box.append(steps);
  row("4 Answer data", r.representation.frame_line.split(" -> ").join(ARROW), "r-" + r.reasoning.status);
  row("  shown to the brain", r.representation.shown_to_brain.length ? r.representation.shown_to_brain.join("  ") :
    "(" + r.representation.note + ")");
  if (r.expression) {
    row("5 The brain's reply", (r.expression.said ? "expressed it" : "did NOT express it") + " (" + r.expression.check +
      "; expected: " + r.expression.expected + "). This is measured only; the reply is never changed.",
      r.expression.said ? "" : "r-miss");
  }
  return box;
}

function addKnowledgeNotices(k) {
  if (!k) return;
  k.learned.forEach((f) => messagesEl.append(el("div", "notice memory", "Learned: " + f.about + " " + f.relation + " " +
    (f.negated ? "NOT " : "") + f.value)));
  k.corrected.forEach((c) => messagesEl.append(el("div", "notice memory", "Corrected: " + c.from + " -> " + c.to)));
  k.changed.forEach((c) => messagesEl.append(el("div", "notice memory", "Changed (flagged): " + c.from + " -> " + c.to)));
  k.confirmed.forEach((f) => messagesEl.append(el("div", "notice memory", "Confirmed again: " + f.about + " " +
    f.relation + " " + f.value)));
}

function addMemoryNotices(m) {
  if (!m) return;
  m.formed.forEach((f) => messagesEl.append(el("div", "notice memory", "Memory stored: " + f.summary)));
  m.forgotten.forEach((f) => messagesEl.append(el("div", "notice memory", "Memory deleted: " + f.summary)));
  (m.contradicted || []).forEach((f) => messagesEl.append(el("div", "notice memory", "Memory set aside (you said otherwise): " +
    f.summary)));
}

function renderMeta(d) {
  const wrap = el("div", "meta");
  const line = el("div", "meta-line");
  const parts = [];
  if (isReplyStage(d) && d.attention_words && d.attention_words.length) {
    parts.push("attention over your message: " + d.attention_words.slice(0, 3)
      .map(([w, s]) => "\"" + w + "\" " + Math.round(s * 100) + "%").join(", "));
  } else if (!isReplyStage(d)) {
    parts.push("input: only the last " + d.context_size + " characters");
  }
  const recalled = (d.memory_turns_in_prompt || 0) / 2;          // memories and learned facts given to the brain
  const r = d.reasoning;
  if (d.reference) parts.push("read as: " + d.reference.resolved);
  if (r && r.used) {
    const answer = r.reasoning.status === "unknown" ? "unknown" : r.reasoning.status + ": " + r.reasoning.answer.join(", ");
    parts.push("reasoning: " + answer + (r.expression ? (r.expression.said ? " (the reply says it)" : " (the reply does not say it)") : ""));
  } else if (d.statement) {
    parts.push("your statement, read: " + d.statement.shown_to_brain[0] +
      (d.statement.expression.said ? " (the reply says it)" : " (the reply does not say it)"));
  } else if (recalled) parts.push("recalled " + recalled + (recalled === 1 ? " memory or fact" : " memories or facts"));
  const q = d.self && d.self.question;
  if (q) parts.push(q.understood ? "self-model report: " + q.status + " (not shown to the brain)" : "outside the self-model");
  parts.push("mean next-token probability " + Math.round(d.mean_token_probability * 100) + "%");
  parts.push("sampled at temperature " + d.temperature);
  parts.push(d.elapsed_ms + " ms");
  if (d.turns_available > d.turns_in_prompt && !d.answer_alone) parts.push("older turns did not fit");
  line.append(el("span", "", parts.join("  " + DOT + "  ")));

  const panel = renderDetails(d);
  panel.hidden = true;
  const toggle = el("button", "link-btn", "Details");
  toggle.type = "button";
  toggle.setAttribute("aria-expanded", "false");
  toggle.addEventListener("click", () => {
    panel.hidden = !panel.hidden;
    toggle.textContent = panel.hidden ? "Details" : "Hide details";
    toggle.setAttribute("aria-expanded", String(!panel.hidden));
  });
  line.append(toggle);
  wrap.append(line, panel);
  return wrap;
}

// ---------------------------------------------------------------- actions
function updateSendButton() {
  sendBtn.disabled = busy || input.value.trim() === "";
}

function setBusy(value) {
  busy = value;
  input.setAttribute("aria-busy", String(value));
  clearBtn.disabled = value;
  updateSendButton();
}

function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 170) + "px";
  input.style.overflowY = input.scrollHeight > 170 ? "auto" : "hidden";
}

async function send() {
  const text = input.value.trim();
  if (!text || busy) return;
  setBusy(true);
  removeWelcome();
  addMessage("user", text);
  input.value = "";
  autoGrow();
  const typing = addTyping();
  try {
    const data = await api("chat", { message: text });
    typing.remove();
    addMemoryNotices(data.details.memory);
    addKnowledgeNotices(data.details.knowledge);
    addMessage("ai", data.reply, data.details);
    if (data.details.memory) loadMemories();
    if (data.details.knowledge) loadKnowledge();
    if (data.details.reasoning) { lastTried = null; if (!worldPanel.hidden) loadWorld(); else loadWorldCount(); }
    if (data.details.self) { if (!selfPanel.hidden) loadSelf(); else loadSelfCount(); }
    if (data.details.state && !statePanel.hidden) loadState();
  } catch (err) {
    typing.remove();
    addError(err.message);
  } finally {
    setBusy(false);
    input.focus();
  }
}

async function clearConversation() {
  if (busy || !window.confirm("Clear the whole conversation?")) return;
  try {
    await api("clear", {});
    messagesEl.replaceChildren();
    showWelcome();
  } catch (err) {
    addError(err.message);
  }
}

// ---------------------------------------------------------------- long-term memory panel
const TYPE_TITLES = [["fact", "Facts"], ["relationship", "People, pets and things"],
  ["preference", "Likes and dislikes"], ["episodic", "Events"]];

function when(seconds) {
  return seconds ? new Date(seconds * 1000).toLocaleString() : "never";
}

async function loadMemories() {
  if (!info || !info.memory) return;
  try {
    renderMemories(await api("memories"));
  } catch (err) {
    memoryList.replaceChildren(el("p", "memory-empty", err.message));
  }
}

function renderMemories(data) {
  memoryCount.textContent = String(data.memories.length);
  historyInfo.textContent = "Conversation history (kept separately from memory): " + data.history.turns +
    " messages in " + data.history.sessions + " sessions, in state/" + data.database + ".";
  memoryList.replaceChildren();
  if (!data.memories.length) {
    memoryList.append(el("p", "memory-empty", "Nothing is remembered yet. Tell it something lasting, like " +
      "\"My name is ...\" or \"I have a dog named ...\"."));
    return;
  }
  TYPE_TITLES.forEach(([type, title]) => {
    const items = data.memories.filter((m) => m.type === type);
    if (!items.length) return;
    memoryList.append(el("h3", "", title));
    items.forEach((m) => memoryList.append(memoryItem(m)));
  });
}

function memoryItem(m) {
  const box = el("div", "mem-item" + (m.pinned ? " pinned" : ""));
  box.append(el("div", "mem-summary", m.summary), el("div", "mem-words", "\u201c" + m.content + "\u201d"),
    el("div", "mem-meta", m.source_label + " " + DOT + " confidence " + Math.round(m.confidence * 100) + "% " + DOT +
      " importance " + Math.round(m.importance * 100) + "%" + (m.pinned ? " (marked important)" : "") + " " + DOT +
      " recalled " + m.retrieval_count + "x"));
  const why = el("div", "mem-why");
  why.hidden = true;
  const actions = el("div", "mem-actions");
  const whyBtn = el("button", "link-btn", "Why?");
  whyBtn.type = "button";
  whyBtn.addEventListener("click", async () => {
    if (!why.hidden) { why.hidden = true; return; }
    try {
      const w = await api("memories/why?id=" + m.id);
      why.replaceChildren(
        el("div", "", "Why: " + w.reason + "."),
        el("div", "", "Your message: \u201c" + w.your_message + "\u201d"),
        el("div", "", "Stored: " + when(w.created) + " " + DOT + " last recalled: " + when(w.last_retrieved)),
        el("div", "", "Source: " + w.source_label + ". The model's own words are never stored as memories."));
      why.hidden = false;
    } catch (err) { addError(err.message); }
  });
  const pin = el("button", "link-btn", m.pinned ? "Unmark important" : "Mark important");
  pin.type = "button";
  pin.addEventListener("click", async () => {
    try { await api("memories/important", { id: m.id, important: !m.pinned }); loadMemories(); }
    catch (err) { addError(err.message); }
  });
  const del = el("button", "link-btn danger", "Delete");
  del.type = "button";
  del.addEventListener("click", async () => {
    if (!window.confirm("Delete this memory? \u201c" + m.summary + "\u201d")) return;
    try { await api("memories/delete", { id: m.id }); loadMemories(); }
    catch (err) { addError(err.message); }
  });
  actions.append(whyBtn, pin, del);
  box.append(actions, why);
  return box;
}

function setMemoryPanel(open) {
  memoryPanel.hidden = !open;
  memoryBtn.setAttribute("aria-expanded", String(open));
  if (open) {
    if (!knowledgePanel.hidden) { knowledgePanel.hidden = true; knowledgeBtn.setAttribute("aria-expanded", "false"); }
    if (!worldPanel.hidden) { worldPanel.hidden = true; worldBtn.setAttribute("aria-expanded", "false"); }
    if (!selfPanel.hidden) setSelfPanel(false);
    if (!statePanel.hidden) setStatePanel(false);
    loadMemories();
  }
}

memoryBtn.addEventListener("click", () => setMemoryPanel(memoryPanel.hidden));
document.getElementById("memory-close").addEventListener("click", () => setMemoryPanel(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !memoryPanel.hidden) setMemoryPanel(false); });
document.getElementById("memory-clear").addEventListener("click", async () => {
  if (!window.confirm("Delete EVERY memory? This cannot be undone.")) return;
  try { await api("memories/clear", { confirm: true }); loadMemories(); } catch (err) { addError(err.message); }
});
document.getElementById("history-clear").addEventListener("click", async () => {
  if (!window.confirm("Delete the stored conversation history? Memories are kept.")) return;
  try { await api("history/clear", { confirm: true }); loadMemories(); } catch (err) { addError(err.message); }
});

// ---------------------------------------------------------------- learned knowledge panel
async function loadKnowledge() {
  if (!info || !info.learning) return;
  try {
    renderKnowledge(await api("knowledge"));
  } catch (err) {
    knowledgeList.replaceChildren(el("p", "memory-empty", err.message));
  }
}

function factItem(f) {
  const box = el("div", "k-fact");
  box.append(el("div", "k-fact-line", ""));
  box.firstChild.append(el("span", "k-rel", f.relation + ": "), el("strong", "", (f.negated ? "NOT " : "") + f.value));
  box.append(el("div", "mem-meta", f.source_label + " " + DOT + " confidence " + Math.round(f.confidence * 100) + "% " + DOT +
    " learned " + when(f.first_learned) + " " + DOT + " confirmed " + f.confirmations + "x" +
    (f.confirmations > 1 ? " (last " + when(f.last_confirmed) + ")" : "")));
  if (f.previous.length) box.append(el("div", "mem-meta", "earlier values (kept as history): " + f.previous.join(", ")));
  if (f.note) box.append(el("div", /conflicts/.test(f.note) ? "k-note" : "mem-meta", f.note));   // red only for a conflict
  const why = el("div", "mem-why");
  why.hidden = true;
  const actions = el("div", "mem-actions");
  const whyBtn = el("button", "link-btn", "Why?");
  whyBtn.type = "button";
  whyBtn.addEventListener("click", async () => {
    if (!why.hidden) { why.hidden = true; return; }
    try {
      const w = await api("knowledge/why?id=" + f.id);
      why.replaceChildren(el("div", "", "Why: " + w.reason + "."), el("div", "", "Taught by: \u201c" + w.text + "\u201d"),
        el("div", "", "Source: " + w.source_label + ". The model's own words are never learned."));
      why.hidden = false;
    } catch (err) { addError(err.message); }
  });
  const del = el("button", "link-btn danger", "Remove");
  del.type = "button";
  del.addEventListener("click", async () => {
    if (!window.confirm("Remove this learned fact and its history? \u201c" + f.text + "\u201d")) return;
    try { await api("knowledge/remove", { id: f.id }); loadKnowledge(); } catch (err) { addError(err.message); }
  });
  actions.append(whyBtn, del);
  box.append(actions, why);
  return box;
}

function renderKnowledge(data) {
  const c = data.counts;
  knowledgeCount.textContent = String(c.about_you + c.taught + c.general + c.skill);
  document.getElementById("knowledge-info").textContent = (c.about_you + c.taught + c.general + c.skill) +
    " current facts about " + c.entities + " things, " + c.superseded + " earlier values kept as history, in state/" +
    data.database + ".";
  knowledgeList.replaceChildren();
  let any = false;
  data.groups.forEach((g) => {
    if (!g.entities.length) return;
    any = true;
    knowledgeList.append(el("h3", "", g.label));
    g.entities.forEach((e) => {
      const box = el("div", "k-entity");
      const head = el("div", "k-entity-name", e.label);
      const extra = [];
      if (e.former_names.length) extra.push("formerly " + e.former_names.join(", "));
      if (e.inferred_names.length) extra.push("connected by the system: " + e.inferred_names.join(", "));
      if (extra.length) head.append(" ", el("span", "k-entity-names", "(" + extra.join("; ") + ")"));
      box.append(head);
      if (e.connections.length) box.append(el("div", "k-connections", "connected to: " + e.connections.join(", ")));
      e.facts.forEach((f) => box.append(factItem(f)));
      knowledgeList.append(box);
    });
  });
  if (!any) {
    knowledgeList.append(el("p", "memory-empty", "Nothing learned yet. Teach it something, e.g. \"Learn that water " +
      "freezes at 0 degrees Celsius.\" or \"The capital of France is Paris.\""));
  }
  if (data.experience.length) {
    knowledgeList.append(el("h3", "", "Learning experience (newest first)"));
    data.experience.slice(0, 15).forEach((x) =>
      knowledgeList.append(el("div", "k-event", when(x.t) + " " + DOT + " " + x.kind + ": " + x.detail)));
  }
}

function setKnowledgePanel(open) {
  knowledgePanel.hidden = !open;
  knowledgeBtn.setAttribute("aria-expanded", String(open));
  if (open) { setMemoryPanel(false); setWorldPanel(false); setSelfPanel(false); setStatePanel(false); loadKnowledge(); }
}

knowledgeBtn.addEventListener("click", () => setKnowledgePanel(knowledgePanel.hidden));
document.getElementById("knowledge-close").addEventListener("click", () => setKnowledgePanel(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !knowledgePanel.hidden) setKnowledgePanel(false); });
document.getElementById("knowledge-clear").addEventListener("click", async () => {
  if (!window.confirm("Remove EVERYTHING that was learned? This cannot be undone.")) return;
  try { await api("knowledge/clear", { confirm: true }); loadKnowledge(); } catch (err) { addError(err.message); }
});

// ---------------------------------------------------------------- world model + reasoning panel
let lastTried = null;

async function loadWorld() {
  if (!info || !info.reasoning) return;
  try {
    renderWorld(await api("world"));
  } catch (err) {
    worldList.replaceChildren(el("p", "memory-empty", err.message));
  }
}

function edgeItem(x, entityLabel) {
  const box = el("div", "k-fact");
  const line = el("div", "k-fact-line");
  const status = x.status !== "known" ? " (" + x.status + ")" : "";
  line.append(el("span", "k-rel", x.relation + ARROW), el("strong", "", (x.implicit ? x.target : x.value) + status));
  if (entityLabel) line.prepend(el("span", "k-rel", entityLabel + ARROW));
  box.append(line);
  box.append(el("div", "mem-meta", x.source + " " + DOT + " confidence " + Math.round(x.confidence * 100) + "%" +
    (x.first_learned ? " " + DOT + " learned " + when(x.first_learned) : "") +
    (x.confirmations > 1 ? " " + DOT + " confirmed " + x.confirmations + "x (last " + when(x.last_confirmed) + ")" : "") +
    (x.quantity ? " " + DOT + " a quantity: " + x.quantity : "")));
  if (x.previous && x.previous.length) {
    box.append(el("div", /changed/.test(x.previous[0].how) ? "k-note" : "mem-meta", "previous values: " +
      x.previous.map((h) => h.value + " (" + h.how + ")").join(", ")));
  }
  if (x.memory_id) box.append(el("div", "mem-meta", "read from long-term memory (learning was off when you said it)"));
  return box;
}

function renderWorld(data) {
  const c = data.counts;
  worldCount.textContent = String(c.entities);
  document.getElementById("world-info").textContent = c.entities + " entities, " + c.facts + " facts, " +
    c.relations + " relations (all learned from you; none predefined), " + c.with_history + " with correction history.";
  worldList.replaceChildren();
  const last = lastTried || data.last;
  if (last && last.interpretation) {
    worldList.append(el("h3", "", lastTried ? "Reasoning (your test question)" : "Last reasoning"));
    worldList.append(reasoningView(last));
  }
  worldList.append(el("h3", "", "Entities"));
  if (!data.entities.length) {
    worldList.append(el("p", "memory-empty", "Nothing known yet. Tell it something, e.g. \"Water freezes at 0 degrees " +
      "Celsius.\" or \"I have a dog named Max.\""));
  }
  data.entities.forEach((e) => {
    const box = el("div", "k-entity");
    const head = el("div", "k-entity-name", e.label);
    const extra = [e.kind];
    if (e.former_names.length) extra.push("formerly " + e.former_names.join(", "));
    head.append(" ", el("span", "k-entity-names", "(" + extra.join("; ") + ")"));
    box.append(head);
    if (e.properties.length) box.append(el("div", "k-connections", "properties"));
    e.properties.forEach((x) => box.append(edgeItem(x)));
    if (e.relationships.length) box.append(el("div", "k-connections", "relationships"));
    e.relationships.forEach((x) => box.append(edgeItem(x)));
    if (e.referred_to_by.length) {
      box.append(el("div", "k-connections", "referred to by: " + e.referred_to_by.map((x) =>
        x.subject + ARROW + x.relation).join(", ")));
    }
    worldList.append(box);
  });
  const rel = Object.entries(data.relations);
  if (rel.length) {
    worldList.append(el("h3", "", "Relations learned so far"));
    worldList.append(el("p", "plain", rel.map(([name, n]) => name + " (" + n + ")").join(", ")));
  }
}

async function loadWorldCount() {
  try { worldCount.textContent = String((await api("world")).counts.entities); } catch (err) { /* shown when opened */ }
}

function setWorldPanel(open) {
  worldPanel.hidden = !open;
  worldBtn.setAttribute("aria-expanded", String(open));
  if (open) {
    if (!memoryPanel.hidden) { memoryPanel.hidden = true; memoryBtn.setAttribute("aria-expanded", "false"); }
    if (!knowledgePanel.hidden) { knowledgePanel.hidden = true; knowledgeBtn.setAttribute("aria-expanded", "false"); }
    if (!selfPanel.hidden) setSelfPanel(false);
    if (!statePanel.hidden) setStatePanel(false);
    loadWorld();
  }
}

worldBtn.addEventListener("click", () => setWorldPanel(worldPanel.hidden));
document.getElementById("world-close").addEventListener("click", () => setWorldPanel(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !worldPanel.hidden) setWorldPanel(false); });
document.getElementById("world-try").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = document.getElementById("world-question").value.trim();
  if (!q) return;
  try {
    lastTried = await api("world/reason?q=" + encodeURIComponent(q));
    loadWorld();
  } catch (err) { addError(err.message); }
});

// ---------------------------------------------------------------- self-model panel
// Every line comes from the server with its evidence (event, fact, memory or turn numbers). Nothing is written here.
const SELF_SECTIONS = [
  ["current_state", "Current state", [["run", "Running"], ["conversation", "Conversation"], ["task", "Task"],
    ["focus", "Focus"], ["processing", "Processing"], ["uncertainty", "Uncertainty"]]],
  ["capabilities", "Capabilities", [["demonstrated", "Demonstrated"], ["failed", "Failed"], ["available", "Available"],
    ["unavailable", "Unavailable"], ["measured", "Measured (benchmark reports for the loaded brain file)"]]],
  ["experience", "Experience (newest first)", [["episodes", ""]]],
  ["uncertainty", "Uncertainty", [["knows", "Knows"], ["does_not_know", "Does not know"], ["unsure_about", "Unsure about"],
    ["inferred", "Based on inference"]]],
  ["history", "History", [["runs", "Runs"], ["changes", "Changes"], ["measurements", "Capability measurements"]]],
  ["environment", "Environment", [["inputs", "Inputs available"], ["outputs", "Outputs available"],
    ["observations", "Observations received"], ["actions", "Actions taken"]]],
  ["internal_state", "Internal state (numbers, read from the internal state)", [["current", "Current"],
    ["changes", "Recent changes and their causes"], ["baselines", "Baselines"], ["expectations", "Expectations"]]],
];
let lastAsked = null;

function evidenceText(evidence) {
  return "evidence: " + (evidence || []).map((e) => {
    const [kind, value] = Object.entries(e)[0];
    return kind + " " + value;
  }).join(", ");
}

function selfItem(it) {
  const box = el("div", "k-fact");
  box.append(el("div", "k-fact-line", it.text));
  if (it.actions && it.actions.length) box.append(el("div", "mem-meta", "did: " + it.actions.join("; ")));
  if (it.results && it.results.length) box.append(el("div", "mem-meta", "result: " + it.results.join("; ")));
  if (it.lessons && it.lessons.length) box.append(el("div", "mem-meta", "record after this: " + it.lessons.join("; ")));
  box.append(el("div", "mem-meta s-evidence", evidenceText(it.evidence)));
  return box;
}

function selfReportView(report, brainReply) {
  const box = el("div");
  if (!report.understood) {
    box.append(el("p", "plain", report.note));
    return box;
  }
  box.append(el("p", "plain", report.title + ": " + (report.items.length ? report.items.length + " recorded item" +
    (report.items.length === 1 ? "" : "s") : "nothing recorded") + ". " + report.note));
  report.items.forEach((it) => box.append(selfItem(it)));
  if (brainReply) {
    box.append(el("p", "plain", "Measured only: " + (brainReply.said_recorded_state === null ? brainReply.note :
      "the brain's own reply " + (brainReply.said_recorded_state ? "mentions " + brainReply.mentioned.join(", ") :
        "mentions none of the " + brainReply.checked + " recorded values or counts checked") + ".")));
  }
  return box;
}

async function loadSelf() {
  if (!info || !info.self_model) return;
  try {
    const [data, log] = await Promise.all([api("self"), api("self/events")]);
    renderSelf(data, log.events);
  } catch (err) {
    selfList.replaceChildren(el("p", "memory-empty", err.message));
  }
}

async function loadSelfCount() {
  try { selfCount.textContent = String((await api("self")).record.events); } catch (err) { /* shown when opened */ }
}

function renderSelf(data, events) {
  const r = data.record;
  selfCount.textContent = String(r.events);
  document.getElementById("self-info").textContent = r.events + " events in " + r.runs + (r.runs === 1 ? " run" : " runs") +
    ", " + r.messages + " messages (" + r.database + ")." + (r.calibration.predictions ? " Expected-success estimates: " +
    r.calibration.predictions + " checked, calibration error " + r.calibration.ece + "." : "");
  selfList.replaceChildren();
  if (lastAsked) {
    selfList.append(el("h3", "", "Your question: " + lastAsked.question));
    selfList.append(selfReportView(lastAsked));
  }
  SELF_SECTIONS.forEach(([key, title, groups]) => {
    selfList.append(el("h3", "", title));
    let any = false;
    groups.forEach(([group, label]) => {
      const items = data[key][group] || [];
      if (!items.length) return;
      any = true;
      if (label) selfList.append(el("div", "k-connections", label));
      items.forEach((it) => selfList.append(selfItem(it)));
    });
    if (!any) selfList.append(el("p", "memory-empty", "Nothing recorded yet."));
  });
  selfList.append(el("h3", "", "Event log (newest " + events.length + ")"));
  const pre = el("pre", "mono s-log");
  pre.textContent = events.map((e) => "#" + e.id + "  " + when(e.t) + "  " + e.kind + "  (" + e.source + ")" +
    (e.cause ? "  caused by #" + e.cause : "") + (Object.keys(e.refs).length ? "  refs " + JSON.stringify(e.refs) : "") +
    (Object.keys(e.data).length ? "  " + JSON.stringify(e.data) : "")).join("\n");
  selfList.append(pre);
}

function setSelfPanel(open) {
  selfPanel.hidden = !open;
  selfBtn.setAttribute("aria-expanded", String(open));
  if (open) {
    if (!memoryPanel.hidden) { memoryPanel.hidden = true; memoryBtn.setAttribute("aria-expanded", "false"); }
    if (!knowledgePanel.hidden) { knowledgePanel.hidden = true; knowledgeBtn.setAttribute("aria-expanded", "false"); }
    if (!worldPanel.hidden) { worldPanel.hidden = true; worldBtn.setAttribute("aria-expanded", "false"); }
    if (!statePanel.hidden) setStatePanel(false);
    loadSelf();
  }
}

selfBtn.addEventListener("click", () => setSelfPanel(selfPanel.hidden));
document.getElementById("self-close").addEventListener("click", () => setSelfPanel(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !selfPanel.hidden) setSelfPanel(false); });
document.getElementById("self-try").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = document.getElementById("self-question").value.trim();
  if (!q) return;
  try {
    lastAsked = await api("self/ask?q=" + encodeURIComponent(q));
    loadSelf();
  } catch (err) { addError(err.message); }
});
document.getElementById("self-clear").addEventListener("click", async () => {
  if (!window.confirm("Delete the event log? The self-model will start again from this run. Memory and knowledge are kept.")) return;
  try { await api("self/clear", { confirm: true }); lastAsked = null; loadSelf(); } catch (err) { addError(err.message); }
});

// ---------------------------------------------------------------- internal state panel
// Only numbers, their definitions, and their causes with evidence (event ids). No first-person text anywhere.
const STATE_DIMS = [["valence", "Valence"], ["activation", "Activation"], ["surprise", "Prediction error (surprise)"],
  ["novelty", "Novelty"], ["uncertainty", "Uncertainty"], ["information_need", "Information need"],
  ["learning_gain", "Learning gain"]];
const STATE_COLORS = { valence: "#4f46e5", activation: "#d97706", surprise: "#dc2626", novelty: "#059669" };

function signed(x) { return (x >= 0 ? "+" : "") + x.toFixed(2); }

function stateBar(value, range) {
  const [lo, hi] = range;
  const bar = el("div", "s-bar");
  const fill = el("div", "s-fill" + (value < 0 ? " neg" : ""));
  if (lo < 0) {                                   // centred: -1 .. 1
    const half = Math.min(Math.abs(value) / hi, 1) * 50;
    fill.style.left = (value < 0 ? 50 - half : 50) + "%";
    fill.style.width = half + "%";
    bar.append(el("div", "s-zero"));
  } else {
    fill.style.left = "0";
    fill.style.width = Math.max(0, Math.min(1, (value - lo) / (hi - lo))) * 100 + "%";
  }
  bar.append(fill);
  return bar;
}

function changeText(change) {
  const parts = Object.entries(change || {}).filter(([, v]) => Math.abs(v) >= 0.005)
    .map(([k, v]) => k.replace("_", " ") + " " + signed(v));
  return parts.length ? parts.join(", ") : "no change above 0.005";
}

function stateChangeView(s) {
  const box = el("div");
  box.append(el("p", "plain", "Change: " + changeText(s.change) + ". Salience " + s.appraisal.salience.toFixed(2) + "."));
  const list = el("div", "r-steps");
  s.causes.forEach((c) => list.append(el("div", "", c)));
  box.append(list);
  const fx = s.effects || {};
  (fx.tagged || []).forEach((t) => box.append(el("p", "plain", "Memory " + t.memory + " tagged as significant (salience " +
    t.salience.toFixed(2) + "): importance +" + t.importance_added.toFixed(2) + ".")));
  (fx.reinstated || []).forEach((r) => box.append(el("p", "plain", "Recalled memory " + r.memory +
    " moved the state toward its state at encoding (tag strength " + r.strength.toFixed(2) + ").")));
  box.append(el("div", "mem-meta s-evidence", "evidence: events " + s.evidence.join(", ") + " " + DOT +
    " history row " + s.history_row));
  return box;
}

function stateChart(points) {
  // Built only from numbers (no text from anywhere), so setting the markup is safe; the HTML parser gives <svg> its
  // namespace itself.
  const wrap = el("div");
  const y = (v, centred) => (centred ? 45 - v * 40 : 85 - v * 80).toFixed(1);
  let lines = "";
  if (points.length >= 2) {
    Object.entries(STATE_COLORS).forEach(([dim, color]) => {
      const pts = points.map((p, i) => (i * 300 / (points.length - 1)).toFixed(1) + "," +
        y(Number(p[dim]) || 0, dim === "valence")).join(" ");
      lines += '<polyline points="' + pts + '" fill="none" stroke="' + color + '" stroke-width="1.5"></polyline>';
    });
  }
  wrap.innerHTML = '<svg viewBox="0 0 300 90" class="s-chart" role="img" aria-label="Valence, activation, surprise ' +
    'and novelty after each of the last episodes"><line x1="0" x2="300" y1="45" y2="45" class="s-axis"></line>' +
    lines + "</svg>";
  return wrap;
}

async function loadState() {
  if (!info || !info.internal_state) return;
  try { renderState(await api("state")); } catch (err) { stateList.replaceChildren(el("p", "memory-empty", err.message)); }
}

function renderState(d) {
  stateList.replaceChildren();
  stateList.append(el("h3", "", "Current state"));
  STATE_DIMS.forEach(([key, label]) => {
    const row = el("div", "k-fact");
    const head = el("div", "k-fact-line");
    head.append(el("strong", "", label + ": " + (d.ranges[key][0] < 0 ? signed(d.now[key]) : d.now[key].toFixed(2))));
    if (key in d.baselines && (key === "valence" || key === "activation")) {
      head.append(el("span", "k-rel", "  (baseline " + signed(d.baselines[key]) + ")"));
    }
    row.append(head, stateBar(d.now[key], d.ranges[key]), el("div", "mem-meta", d.definitions[key] +
      (d.half_life_seconds[key] ? " " + DOT + " half-life " + Math.round(d.half_life_seconds[key] / 60) + " min" : "")));
    stateList.append(row);
  });
  stateList.append(el("h3", "", "History (after each episode)"));
  stateList.append(stateChart(d.chart));
  const legend = el("p", "mem-meta", "");
  Object.entries(STATE_COLORS).forEach(([dim, color]) => {
    const sw = el("span", "s-swatch");
    sw.style.background = color;
    legend.append(sw, " " + dim + "   ");
  });
  stateList.append(legend);
  stateList.append(el("h3", "", "Recent state changes"));
  if (!d.history.length) stateList.append(el("p", "memory-empty", "No changes yet: the state starts neutral."));
  d.history.forEach((h) => {
    const box = el("div", "k-fact");
    box.append(el("div", "k-fact-line", when(h.t) + ": " + (h.kind === "episode" ? changeText(h.change) : h.kind + " (by you)")));
    h.causes.forEach((c) => box.append(el("div", "mem-meta", "cause: " + c)));
    if (h.events.length) box.append(el("div", "mem-meta s-evidence", "evidence: events " + h.events.join(", ")));
    stateList.append(box);
  });
  stateList.append(el("h3", "", "Expectations (what \"expected\" means)"));
  if (!d.expectations.length) stateList.append(el("p", "memory-empty", "None yet: every outcome starts at 0.50."));
  d.expectations.forEach((x) => stateList.append(el("div", "mem-meta", x.label + ": expected " + x.p.toFixed(2) +
    " after " + x.n + " outcomes " + DOT + " learning gain " + x.gain.toFixed(2) + " " + DOT + " uncertainty " +
    x.uncertainty.toFixed(2))));
  stateList.append(el("h3", "", "Memories tagged as significant"));
  if (!d.tags.length) stateList.append(el("p", "memory-empty", "None: only memories formed in a significant episode are tagged."));
  d.tags.forEach((t) => stateList.append(el("div", "mem-meta", "memory " + t.memory + ": " + t.summary + " " + DOT +
    " salience " + t.salience.toFixed(2) + ", strength now " + t.strength_now.toFixed(2) + ", importance +" +
    t.importance_added.toFixed(2) + " (event " + t.event + ")")));
  if (d.gaps.length) {
    stateList.append(el("h3", "", "Open gaps (unknown questions not yet answered)"));
    d.gaps.forEach((g) => stateList.append(el("div", "mem-meta", g.missing + " (asked " + when(g.opened) + ", event " + g.event + ")")));
  }
  const sw = d.switches;
  document.getElementById("state-info").textContent = d.record.history_rows + " history rows (" + d.record.database + ") " +
    DOT + " baselines adapt: " + (sw.baseline_adaptation ? "on" : "off") + " " + DOT + " recall order: " +
    (sw.recall_order ? "on" : "off") + " " + DOT + " importance at encoding: " + (sw.importance_at_encoding ? "on" : "off");
}

function setStatePanel(open) {
  statePanel.hidden = !open;
  stateBtn.setAttribute("aria-expanded", String(open));
  if (open) {
    if (!memoryPanel.hidden) { memoryPanel.hidden = true; memoryBtn.setAttribute("aria-expanded", "false"); }
    if (!knowledgePanel.hidden) { knowledgePanel.hidden = true; knowledgeBtn.setAttribute("aria-expanded", "false"); }
    if (!worldPanel.hidden) { worldPanel.hidden = true; worldBtn.setAttribute("aria-expanded", "false"); }
    if (!selfPanel.hidden) { selfPanel.hidden = true; selfBtn.setAttribute("aria-expanded", "false"); }
    loadState();
  }
}

stateBtn.addEventListener("click", () => setStatePanel(statePanel.hidden));
document.getElementById("state-close").addEventListener("click", () => setStatePanel(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !statePanel.hidden) setStatePanel(false); });
document.getElementById("state-reset-baselines").addEventListener("click", async () => {
  if (!window.confirm("Reset the baselines to 0? The current values, expectations and history are kept.")) return;
  try { await api("state/reset", { what: "baselines", confirm: true }); loadState(); } catch (err) { addError(err.message); }
});
document.getElementById("state-reset-all").addEventListener("click", async () => {
  if (!window.confirm("Reset the whole internal state (values, expectations, tags, history)? Memory and knowledge are kept.")) return;
  try { await api("state/reset", { what: "everything", confirm: true }); loadState(); } catch (err) { addError(err.message); }
});

// ---------------------------------------------------------------- light / dark
function isDark() {
  const chosen = document.documentElement.dataset.theme;
  if (chosen) return chosen === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function applyTheme(theme) {
  if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;
}

try { applyTheme(localStorage.getItem("theme")); } catch (e) { /* storage unavailable */ }

themeBtn.addEventListener("click", () => {
  const next = isDark() ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
});

// ---------------------------------------------------------------- events + start
form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send();
  }
});
input.addEventListener("input", () => { autoGrow(); updateSendButton(); });
clearBtn.addEventListener("click", clearConversation);

async function start() {
  try {
    info = await api("info");
    infoEl.textContent = "Stage " + info.stage + " " + DOT + " " + info.model + " " + DOT + " " +
      info.parameters.toLocaleString() + " parameters " + DOT + " trained " + info.trained_on;
  } catch (err) {
    infoEl.textContent = "Model information unavailable";
  }
  if (info && info.memory) loadMemories();
  else memoryBtn.hidden = true;
  if (info && info.learning) loadKnowledge();
  else knowledgeBtn.hidden = true;
  if (info && info.reasoning) loadWorld();
  else worldBtn.hidden = true;
  if (info && info.self_model) loadSelfCount();
  else selfBtn.hidden = true;
  if (!info || !info.internal_state) stateBtn.hidden = true;
  try {
    const data = await api("history");
    if (data.history.length === 0) showWelcome();
    data.history.forEach((m) => addMessage(m.role, m.text, m.details));
  } catch (err) {
    showWelcome();
    addError(err.message);
  }
  input.focus();
}

start();
