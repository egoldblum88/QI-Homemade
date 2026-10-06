"""
The chat engine: the only bridge between the chat page and our brain.

It works with either brain, and says honestly which one is answering:

  Stage 1    character prediction. The conversation is written as book dialogue
             ("alternating quoted paragraphs") and the model continues it. It sees the last
             32 characters and was never trained to reply.
  Stage 1.5  conditional reply generation. The conversation is written with markers:
                 <user> earlier message <ai> earlier reply <end> <user> your message <ai>
             and the model writes the reply until it produces <end>. It was trained on
             about 32,000 message -> reply examples and sees the last 256 characters.

What happens when you send a message:
  1. It is cleaned the same way the training text was (curly quotes -> straight, accents
     removed), and characters the model has never seen are removed.
  2. With long-term memory (memory/) and learning (learning/) on: "forget ..." is handled, and up to 3
     relevant memories or learned facts are found (the same sentence is used once). Each becomes an
     EARLIER TURN in the prompt: your own words from then, and the reply given then. Nothing else is
     added: no system prompt, no identity, no summary written by us.
     With the world model (world/, Stage 2c) on, a question its rules understand goes through
     interpretation -> retrieval -> reasoning first. Then the brain is shown the reasoning result
     instead of the recalled items: Stage 1.5b3, the facts the answer rests on as plain statements (nothing
     when unknown); Stage 1.5b4 and later, the status line ("known: ...", "unknown: ...") and ONLY your
     message after it, without earlier turns (D-071, D-072). The reply is still written by the network alone; whether it
     expressed the reasoning result is measured and shown, never corrected.
     A STATEMENT (D-079) is read before the reply on a throwaway copy of the knowledge store: what it told goes
     to a status-line brain the same way ("known: I do not have a dog.", alone with your message), and an
     unfinished message nothing understood gets "unknown: what you mean.". It is stored in step 5, as before.
  3. It is written in the brain's format (above) together with the last few turns.
  4. The model predicts one token at a time until it stops (see brain/generate.py).
  5. With memory on, YOUR message (never the reply) is checked for lasting information, and the
     exchange is saved in the conversation history (state/memory.db). With learning on, the same message
     goes through the learning pipeline (state/knowledge.db): new, confirmed, corrected, connected.
  6. The reply, and a record of how it was made, go back to the chat page: which words of your
     message got attention, and which memories were found (retrieval) apart from the reply itself
     (generation).
  7. With the self-model on (self_model/): AFTER the reply, the message and what it caused are added to the
     event log (references only), and a question about recorded state gets a report with its evidence. That
     report goes to the Details and the Self panel only: the brain never sees the self-model.
  8. With the internal state on (internal_state/): AFTER that, the episode is appraised and the state changes,
     with the event ids that caused it. Its only effects on later messages: the importance of memories formed in
     a significant episode, and a small bonus in the ORDER of already-relevant recalled memories. It never
     writes, chooses or edits text, and the brain never sees it.

Nothing here writes, chooses or edits the reply's words, except that a Stage 1 reply cut
off at the length limit is trimmed back to its last complete sentence. There is no
calculator and no lookup of answers: every reply comes from the neural network. Memory only
decides what the network is SHOWN; the network alone decides what to say.
"""

import re
from dataclasses import dataclass

import numpy as np

from brain.dialogue import encode_prompt
from brain.generate import generate, reply_room, reply_to
from tools.prepare_data import normalize_characters

MAX_MESSAGE_CHARS = 500
MAX_REPLY_CHARS = 200           # Stage 1 (Stage 1.5 uses brain/dialogue.py's limit)
MIN_TRIMMED_CHARS = 20          # Stage 1: a reply cut at the limit is trimmed back to its last . ! or ?
MAX_ATTEMPTS = 3                # if a reply has no letters at all, try again (reported in details)
PROMPT_TURNS = 6                # how many recent turns are written into the prompt
STAGE1_STOP = {'"', "\n"}       # Stage 1: closing quote = end of its line

# What each brain is, in plain words. "The network itself remembers nothing beyond its prompt": long-term
# memory (memory/) is a separate system that can put up to 3 recalled memories into that prompt.
REMEMBERS = ("the network itself remembers nothing beyond what is in its prompt (the last few turns, plus up to "
             "3 recalled memories or facts when long-term memory and the world model are on)")
CAPABILITIES = {
    "1": ("Stage 1: character prediction. It continues text from the last 32 characters, the way "
          "dialogue in old storybooks continues. It was never trained to reply to anything."),
    "1.5": ("Stage 1.5: conditional reply generation. It was trained on about {examples:,} "
            "message-to-reply examples (dialogue from public-domain books and plays, plus arithmetic "
            "practice made by our own program) and sees the last {context} characters. It predicts a likely "
            "reply one character at a time. It does not understand meaning, knows no facts beyond "
            "patterns in that text, " + REMEMBERS + ", and "
            "anything it says about itself is an echo of its training text, not a belief."),
    "1.5b": ("Stage 1.5b: conditional reply generation, trained on five separate data categories "
             "(book language, literary dialogue, neutral conversation conventions, arithmetic and "
             "input-dependent tasks, about {examples:,} examples, all human-written or made by our own "
             "program). It reads the last {context} subword tokens and predicts a likely reply one token at "
             "a time. It does not understand meaning, knows no facts beyond patterns in that data, "
             + REMEMBERS + ", and anything it says about itself is an echo of its training text, not a belief."),
    "1.5b2": ("Stage 1.5b, round 2: the same kind of network with relative (rotary) positions and a tied output "
              "layer, trained on six separate data categories, including questions whose answer is in an EARLIER "
              "turn (about {examples:,} examples, all human-written or made by our own program). It reads the last "
              "{context} subword tokens and predicts a likely reply one token at a time. It can copy some things from "
              "earlier in this conversation; it does not understand meaning, knows no facts beyond patterns in that "
              "data, " + REMEMBERS + ", and anything it says about itself is an echo of its training text, "
              "not a belief."),
    "1.5b3": ("Stage 1.5b, round 3: the round 2 network, trained further to USE information you gave earlier (your "
              "name, favorites, likes, pets, family, places, or saying you have not told it), the way long-term "
              "memory shows it. It reads the last {context} subword tokens and predicts a likely reply one token at a "
              "time. It has no favorites or opinions of its own: its training only repeats YOUR information back. It "
              "does not understand meaning, " + REMEMBERS + ", and anything it says about itself is an echo of its "
              "training text, not a belief."),
    "1.5b4": ("Stage 1.5b, round 4: the round 3 network, trained further to SAY a reasoning result. When the world "
              "model's reasoning answers a question, the network is shown that result (known, unknown, conflicting, "
              "uncertain, yes, no, or which one) with the facts it rests on, and it practised turning such results "
              "into sentences on made-up worlds. It does not reason or look anything up itself, does not understand "
              "meaning, " + REMEMBERS + ", and anything it says about itself is an echo of its training text, not a "
              "belief."),
    "1.5b4b": ("Stage 1.5b, round 4b: like round 4 (trained to SAY a reasoning result: known, unknown, conflicting, "
               "uncertain, yes, no, or which one), with practice on facts about YOU too (your name, age, home, work, "
               "likes). It does not reason or look anything up itself, does not understand meaning, " + REMEMBERS +
               ", and anything it says about itself is an echo of its training text, not a belief."),
    "1.5b4d": ("Stage 1.5b, round 4d: like round 4c, with practice on saying the self-model's answer to a question about "
               "recorded state (what happened, what was learned, what failed, or that nothing is recorded). It does not "
               "reason or look anything up itself, does not understand meaning, " + REMEMBERS + ", and anything it "
               "says about itself is an echo of its training text, not a belief."),
    "1.5b4c": ("Stage 1.5b, round 4c: like round 4b, with practice on NEGATIVE results too (what you said is not so: "
               "\"you do not have a ...\", \"no\"). It does not reason or look anything up itself, does not understand "
               "meaning, " + REMEMBERS + ", and anything it says about itself is an echo of its training text, not a "
               "belief."),
}


@dataclass
class RecallItem:
    """One thing recalled for the prompt: a memory or a learned fact. It is shown to the brain as an earlier
    turn: your words (text), then the reply given at the time (generated text, only the turn in between)."""
    kind: str               # memory | knowledge | reasoning (a fact an answer rests on, written as a statement)
                            # | statement (what your statement told, read before the reply, D-079)
    id: int
    text: str
    reply: str
    created: float
    score: float
    entities: list


def text_key(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


BETWEEN_TOKENS = 12        # the turn after a recalled sentence: the brain's own short reply to it
REASONING_TOKENS = 96      # the most a reasoning result may take in the prompt (recalled memories: 64)
STRUCTURED = {"reasoning", "statement", "self"}     # a structured line: a reasoning result, a statement read (D-079),
                                                    # or the self-model's answer (D-080)


def recalled_turns(items, reply_for=None):
    """Recalled items as earlier conversation turns, oldest first: [your words, a reply, ...].

    reply_for(item) gives the turn in between. The chat uses the brain's own reply to that sentence ALONE,
    generated now (ChatEngine.between): it never sees anything else, so it cannot carry an old value. (Until
    2026-10-01 the reply stored at the time was replayed; the learning benchmark found that a reply written
    before a correction could put the OLD value back in front of the brain, D-062.) Without reply_for (the
    Stage 1 brain), the stored reply is used, shortened to about 60 characters.
    """
    from memory.system import RECALLED_REPLY_CHARS
    turns = []
    for item in sorted(items, key=lambda i: i.created):
        reply = reply_for(item) if reply_for else item.reply
        if len(reply) > RECALLED_REPLY_CHARS:
            reply = reply[:RECALLED_REPLY_CHARS].rsplit(" ", 1)[0]
        turns += [item.text, reply]
    return turns


class ChatEngine:

    def __init__(self, model, tokenizer, record, temperature=0.8, top_k=20, seed=None, memory=None, learning=None,
                 world=None, self_model=None, brain_file=None, internal_state=None):
        self.model, self.tokenizer, self.record = model, tokenizer, record
        self.memory = memory                          # a memory.system.MemorySystem, or None (memory off)
        self.learning = learning                      # a learning.system.LearningSystem, or None (learning off)
        self.world = world                            # a world.system.WorldSystem, or None (reasoning off)
        self._between = {}                            # cache: a recalled sentence -> the brain's reply to it alone
        if str(record.get("stage", "")).startswith("1.5b") and record["stage"] in CAPABILITIES:
            self.stage = record["stage"]             # every described Stage 1.5b round, not a fixed list
        else:
            self.stage = "1.5" if tokenizer.specials else "1"
        self.answer_format = record.get("answer_format", "statements")    # how it reads a reasoning result
        self.self_format = record.get("self_format")      # "self line": trained to say the self-model's answer (D-080)
        self.resolve_references = True                   # a follow-up's it / its / she ... (D-081); off: as before
        self.temperature, self.top_k = temperature, top_k
        self.seed = seed
        self.history = []     # [{"role": "user" | "ai", "text": ..., "model_text": ..., "details": ...}]
        self.self_model = self_model                  # a self_model.system.SelfSystem, or None; NEVER shown to the brain
        if self_model is not None:
            self_model.start(self, brain_file)        # "run started": the running program, as it actually is
        # the internal state needs the event log (its evidence): without the self-model it is off. NEVER shown to the brain
        self.internal_state = internal_state if self_model is not None else None
        if self.internal_state is not None:
            self.internal_state.attach(self)

    # ---------------------------------------------------------------- the messages
    def prepare_message(self, message):
        """Clean a message into text the model can read. Returns (text, removed_chars)."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("Please type a message first.")
        if len(message) > MAX_MESSAGE_CHARS:
            raise ValueError(f"Messages can be at most {MAX_MESSAGE_CHARS} characters.")
        text, dropped = normalize_characters(message)
        if self.stage == "1":
            text = text.replace('"', "'")             # a " would end the quoted line early
        text = re.sub(r"\s+", " ", text).strip()      # one paragraph: no line breaks
        unknown = self.tokenizer.unknown_characters(text)
        text = "".join(ch for ch in text if ch not in unknown)
        removed = sorted(set(dropped) | set(unknown))
        if not re.search(r"[A-Za-z0-9]", text):
            raise ValueError("The model only knows English letters, digits and basic punctuation. "
                             "Try typing some words.")
        return text, removed

    def build_prompt(self, recalled=()):
        """Stage 1: the conversation as book dialogue, ending with an opening quote."""
        texts = list(recalled) + [turn["model_text"] for turn in self.history[-PROMPT_TURNS:]]
        lines = [f'"{t}"' for t in texts]
        return "\n\n" + "\n\n".join(lines) + '\n\n"'

    def reply(self, message):
        """Handle one user message. Returns {"reply": text, "details": {...}}."""
        text, removed = self.prepare_message(message)
        found = learned_found = formed = learned = None
        if self.memory is not None:                   # steps 1-2: forget, then retrieve
            bias = self.internal_state.recall_bias() if self.internal_state is not None else None    # M2: order only
            found = self.memory.before_reply(text, visible_turns=min(len(self.history), PROMPT_TURNS - 1), bias=bias)
        if self.learning is not None:
            learned_found = self.learning.before_reply(text)
        thought = self.world.think(text, self.answer_format) if self.world is not None else None   # reason
        reference = None                                        # a follow-up: "What is its name?" (D-081)
        if thought is not None and self.resolve_references:
            thought, reference = self.resolve(text, thought)
        if thought is not None and not thought.answered:        # a statement: read now, stored after (D-079)
            self.world.read_statement(thought, self.learning, self.answer_format)
        own = None                                              # a question about recorded state (D-080)
        if (self.self_format == "self line" and self.self_model is not None
                and not (thought is not None and (thought.answered or thought.statement is not None))):
            own = self.self_model.line(text)
        self.history.append({"role": "user", "text": message.strip(), "model_text": text})
        if thought is not None and thought.answered:  # only the facts the answer rests on (none if unknown)
            items = self.reasoning_items(thought)
        elif thought is not None and thought.statement is not None:
            items = self.statement_items(thought)
        elif own is not None:
            items = [RecallItem("self", 0, own.line, "", 0.0, 1.0, list(own.keys))]
        else:
            items = self.recall_items(found, learned_found)
        if self.stage == "1":
            used = items
            reply, details = self._reply_stage1(recalled_turns(used))
        else:
            reply, details, used = self._reply_stage15(items)
        details["removed_characters"] = removed
        used_texts = {text_key(i.text) for i in used}
        mentions = sorted({e for i in used for e in i.entities
                           if e and re.search(r"\b" + re.escape(e.lower()) + r"\b", reply.lower())})
        if self.memory is not None:                   # steps 5-6: form memories from YOUR message, save
            formed = self.memory.after_reply(text, reply, found["analysis"])
            details["memory"] = self.memory_details(found, formed, used_texts, mentions)
        if self.learning is not None:                 # learn from YOUR message (never from the reply)
            ids = self.memory.recent_turn_ids if self.memory is not None else []
            learned = self.learning.after_reply(text, reply, turn_id=ids[-2] if len(ids) >= 2 else None)
            details["knowledge"] = self.knowledge_details(learned_found, learned, used_texts, mentions)
        if thought is not None:                       # every step apart; did the reply express the result?
            details["reasoning"] = self.world.details(thought, reply, used if thought.answered else None)
            if thought.statement is not None:
                details["statement"] = self.world.statement_details(thought.statement, reply)
            details["referents"] = self.referents_of(thought)    # for the NEXT message's it / she (D-081)
        if reference is not None:
            details["reference"] = reference
        if own is not None:
            from self_model.say import self_said
            details["self_line"] = {"status": own.status, "form": own.form, "shown_to_brain": [own.line],
                                    "expression": self_said(own, reply)}
        if self.self_model is not None:               # recorded AFTER the reply, so it cannot change it (D-073)
            details["self"] = self.self_model.after_reply(self, text, removed, reply, details, thought, used, found,
                                                          formed, learned_found, learned)
        if self.internal_state is not None and details.get("self"):      # appraised AFTER the reply (D-074)
            details["state"] = self.internal_state.after_episode(details["self"]["events"][0], text)
        self.history.append({"role": "ai", "text": reply, "model_text": reply, "details": details})
        return {"reply": reply, "details": details}

    def resolve(self, text, thought):
        """A follow-up question with it / its / that / this / he / she / they (D-081): read as the question about the
        ONE fitting thing of the previous turn's structured result, and used only if reasoning then understands it
        (and the first reading did not, or read the pronoun itself as a thing: "its name"). Returns (thought,
        reference or None). Your message is not changed: history and memory keep it as you wrote it."""
        from world import discourse
        previous = self.history[-1] if self.history and self.history[-1]["role"] == "ai" else None
        about = (previous or {}).get("details", {}).get("referents") or []
        if not about or (thought.answered and not discourse.mentions_reference(thought)):
            return thought, None
        resolution = discourse.resolve(text, about)
        if resolution is None:
            return thought, None
        resolved = self.world.think(resolution.text, self.answer_format)
        if not resolved.answered or discourse.mentions_reference(resolved):
            return thought, None
        return resolved, {"original": text, "resolved": resolution.text,
                          "substitutions": [list(s) for s in resolution.substitutions],
                          "note": "read with the one fitting thing of the previous turn's result; your message is "
                                  "kept as you wrote it"}

    @staticmethod
    def referents_of(thought):
        """What this turn's structured result was about, kept in its details (nothing is stored elsewhere)."""
        from world.discourse import referents
        if thought.answered:
            return referents(thought.world, thought.frame.facts, thought.query)
        if thought.statement is not None:
            return thought.statement.referents
        return []

    @staticmethod
    def recall_items(found, learned_found):
        """Recalled memories and learned facts, best first. The same sentence can be both (a memory, and the
        fact learned from it): it is given to the brain once."""
        from memory.retrieval import MAX_RECALLED
        items = [RecallItem("memory", r.memory.id, r.memory.content, r.memory.reply_after, r.memory.created, r.score,
                            r.memory.entities) for r in (found["retrieved"] if found else [])]
        items += [RecallItem("knowledge", r.fact["id"], r.fact["text"], r.fact["reply_after"], r.fact["first_learned"],
                             r.score, [r.fact["value"]]) for r in (learned_found["recalled"] if learned_found else [])]
        best = {}
        for item in items:
            key = text_key(item.text)
            if key not in best or item.score > best[key].score:
                best[key] = item
        return sorted(best.values(), key=lambda i: -i.score)[:MAX_RECALLED]

    @staticmethod
    def reasoning_items(thought):
        """The statements the reasoning layer made from the facts an answer rests on, in order."""
        return [RecallItem("reasoning", i, sentence, "", float(i), 1.0, list(thought.frame.answer))
                for i, sentence in enumerate(thought.sentences or [])]

    @staticmethod
    def statement_items(thought):
        """What your statement told (D-079), as ONE structured line, shown like a reasoning result."""
        values = [e.value for e in thought.statement.facts]
        return [RecallItem("statement", 0, thought.statement.line, "", 0.0, 1.0, values)]

    def between(self, item):
        """The turn after a recalled sentence: the brain's own reply to that sentence alone (greedy, generated
        now, at most BETWEEN_TOKENS tokens). Model-generated text, used only as the turn in between.

        A reasoning result shown to a brain trained on the status line gets an EMPTY turn instead (D-071): its
        reply to the line alone can be a broken echo ("Your name is My name is My name"), which it then copied
        into the answer. Measured with sampling: names 97.2% -> 100%, never-trained examples 96.3% -> 96.7%."""
        if item.kind in STRUCTURED and self.answer_format == "status line":
            return ""
        key = text_key(item.text)
        if key not in self._between:
            result, _ = reply_to(self.model, self.tokenizer, [item.text], temperature=0, loop_penalty=1.0,
                                 max_chars=BETWEEN_TOKENS)
            self._between[key] = result.text.strip()
        return self._between[key]

    ANSWER_LAST = True        # D-072; False = the order before (kept for the measurement that compares them)
    ANSWER_ALONE = True       # D-072: the structured answer and your message only, no recent turns

    def answer_last(self, items):
        """When the reasoning layer answered, a brain trained on the status line is shown ONLY the structured
        answer and your message (D-072). Placed first, with the recent turns after it, the brain copied a name
        from those turns ("my name is etai" ... "Your name is Charlie."); in training only neutral turns ever
        stood there. Measured on long conversations (330 questions, nothing cleared): answer first 97.9%,
        answer last 97.6%, answer alone 98.8%; your name in a long chat 32/36, 36/36, 36/36.
        Recalled memories (and Stage 1.5b3's statements) keep their place before the recent turns."""
        return (self.ANSWER_LAST and self.answer_format == "status line" and bool(items)
                and all(i.kind in STRUCTURED for i in items))

    def fit_memories(self, items, context):
        """Recalled items + recent turns that fit the prompt. Older recent turns go first, then the least
        relevant recalled items: the current message is always kept whole."""
        used = sorted(items, key=lambda i: -i.score)
        budget = self.model.context - reply_room(self.model)
        size = lambda turns: len(encode_prompt(self.tokenizer, turns, 10**6))
        while used and size(recalled_turns(used, self.between)) - 1 > self.memory_token_limit(used):
            used = used[:-1]
        while size(recalled_turns(used, self.between) + context) > budget:
            if len(context) > 1:
                context = context[2:]                 # the oldest exchange of this session
            elif used:
                used = used[:-1]
            else:
                break
        return used, context

    @staticmethod
    def memory_token_limit(items=()):
        """Recalled items get at most 64 tokens; a reasoning result (one line, or the facts of a comparison or a
        description) up to 96, so it is never cut off: measured, its longest form is about 76."""
        from memory.system import MAX_MEMORY_TOKENS
        if items and all(i.kind in STRUCTURED for i in items):
            return REASONING_TOKENS
        return MAX_MEMORY_TOKENS

    def memory_details(self, found, formed, used_texts, mentions):
        """What memory did for this reply. Retrieval and the reply are reported separately."""
        retrieved = [{**r.to_public(), "in_prompt": text_key(r.memory.content) in used_texts}
                     for r in found["retrieved"]]
        return {
            "enabled": True,
            "database": self.memory.path.name,
            "retrieved": retrieved,
            "in_prompt": sum(r["in_prompt"] for r in retrieved),
            "reply_mentions": mentions,
            "forgotten": found["forgotten"],
            "contradicted": found.get("contradicted", []),
            "formed": formed["formed"],
            "replaced": formed["replaced"],
            "note": formed["note"] if not found["forgotten"] else found["analysis"].note,
        }

    def knowledge_details(self, learned_found, learned, used_texts, mentions):
        """What learning did for this reply: what was recalled (and given to the brain), what was learned."""
        recalled = [{**self.learning.learner.public(r.fact), "relevance": round(r.relevance, 3),
                     "parts": {k: round(v, 3) for k, v in r.parts.items()},
                     "in_prompt": text_key(r.fact["text"]) in used_texts} for r in learned_found["recalled"]]
        return {
            "enabled": True,
            "database": self.learning.path.name,
            "recalled": recalled,
            "in_prompt": sum(r["in_prompt"] for r in recalled),
            "reply_mentions": mentions,
            "forgotten": learned_found["forgotten"],
            "learned": learned["learned"],
            "confirmed": learned["confirmed"],
            "corrected": learned["corrected"],
            "changed": learned["changed"],
            "connected": learned["connected"],
            "note": learned["note"],
        }

    def _generate(self, make):
        """Run generation up to MAX_ATTEMPTS times until the reply contains a letter."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            seed = None if self.seed is None else self.seed + attempt
            result, prompt_ids = make(seed)
            if re.search(r"[A-Za-z0-9]", result.text):
                break
        return result, prompt_ids, attempt

    def _common_details(self, result, prompt_ids, attempt):
        probs = result.chosen_probabilities
        return {
            "stage": self.stage,
            "capability": self.capability(),
            "prompt": self.tokenizer.decode(prompt_ids, show_specials=True)[-700:],
            "context_window": self.tokenizer.decode(result.context_ids, show_specials=True),
            "context_ids": [int(i) for i in result.context_ids],
            "context_size": self.model.context,
            "steps": result.steps,
            "chars_generated": len(result.text),
            "reply_tokens": len(result.ids),
            # the average probability the MODEL gave to each token that was chosen. This is a
            # property of the next-token distribution, not a measure of confidence or correctness.
            "mean_token_probability": round(sum(probs) / len(probs), 3) if probs else 0.0,
            "attempts": attempt,
            "decoding": f"sampling, temperature {self.temperature}, top-k {self.top_k}",
            "temperature": self.temperature,
            "top_k": self.top_k,
            "elapsed_ms": round(result.elapsed_ms),
        }

    def _reply_stage1(self, recalled=()):
        prompt = self.build_prompt(recalled)

        def make(seed):
            return generate(self.model, self.tokenizer, prompt, max_chars=MAX_REPLY_CHARS,
                            temperature=self.temperature, top_k=self.top_k, stop=STAGE1_STOP,
                            seed=seed, record_steps=8), self.tokenizer.encode(prompt)

        result, prompt_ids, attempt = self._generate(make)
        reply = result.text.strip()
        stopped_by = {"stop marker": "it closed the quote or ended the paragraph",
                      "length limit": f"it reached the {MAX_REPLY_CHARS}-character limit"}[result.stopped_by]
        if result.stopped_by == "length limit":         # keep only complete sentences
            last_end = max(reply.rfind(ch) for ch in ".!?")
            if last_end + 1 >= MIN_TRIMMED_CHARS:
                reply = reply[:last_end + 1]
                stopped_by += " (trimmed back to the end of its last sentence)"
        details = self._common_details(result, prompt_ids, attempt)
        details["stopped_by"] = stopped_by
        return reply, details

    def _reply_stage15(self, items=()):
        context = [turn["model_text"] for turn in self.history[-PROMPT_TURNS:]]
        used = []
        if items:
            if len(context) % 2 == 0:
                context = context[1:]                 # recalled turns are followed by one of YOUR turns
            used, context = self.fit_memories(items, context)
        memory_turns = recalled_turns(used, self.between) if used else []
        if memory_turns and self.answer_last(used):
            recent = [] if self.ANSWER_ALONE else context[:-1]
            context = recent + memory_turns + context[-1:]             # the structured answer, then your question
        else:
            context = memory_turns + context

        def make(seed):
            return reply_to(self.model, self.tokenizer, context, temperature=self.temperature,
                            top_k=self.top_k, seed=seed, record_steps=8)

        result, prompt_ids, attempt = self._generate(make)
        details = self._common_details(result, prompt_ids, attempt)
        if result.stopped_by == "length limit":
            details["stopped_by"] = "it reached the length limit without ending"
        elif result.stop_token == "\n":
            details["stopped_by"] = "it started a new line"
        else:
            details["stopped_by"] = f"it produced the {result.stop_token} marker"
        recent_available = min(len(self.history), PROMPT_TURNS)
        details["turns_in_prompt"] = result.turns_used - len(memory_turns)    # nothing is dropped silently
        details["turns_available"] = recent_available
        details["memory_turns_in_prompt"] = len(memory_turns)
        details["recalled_turns"] = memory_turns                   # your sentences + the brain's replies to them
        details["answer_alone"] = bool(memory_turns) and self.answer_last(used) and self.ANSWER_ALONE
        details["message_cut"] = result.message_cut
        details.update(self.attention_to_message(prompt_ids, result.ids))
        return result.text.strip(), details, used

    def attention_to_message(self, prompt_ids, reply_ids):
        """The attention distribution over the words of your message, while the reply was written.

        We run the finished conversation through the model once more and read its attention
        weights (all layers and heads averaged) from the positions where it predicted the reply.
        A token's weight is shared by every word it overlaps. This shows where the computation
        drew information from. It is not evidence that anything was understood.
        """
        ids = (list(prompt_ids) + list(reply_ids))[-self.model.context:]
        offset = len(prompt_ids) + len(reply_ids) - len(ids)
        self.model.forward(np.array([ids]))
        maps = [self.model.attention_map(layer) for layer in range(len(self.model.blocks))]
        attention = np.mean(maps, axis=0)                              # (T, T)
        start_reply = len(prompt_ids) - 1 - offset                     # the <ai> position onwards
        rows = attention[max(start_reply, 0):]
        weights = rows.mean(axis=0)                                    # attention per position
        user_id = self.tokenizer.special("<user>")
        prompt_part = ids[:len(prompt_ids) - offset]
        last_user = max(i for i, t in enumerate(prompt_part) if t == user_id) if user_id in prompt_part else -1
        message_positions = list(range(last_user + 1, len(prompt_part) - 1))
        pieces = [self.tokenizer.decode([ids[i]]) for i in message_positions]   # 1+ characters each
        message = "".join(pieces)
        spans, start = [], 0
        for piece in pieces:
            spans.append((start, start + len(piece)))
            start += len(piece)
        share_on_message = float(weights[message_positions].sum()) if message_positions else 0.0
        words = []
        for match in re.finditer(r"[A-Za-z0-9']+", message):
            w = sum(float(weights[pos]) for pos, (s, e) in zip(message_positions, spans)
                    if s < match.end() and e > match.start())
            words.append([match.group(0), w])
        total = sum(w for _, w in words) or 1.0
        words = [[word, round(w / total, 3)] for word, w in words]
        return {"message_seen": message,
                "attention_words": sorted(words, key=lambda x: -x[1])[:5],
                "attention_by_word": words,
                "attention_on_message": round(share_on_message, 3)}

    # --------------------------------------------------------------- page helpers
    def clear(self):
        self.history = []

    def public_history(self):
        out = []
        for turn in self.history:
            item = {"role": turn["role"], "text": turn["text"]}
            if "details" in turn:
                item["details"] = turn["details"]
            out.append(item)
        return out

    def capability(self):
        return CAPABILITIES[self.stage].format(examples=self.record.get("dialogue_examples", 0),
                                               context=self.model.context)

    def info(self):
        record = self.record
        return {
            "stage": self.stage,
            "capability": self.capability(),
            "model": {"1": "character network (Stage 1)", "1.5": "attention network (Stage 1.5)",
                      "1.5b": "attention network with our subword tokens (Stage 1.5b)",
                      "1.5b2": "attention network, relative positions (Stage 1.5b round 2)",
                      "1.5b3": "attention network, trained to use what you told it (Stage 1.5b round 3)",
                      "1.5b4": "attention network, trained to say reasoning results (Stage 1.5b round 4)",
                      "1.5b4b": "attention network, trained to say reasoning results, incl. about you (round 4b)",
                      "1.5b4c": "attention network, trained to say reasoning results, incl. negative ones (round 4c)",
                      "1.5b4d": "attention network, trained to say reasoning and self-model results (round 4d)",
                      }[self.stage],
            "parameters": self.model.num_parameters(),
            "context": self.model.context,
            "vocab_size": self.tokenizer.vocab_size,
            "val_loss": record.get("val_reply_loss", record.get("val_loss")),
            "loss_kind": "reply loss" if self.stage == "1.5" else "validation loss",
            "best_counting_baseline": min(record.get("baselines", {"-": 0}).values()) if record.get("baselines") else None,
            "dialogue_examples": record.get("dialogue_examples"),
            "trained_on": record.get("trained_on"),
        }
