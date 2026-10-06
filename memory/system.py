"""
The memory system: how long-term memory and the neural brain work together in the chat.

For every message:
    1. "forget ..."  -> the matching memories are deleted FIRST, so the brain never sees them again.
                        A negated statement ("I don't have a dog", "My dog's name is not Max") sets aside the
                        memories it contradicts, also FIRST (D-075). Nothing is stored from it.
    2. RETRIEVAL     -> score every memory against the message; keep at most 3 relevant ones
    3. PROMPT        -> each recalled memory becomes an EARLIER TURN: your own words from then,
                        followed by the reply the brain gave then. (That is the format the Stage 1.5b
                        round 2 brain learned to use: information given in an earlier turn.)
                        No system prompt, no summary written by us, nothing about what the system is.
    4. OUR BRAIN     -> writes the reply (chat/chat_engine.py). Its weights are never changed here.
    5. FORMATION     -> YOUR message (never the reply) is checked for lasting information
    6. STORAGE       -> new memories and the conversation history are saved in SQLite

Retrieval success and reply quality are reported separately: the chat's "Details" shows which
memories were found (with their scores), and whether the reply used them.
"""

from memory import retrieval, writer
from memory.store import DEFAULT_DB, MemoryStore

MAX_MEMORY_TOKENS = 64          # at most 3 recalled items of the 128-token prompt (48 until learned facts, whose
                                # sentences are longer, were added: D-062)
RECALLED_REPLY_CHARS = 60       # a recalled reply is shortened, so it can never crowd out the memory itself


class MemorySystem:

    def __init__(self, path=DEFAULT_DB, brain_name=None, max_recalled=retrieval.MAX_RECALLED):
        self.store = MemoryStore(path)
        self.session = self.store.start_session(brain_name)
        self.max_recalled = max_recalled
        self.recent_turn_ids = []           # this session's turns, to avoid recalling what is already visible

    @property
    def path(self):
        return self.store.path

    def close(self):
        self.store.close()

    # ------------------------------------------------------------------ steps 1-2
    def before_reply(self, message, visible_turns=0, bias=None):
        """Act on "forget ...", then find the memories relevant to this message.

        visible_turns: how many recent turns of THIS session the prompt already shows; memories made from
        those turns are not recalled a second time.
        bias: the internal state's small ordering bonus for significant memories (None: off).
        """
        analysis = writer.analyze(message)
        forgotten = []
        if analysis.forget:
            matches = retrieval.retrieve(self.store.active(), analysis.forget, k=10,
                                         min_relevance=retrieval.FORGET_RELEVANCE)
            best = matches[0].relevance if matches else 0.0
            for r in matches:
                if r.relevance >= 0.8 * best:
                    self.store.delete(r.memory.id, why="you asked to forget it")
                    forgotten.append(r.memory.to_public())
            if not forgotten:
                analysis.note = "nothing matching that was stored, so nothing was forgotten"
        contradicted = []
        for n in analysis.negations:          # you said the opposite: those memories are no longer used
            for m in self.store.active():
                if m.key == n.key and (n.summary is None or m.summary.lower() == n.summary.lower()):
                    self.store.retire(m.id, f"you said: \"{n.sentence}\"")
                    contradicted.append(m.to_public())
        visible = set(self.recent_turn_ids[-visible_turns:]) if visible_turns else set()
        candidates = [m for m in self.store.active() if m.turn_id is None or m.turn_id not in visible]
        retrieved = retrieval.retrieve(candidates, message, k=self.max_recalled, bias=bias)
        self.store.mark_retrieved([r.memory.id for r in retrieved])
        return {"analysis": analysis, "forgotten": forgotten, "retrieved": retrieved, "contradicted": contradicted}

    @staticmethod
    def recalled_turns(retrieved):
        """Recalled memories as earlier conversation turns, oldest first: [your words, the reply then, ...].

        The reply is the model's own reply at the time (generated text, used only as the turn in between),
        shortened to about 60 characters at a word boundary.
        """
        turns = []
        for r in sorted(retrieved, key=lambda r: r.memory.created):
            reply = r.memory.reply_after
            if len(reply) > RECALLED_REPLY_CHARS:
                reply = reply[:RECALLED_REPLY_CHARS].rsplit(" ", 1)[0]
            turns += [r.memory.content, reply]
        return turns

    # ------------------------------------------------------------------ steps 5-6
    def after_reply(self, message, reply, analysis=None):
        """Save the exchange in the conversation history, and store any lasting information from YOUR message."""
        user_turn = self.store.add_turn(self.session, "user", message)
        ai_turn = self.store.add_turn(self.session, "ai", reply)
        self.recent_turn_ids += [user_turn, ai_turn]
        analysis = analysis or writer.analyze(message)
        formed, replaced = [], []
        if not analysis.forget:
            for c in analysis.candidates:
                before = {m.id for m in self.store.active()}
                memory = self.store.add(c.type, c.content, c.summary, c.source, c.confidence, c.importance, c.reason,
                                        key=c.key, subject=c.subject, entities=c.entities, turn_id=user_turn,
                                        reply_after=reply)
                formed.append(memory.to_public())
                replaced += sorted(before - {m.id for m in self.store.active()})
        return {"formed": formed, "replaced": replaced, "note": "" if formed else analysis.note}

    # ------------------------------------------------------------------ your controls
    def memories(self):
        return [m.to_public() for m in self.store.active()]

    def why(self, memory_id):
        """Everything about how a memory came to exist and how it has been used."""
        memory = self.store.get(memory_id)
        if memory is None:
            return None
        turn = self.store.turn(memory.turn_id) if memory.turn_id else None
        return {**memory.to_public(),
                "your_message": turn["text"] if turn else "(the conversation history was deleted)",
                "reply_at_the_time": memory.reply_after,
                "log": self.store.log(memory_id)}

    def delete(self, memory_id):
        return self.store.delete(memory_id)

    def clear(self):
        return self.store.clear()

    def set_important(self, memory_id, important):
        return self.store.set_pinned(memory_id, important)

    def history_counts(self):
        return self.store.history_counts()

    def clear_history(self):
        self.recent_turn_ids = []
        return self.store.clear_history()
