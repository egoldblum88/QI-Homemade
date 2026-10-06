"""
Controlled experiments without the brain: the REAL memory, learning, world model, event log (self-model) and
internal state, driven exactly as the chat engine drives them, with two differences:
    - the clock is controlled (so decay and timing are exact), and
    - the reply's verified checks (said / not said, complete / not) are chosen by the experiment, because there is
      no brain and no reply text at all (the internal state never sees reply words anyway).
Used by tests/test_internal_state.py and python -m internal_state.evaluate. Temporary folders only.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from chat.chat_engine import ChatEngine
from internal_state.system import InternalState
from learning.system import LearningSystem
from memory.system import MemorySystem
from self_model.system import SelfSystem
from world.system import WorldSystem


class _NoBrain:
    context = 0

    @staticmethod
    def num_parameters():
        return 0


class Sim:

    def __init__(self, folder=None, config=None, start=1_800_000_000.0, memory=True, learning=True, reasoning=True,
                 persist=True):
        self._tmp = tempfile.TemporaryDirectory(prefix="state_sim_") if folder is None else None
        self.folder = Path(self._tmp.name if folder is None else folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.t = start
        self.config, self.persist = config, persist
        self.memory = MemorySystem(self.folder / "memory.db") if memory else None
        self.learning = LearningSystem(self.folder / "knowledge.db") if learning else None
        self.world = (WorldSystem(self.learning.store, self.memory.store if self.memory else None)
                      if learning and reasoning else None)
        self.start_run()

    def clock(self):
        return self.t

    def start_run(self):
        """A run of the program (again after restart()): a new event-log run and a freshly loaded internal state."""
        self.self_model = SelfSystem(self.folder / "events.db", persist=self.persist)
        self.self_model.log.clock = self.clock
        self.engine = SimpleNamespace(memory=self.memory, learning=self.learning, world=self.world,
                                      self_model=self.self_model, stage="(no brain)", answer_format="status line",
                                      model=_NoBrain())
        self.self_model.start(self.engine)
        self.state = InternalState(self.folder / "internal_state.db", persist=self.persist, config=self.config,
                                   clock=self.clock)
        self.state.attach(self.engine)

    def restart(self, downtime=0.0):
        """Close the event log and the internal state and open them again (memory and knowledge stay open)."""
        self.self_model.stop()
        self.self_model.close()
        self.state.close()
        self.t += downtime
        self.start_run()

    def close(self):
        for part in (self.self_model, self.state, self.memory, self.learning):
            if part is not None:
                part.close()
        if self._tmp is not None:
            self._tmp.cleanup()

    def say(self, text, said=True, complete=True, attempts=1, advance=60.0):
        """One message, through the same steps as ChatEngine.reply, minus the brain. Returns the details."""
        self.t += advance
        found = self.memory.before_reply(text, bias=self.state.recall_bias()) if self.memory else None
        learned_found = self.learning.before_reply(text) if self.learning else None
        thought = self.world.think(text, "status line") if self.world else None
        used = (ChatEngine.reasoning_items(thought) if thought is not None and thought.answered
                else ChatEngine.recall_items(found, learned_found))
        formed = self.memory.after_reply(text, "", found["analysis"]) if self.memory else None
        ids = self.memory.recent_turn_ids if self.memory else []
        learned = (self.learning.after_reply(text, "", turn_id=ids[-2] if len(ids) >= 2 else None)
                   if self.learning else None)
        details = {"stopped_by": "it produced the <end> marker" if complete else "it reached the length limit without ending",
                   "attempts": attempts, "reply_tokens": 6, "chars_generated": 24,
                   "memory_turns_in_prompt": 2 * len([i for i in used if i.kind != "reasoning"])}
        if thought is not None and thought.answered:
            details["reasoning"] = {"expression": {"said": bool(said)}}
        self_details = self.self_model.after_reply(self.engine, text, [], "", details, thought, used, found, formed,
                                                   learned_found, learned)
        state = self.state.after_episode(self_details["events"][0], text)
        return {"self": self_details, "state": state, "thought": thought, "found": found, "used": used,
                "learned": learned, "formed": formed}

    def values(self):
        return self.state.values(self.t)[0]
