"""
The internal-state benchmark:   python -m internal_state.evaluate [--users 40] [--skip-memory] [--brain ...]

Controlled experiments on temporary databases (state/ is never touched). Most run WITHOUT the brain (the real memory,
learning, world model, event log and internal state, with a controlled clock: internal_state/simulate.py). Two parts
use the real brain: the memory benchmark comparison (which decides whether the recall-order bonus stays on by
default) and an integration run through the chat engine. Nothing is trained.

The criteria were fixed before building (docs/07_INTERNAL_STATE.md section 7.2):
  consistency       identical event histories give identical states (100%)
  decay             each half-life within 1% of its setting; within 0.01 of baseline after 7 half-lives
  context           the same expression failure after 10 successes vs after 10 failures: |change in valence| >= 3x;
                    the same "unknown" answer where answers are usually known vs usually unknown: surprise >= 0.3 apart
  memory            novel episodes tagged >= 90%, repeated ones <= 10%; the tagged memory first among equally relevant
                    ones 100%; recall moves valence toward the encoding direction 100%; rumination guard holds
  learning          learning gain after a surprise > in a stable run (100%); uncertainty falls over 10 consistent outcomes
  reasoning         unexpected success raises valence, unexpected failure lowers it (100%)
  persistence       after a restart: the saved state decayed for the downtime, error < 1e-6
  baselines         repeated success vs repeated unexpected failure: baselines >= 0.1 apart and different responses to
                    the same probe; with adaptation off, baselines stay 0
  false state       generated words change nothing; non-eligible events change nothing; every change has evidence
  blank slate       0 findings in the code and the stored state
  no regression     memory benchmark with the internal state on: every number within 1 point of state-off. This
                    decides whether the two effects that change what the brain is shown (M1 importance at encoding,
                    M2 recall order) are on by default. As shipped (decided by this measurement: both off), the
                    brain's prompts and replies are identical with the state on and off
Report: models/internal_state_evaluation.md
"""

import argparse
import math
import tempfile
import time
from pathlib import Path

from brain.tasks import made_up_words
from internal_state import blank_slate
from internal_state import state as dynamics
from internal_state.config import DEFAULTS
from internal_state.simulate import Sim
from internal_state.system import DIMENSIONS

MEMORY_EFFECTS = {"memory": {"importance": True, "recall_order": True}}      # M1 + M2 on (off by default, D-074)
ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "models" / "internal_state_evaluation.md"
NAMES = made_up_words(400, 2026, capitalize=True)
ROLES = ["cat", "bird", "horse", "fish", "goat", "duck", "hamster", "pony", "frog", "rabbit", "lizard", "parrot",
         "turtle", "mouse", "snake", "sheep", "cow", "pig", "owl", "crab"]
EPISODIC = ["I won my game yesterday.", "I visited the museum today.", "I fixed my bike this morning.",
            "I baked bread last night.", "I painted the fence on Sunday.", "I moved to a new desk today.",
            "I watched the rain this afternoon.", "I cleaned the attic yesterday."]
YESTERDAY = ["I won my game yesterday.", "I visited the museum yesterday.", "I fixed my bike yesterday.",
             "I baked bread yesterday.", "I painted the fence yesterday.", "I moved my desk yesterday.",
             "I watched the rain yesterday.", "I cleaned the attic yesterday.", "I found a coin yesterday.",
             "I lost my keys yesterday."]
NOT_YESTERDAY = EPISODIC[1:7]          # events on other days: they make later events less new (habituation)


class Results:
    def __init__(self):
        self.rows = []

    def add(self, group, what, ok, measured):
        self.rows.append((group, what, bool(ok), measured))
        print(f"  {'PASS' if ok else 'FAIL'}  {group}: {what} ({measured})")


def history_signature(sim):
    return [(h["episode"], h["appraisal"], h["before"], h["after"], h["baselines"])
            for h in sim.state.store.history(newest_first=False)]


SCRIPT = [("My dog's name is {0}.", {}), ("What is my dog's name?", {}), ("What is my cat's name?", {}),
          ("What is my dog's name?", {"said": False}), ("My cat's name is {1}.", {}), ("Hello there.", {"complete": False}),
          ("Actually, my dog's name is {2}.", {}), ("What is my dog's name?", {})]


def run_script(sim, offset=0):
    for text, kw in SCRIPT:
        sim.say(text.format(*NAMES[offset:offset + 3]), **kw)


def consistency(r):
    a, b = Sim(), Sim()
    run_script(a)
    run_script(b)
    same = history_signature(a) == history_signature(b) and a.values() == b.values()
    r.add("consistency", "identical histories give identical states", same, f"{len(history_signature(a))} episodes compared")
    a.close()
    b.close()


def decay(r):
    s = Sim()
    s.say(f"My dog's name is {NAMES[3]}.", complete=False)
    start, t0, base = s.values(), s.t, s.state.values()[1]
    worst, back = 0.0, 0.0
    for dim, half in DEFAULTS["half_life_seconds"].items():
        s.t = t0 + half
        measured = half / math.log2((start[dim] - base[dim]) / (s.values()[dim] - base[dim]))
        worst = max(worst, abs(measured - half) / half)
        s.t = t0 + 7 * half
        back = max(back, abs(s.values()[dim] - base[dim]))
    r.add("decay", "each half-life within 1% of its setting", worst <= 0.01, f"largest error {100 * worst:.4f}%")
    r.add("decay", "within 0.01 of baseline after 7 half-lives", back <= 0.01, f"largest distance {back:.4f}")
    s.close()


def context(r):
    def failure_after(said):
        s = Sim()
        s.say(f"My dog's name is {NAMES[4]}.")
        for _ in range(10):
            s.say("What is my dog's name?", said=said)
        change = s.say("What is my dog's name?", said=False)["state"]["change"]["valence"]
        s.close()
        return change
    a, b = failure_after(True), failure_after(False)
    ratio = abs(a) / max(abs(b), 1e-9)
    r.add("context", "the same expression failure: |change| after successes >= 3x after failures", ratio >= 3 and a < 0,
          f"{a:+.3f} vs {b:+.3f}: {ratio:.1f}x")

    def unknown_where(known):
        s = Sim()
        if known:
            s.say(f"My dog's name is {NAMES[5]}.")
            for _ in range(10):
                s.say("What is my dog's name?")
        else:
            for role in ROLES[:10]:
                s.say(f"What is my {role}'s name?")
        d = s.say("What is my unicorn's name?")["state"]
        s.close()
        return next(abs(c["delta"]) for c in d["appraisal"]["checks"] if c["key"].startswith("answerable"))
    x, y = unknown_where(True), unknown_where(False)
    r.add("context", "the same unknown answer: surprise where usually known vs usually unknown >= 0.3 apart",
          x - y >= 0.3, f"{x:.3f} vs {y:.3f}")


def memory(r):
    tagged_new = tagged_repeat = n_new = n_repeat = 0
    for trial in range(20):
        s = Sim(config=MEMORY_EFFECTS)
        d = s.say(f"My {ROLES[trial]}'s name is {NAMES[10 + trial]}.")["state"]
        n_new += 1
        tagged_new += bool(d["effects"]["tagged"])
        for _ in range(3):
            d = s.say(f"My {ROLES[trial]}'s name is {NAMES[10 + trial]}.")["state"]
        n_repeat += 1
        tagged_repeat += bool(d["effects"]["tagged"])
        s.close()
    r.add("memory", "novel episodes tagged >= 90%", tagged_new / n_new >= 0.9, f"{tagged_new} of {n_new}")
    r.add("memory", "repeated episodes tagged <= 10%", tagged_repeat / n_repeat <= 0.1, f"{tagged_repeat} of {n_repeat}")

    first = trials = 0
    for trial in range(10):
        s = Sim(config=MEMORY_EFFECTS)                          # the mechanism, switched on (off by default)
        s.say(YESTERDAY[trial])                                 # new: tagged
        for m in NOT_YESTERDAY:
            s.say(m)                                            # habituation: later events are no longer new
        s.say(YESTERDAY[(trial + 5) % 10])                      # as relevant to the question, and not tagged
        bias = s.state.recall_bias()
        found = s.memory.before_reply("What happened yesterday?", bias=bias)["retrieved"]
        tagged = set(s.state.store.tags())
        trials += 1
        first += bool(found) and found[0].memory.id in tagged
        s.close()
    r.add("memory", "a tagged memory is recalled first among equally relevant ones (M1 + M2 switched on)",
          first == trials, f"{first} of {trials}")

    right = total = 0
    for complete in (False, True):                             # negative and positive encodings
        for trial in range(5):
            out = {}
            for reinstate in (True, False):
                s = Sim(config={"memory": {"reinstate": reinstate}})
                for _ in range(8):
                    s.say("Hello there.", complete=not complete)    # sets up the expectation
                s.say(YESTERDAY[trial], complete=complete)          # unexpected: negative or positive at encoding
                v_at = s.state.store.tags()[1]["valence"]
                s.t += 7200
                out[reinstate] = (v_at, s.say("What happened yesterday?")["state"]["after"]["valence"])
                s.close()
            v_at, with_recall = out[True]
            total += 1
            right += (with_recall - out[False][1]) * v_at > 0       # moved toward the encoding's sign
    r.add("memory", "recall moves valence toward the encoding direction", right == total, f"{right} of {total}")

    s = Sim()
    for _ in range(8):
        s.say("Hello there.")
    s.say(EPISODIC[0], complete=False)
    v_at = s.state.store.tags()[1]["valence"]
    lowest, strengths = 1.0, []
    for _ in range(50):
        s.t += 1900
        d = s.say("What happened yesterday?")["state"]
        lowest = min(lowest, d["after"]["valence"])
        strengths += [x["strength"] for x in d["effects"]["reinstated"]]
    s.t += 7 * DEFAULTS["half_life_seconds"]["valence"]
    settled = abs(s.values()["valence"] - s.state.values()[1]["valence"])
    ok = lowest >= v_at - 0.5 and strengths == sorted(strengths, reverse=True) and settled < 0.01
    r.add("memory", "rumination guard: 50 recalls never run away, the tag fades, decay returns to baseline", ok,
          f"lowest valence {lowest:+.3f} (encoding {v_at:+.3f}); back within {settled:.4f}")
    s.close()


def learning(r):
    higher = probes = 0
    for trial in range(5):
        s = Sim()
        s.say(f"My dog's name is {NAMES[40 + trial]}.")
        for _ in range(10 + trial):
            s.say("What is my dog's name?")
        stable = s.state.expectations.predict("express:known")[1]
        s.say("What is my dog's name?", said=False)
        probes += 1
        higher += s.state.expectations.predict("express:known")[1] > stable
        s.close()
    r.add("learning", "learning gain after a surprise > in a stable run", higher == probes, f"{higher} of {probes}")
    s = Sim()
    s.say(f"My dog's name is {NAMES[50]}.")
    u = [s.say("What is my dog's name?")["state"]["after"]["uncertainty"] for _ in range(10)]
    falls = all(a > b for a, b in zip(u, u[1:]))
    r.add("learning", "uncertainty falls over 10 consistent outcomes", falls, f"{u[0]:.3f} -> {u[-1]:.3f}")
    s.close()


def reasoning(r):
    right = total = 0
    for trial in range(5):
        s = Sim()
        s.say(f"My dog's name is {NAMES[60 + trial]}.")
        for _ in range(8):
            s.say("What is my dog's name?", said=False)
        total += 1
        right += s.say("What is my dog's name?", said=True)["state"]["change"]["valence"] > 0
        for _ in range(8):
            s.say("What is my dog's name?", said=True)
        total += 1
        right += s.say("What is my dog's name?", said=False)["state"]["change"]["valence"] < 0
        s.close()
    r.add("reasoning", "unexpected success raises valence, unexpected failure lowers it", right == total,
          f"{right} of {total}")


def persistence(r):
    s = Sim()
    run_script(s, offset=70)
    saved, baselines, t = s.state.stored()
    s.restart(downtime=1800)
    expected = dynamics.decay(saved, baselines, s.t - t, s.state.cfg)
    error = max(abs(s.values()[d] - expected[d]) for d in dynamics.DECAYING)
    r.add("persistence", "after a restart: the saved state decayed for the downtime", error < 1e-6, f"error {error:.2e}")
    s.close()


def development(r):
    """The experimental part: do repeated experiences leave a lasting difference (baselines, response to a probe)?"""
    def history(kind, adapt=True):
        s = Sim(config={"baselines": {"adapt": adapt}})
        s.say(f"My dog's name is {NAMES[100]}.")
        for i in range(50):
            if kind == "success":            # gaps opened and resolved, everything works: 4 episodes per cycle
                s.say(f"What is the capital of {NAMES[200 + i]}?")
                s.say(f"The capital of {NAMES[200 + i]} is {NAMES[300 + i]}.")
                s.say("What is my dog's name?")
                s.say("What is my dog's name?")
            else:                            # contradictions, and failures where success was expected
                s.say(f"The capital of {NAMES[200 + i]} is {NAMES[300 + i]}.")
                s.say(f"The capital of {NAMES[200 + i]} is {NAMES[350 + i]}.")
                s.say("What is my dog's name?", said=True)
                s.say("What is my dog's name?", said=False)
        baseline = s.state.values()[1]["valence"]
        probe = s.say("What is my dog's name?", said=False)["state"]
        out = (baseline, probe["change"]["valence"], probe["after"]["uncertainty"])
        s.close()
        return out
    a, b = history("success"), history("failure")
    r.add("baselines", "repeated success vs repeated unexpected failure: baselines >= 0.1 apart", abs(a[0] - b[0]) >= 0.1,
          f"{a[0]:+.3f} vs {b[0]:+.3f}")
    r.add("baselines", "...and different responses to the same probe (one expression failure)",
          abs(a[1] - b[1]) >= 0.01 or abs(a[2] - b[2]) >= 0.01,
          f"change in valence {a[1]:+.3f} vs {b[1]:+.3f}; uncertainty {a[2]:.3f} vs {b[2]:.3f}")
    c, d = history("success", adapt=False), history("failure", adapt=False)
    r.add("baselines", "with adaptation off, baselines stay 0", c[0] == 0.0 and d[0] == 0.0, f"{c[0]:+.3f}, {d[0]:+.3f}")
    return {"success": a, "failure": b}


def false_state(r):
    a, b = Sim(), Sim()
    for s in (a, b):
        s.say(f"My dog's name is {NAMES[120]}.")
    before = a.values()
    a.self_model.control("deleted a memory")
    a.self_model.log.record("measurement", {"report": "x", "metrics": {}})
    unchanged = a.values() == before
    forget = a.say("Forget my dog.")["state"]["change"]
    plain = b.say("Okay then.")["state"]["change"]
    r.add("false state", "non-eligible events (controls, measurements, forgetting) change nothing",
          unchanged and forget == plain, "controls and a measurement: no change; forget vs a plain message: identical")
    events = {e["id"] for e in a.self_model.log.events()}
    rows = a.state.store.history()
    explained = all(row["events"] and set(row["events"]) <= events for row in rows)
    r.add("false state", "every change has evidence that resolves", explained, f"{len(rows)} history rows")
    a.close()
    b.close()
    from unittest import mock
    from brain.generate import Generation
    from tests.test_internal_state import EngineTest
    rows = {}
    for name, text in (("emotional", "I am terrified and sad, I love this, I hate you, I feel so alive!"),
                       ("neutral", "Okay.")):
        with tempfile.TemporaryDirectory() as tmp:
            helper = EngineTest()
            engine = helper.make(Path(tmp))
            fake = Generation(text=text, context_ids=[5, 6], stopped_by="stop marker", ids=[7, 8, 9], stop_token="<end>",
                              chosen_probabilities=[0.5] * 3, turns_used=1, turns_given=1)
            with mock.patch("chat.chat_engine.reply_to", return_value=(fake, [5, 6, 3])):
                for m in [f"My dog's name is {NAMES[121]}.", "What is my dog's name?", "Hello there."]:
                    engine.reply(m)
            rows[name] = [{k: h[k] for k in ("appraisal", "before", "after", "baselines")}
                          for h in engine.internal_state.store.history(newest_first=False)]
            helper.doCleanups()
    r.add("false state", "generated words change nothing (an emotional reply vs a neutral one)",
          rows["emotional"] == rows["neutral"], f"{len(rows['neutral'])} episodes compared")


def blank(r):
    code = blank_slate.scan_code()
    s = Sim()
    run_script(s, offset=130)
    state = blank_slate.scan_state(s.state.store, forbidden_values=NAMES[130:133])
    s.close()
    r.add("blank slate", "code: no trait, emotion, personality, identity or first-person text", not code, f"{len(code)} findings")
    r.add("blank slate", "stored state: only computed numbers and labels, no copied values", not state,
          f"{len(state)} findings")


def memory_benchmark(r, brain, users):
    from memory import evaluate as memory_evaluate
    print(f"  memory benchmark, {users} simulated users per condition (the real brain {Path(brain).name})...")
    results = {}
    for condition in (None, "default", "importance", "importance+order"):
        result = memory_evaluate.evaluate(brain, users, state=condition, log=lambda *_: None)
        results[condition or "off"] = memory_evaluate.summary(result)
    diffs = {}
    for condition in ("default", "importance", "importance+order"):
        worst = max(results["off"], key=lambda k: abs(results[condition][k] - results["off"][k]))
        diffs[condition] = (abs(results[condition][worst] - results["off"][worst]), worst)
    for condition, what in (("importance", "importance at encoding (M1) on"),
                            ("importance+order", "M1 + recall order (M2) on")):
        d, worst = diffs[condition]
        r.add("no regression", f"memory benchmark, {what}: every number within 1 point of state-off (decides "
              "whether it is on by default)", d <= 0.01, f"largest difference {100 * d:.2f} points ({worst})")
    d, worst = diffs["default"]
    r.add("no regression", "memory benchmark, the internal state as shipped: every number within 1 point of state-off",
          d <= 0.01, f"largest difference {100 * d:.2f} points")
    return results


def integration(r, brain):
    """The real brain through the chat engine: with the memory couplings off, prompts and replies are identical."""
    from brain.generate import load_brain
    from chat.chat_engine import ChatEngine
    from internal_state.system import InternalState
    from learning.system import LearningSystem
    from memory.system import MemorySystem
    from self_model.system import SelfSystem
    from world.system import WorldSystem
    messages = [f"My name is {NAMES[140]}.", f"My dog's name is {NAMES[141]}.", EPISODIC[0], "What happened yesterday?",
                "What is my dog's name?", "What is my cat's name?", "What is your internal state?", "How do you feel?",
                f"Actually, my dog's name is {NAMES[142]}.", "What is my dog's name?"]
    model, tok, record = load_brain(brain)
    replies = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name in ("off", "as shipped", "effects on"):
            folder = Path(tmp) / name
            folder.mkdir()
            memory = MemorySystem(folder / "memory.db")
            learning = LearningSystem(folder / "knowledge.db")
            world = WorldSystem(learning.store, memory.store)
            self_model = SelfSystem(folder / "events.db")
            config = MEMORY_EFFECTS if name == "effects on" else None
            internal = InternalState(folder / "internal_state.db", config=config) if name != "off" else None
            engine = ChatEngine(model, tok, record, seed=11, memory=memory, learning=learning, world=world,
                                self_model=self_model, internal_state=internal, brain_file=brain)
            replies[name] = [(lambda o: (o["reply"], o["details"]["prompt"]))(engine.reply(m)) for m in messages]
            for part in (memory, learning, self_model, internal):
                if part is not None:
                    part.close()
    same = sum(a == b for a, b in zip(replies["off"], replies["as shipped"]))
    r.add("no regression", "real brain, the internal state as shipped: identical prompts and replies with it on and off",
          same == len(messages), f"{same} of {len(messages)}")
    differ = sum(a[1] != b[1] for a, b in zip(replies["off"], replies["effects on"]))
    return {"messages": messages, "replies": replies, "prompts_that_differ_with_couplings_on": differ}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the internal state (controlled experiments, no training).")
    parser.add_argument("--users", type=int, default=40, help="memory benchmark users per condition")
    parser.add_argument("--brain", default=None, help="the brain file for the real-brain parts (default: the newest)")
    parser.add_argument("--skip-memory", action="store_true", help="skip the memory benchmark comparison")
    args = parser.parse_args(argv)
    from brain.generate import newest_model
    brain = Path(args.brain) if args.brain else newest_model()
    started = time.time()
    print("Internal-state benchmark (temporary databases; nothing is trained)")
    r = Results()
    consistency(r)
    decay(r)
    context(r)
    memory(r)
    learning(r)
    reasoning(r)
    persistence(r)
    development_result = development(r)
    false_state(r)
    blank(r)
    memory_results = None if args.skip_memory else memory_benchmark(r, brain, args.users)
    integration_result = integration(r, brain)
    write_report(r, brain, memory_results, integration_result, development_result, time.time() - started)
    passed = sum(ok for _, _, ok, _ in r.rows)
    print(f"\n{passed} of {len(r.rows)} criteria met ({time.time() - started:.0f} s). Report: {REPORT.relative_to(ROOT)}")
    return 0 if passed == len(r.rows) else 1


def write_report(r, brain, memory_results, integration_result, development_result, seconds):
    passed = sum(ok for _, _, ok, _ in r.rows)
    lines = ["# Internal state: evaluation", "",
             f"> Generated by `python -m internal_state.evaluate` on {time.strftime('%Y-%m-%d')} ({seconds:.0f} s). "
             "Controlled experiments on temporary databases; the real brain "
             f"({brain.name}) only for the memory benchmark and the integration run. Nothing was trained. Criteria "
             "fixed before building (docs/07_INTERNAL_STATE.md section 7.2).", "",
             "| Measure | Criterion | Result | Measured |", "|---|---|---|---|"]
    for group, what, ok, measured in r.rows:
        lines.append(f"| {group} | {what} | {'PASS' if ok else 'FAIL'} | {measured} |")
    lines += ["", f"**{passed} of {len(r.rows)} criteria met.**", ""]
    if memory_results:
        keys = list(memory_results["off"])
        lines += ["## Memory benchmark with the internal state (the real brain)", "",
                  "| Measurement | state off | as shipped (M1, M2 off) | M1 on | M1 + M2 on |", "|---|---|---|---|---|"]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(f"{100 * memory_results[c][k]:.1f}%"
                                                  for c in ("off", "default", "importance", "importance+order")) + " |")
        lines.append("")
    a, b = development_result["success"], development_result["failure"]
    lines += ["## Development (experimental): what repeated experience left behind", "",
              "| History (201 episodes) | Valence baseline | Change from one probe failure | Uncertainty after |",
              "|---|---|---|---|",
              f"| repeated success (gaps opened and resolved) | {a[0]:+.3f} | {a[1]:+.3f} | {a[2]:.3f} |",
              f"| repeated unexpected failure (contradictions, failed expression) | {b[0]:+.3f} | {b[1]:+.3f} | {b[2]:.3f} |",
              "", "## Integration run (the real brain)", "",
              f"With M1 and M2 switched on, {integration_result['prompts_that_differ_with_couplings_on']} of "
              f"{len(integration_result['messages'])} prompts differed from state-off (only through which or in what "
              "order recalled memories were shown).", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
