# The experience loop (integration scaffold)

> Written 2026-10-06. An orchestration layer that connects parts QI already has. It trains nothing, adds no goal,
> identity, preference or emotion, and makes no claim about consciousness or wanting. With no target, the action is
> whatever the existing planner and curiosity mechanisms return.

## The loop

```
ENVIRONMENT (sandbox simulator)
   │  observe the current state (grid, cell, held key, experience memory)
   ▼
WORLD MODEL ─ predicts each action's outcome (official 3 networks, frozen; input = state + action only)
   ▼
CURIOSITY ─── uncertainty U and expected information gain per action (sandbox/curiosity.py, unchanged)
   ▼
PLANNER ───── chooses the action (sandbox/planner.py, unchanged; no target → its own curiosity fallback)
   ▼
SIMULATOR ─── executes it: the only place the true outcome exists, consulted after the prediction and the choice
   ▼
COMPARE ───── prediction vs actual: right/wrong, 1 − P(actual), information actually gained
   ▼
RECORD ────── one JSON line per step, written at once (state/experience_loop/<episode>.jsonl; never overwritten)
   ▼
UPDATE ────── the agent's experience memory (sandbox/experience.py's rule: what a switch was seen to do)
   └──────── loop
```

## Boundaries (kept separate on purpose)

| Part | What it is | Where | Changed here? |
|---|---|---|---|
| Brain | neural networks (the world model's 3 networks) | `models/world/world_model_v3_mix25_seed*.npz` | no: loaded, run forward only |
| World model | learned predictions of what an action does | `sandbox/model.py`, `sandbox/curiosity.py` (Ensemble) | no |
| Curiosity | information-gain measure | `sandbox/curiosity.py` | no |
| Planner | action-sequence selection | `sandbox/planner.py` | no |
| Simulator | the world's true rules | `sandbox/env.py` | no |
| Memory (agent) | the experience memory carried in the state | `sandbox/experience.py` | no |
| Memory (persistent) | the experience log | `state/experience_loop/*.jsonl` | new, append-only |
| Experience loop | orchestration only | `sandbox/experience_loop.py` | new |
| Inspector | a local page to watch it | `sandbox/experience_loop_inspector.py` | new |

The loop object only calls these parts in order; it holds no learned weights and makes no decision of its own.

## Integrity rules it keeps

- Prediction, scoring and choice use only the state and the action (the world model's 77 input tokens); the
  simulator is called once per step, after the planner has chosen (tested: `tests/test_experience_loop.py`).
- The recorded experience holds what the agent observed (state, action, changes, events); no simulator labels.
- Episodes are deterministic: the same world gives the same steps; a reset starts the same episode again.
- No training, no checkpoint is written, no external AI, API or internet; the official checksums are tested.

## Not connected yet (and why)

The conversational brain (`brain/`, `chat/`), its persistent conversational memory (`state/memory.db`), learned
knowledge (`state/knowledge.db`), the internal state (`internal_state/`, `state/internal_state.db`) and the self-model
(`self_model/`, `state/events.db`) each read and write their own databases. Connecting them would write into those
stores, so this scaffold leaves them untouched. The clean next step is an interface, not an implementation:

- **Experience → self-model/event log:** the loop's step records already have the shape of events (what was
  observed, done, predicted, what happened, the prediction error); a reader could turn a finished episode into event
  log entries, so the self-model could answer "what happened" from recorded facts, as it does for chat.
- **Conversational brain → loop:** the brain cannot read grids and the world model cannot read words, so any link
  needs an explicit, inspectable translation layer (for example: a question about an episode answered by the
  self-model from the log), never a hidden mapping. Deciding that layer is the next research task.

## What one episode shows

In official test world 200000 the first 30 steps are moves and interactions whose outcomes the world model already
predicts correctly (30 of 30); every expected gain is 0 because no unseen switch is under the agent, so the
planner's curiosity fallback takes its seeded tie order. Curiosity, as it exists (D-083), scores one step ahead: it
values pressing an unseen switch when standing on one, not walking toward one.

## Running it

```
python -m sandbox.experience_loop_inspector        # http://localhost:8770
python -m unittest tests.test_experience_loop      # 8 integration checks, < 1 s
```

## The translation layer to the conversational brain (D-087)

`sandbox/experience_translation.py` is ordinary code with fixed rules, not a network:

```
finished loop records ──copy──▶ structured facts ──fixed templates──▶ "recorded: …" (≤ 3 sentences)
      ──▶ conversational brain (its own self-line layout + the person's message; no memory, learning, reasoning,
          self-model or internal state attached; fixed seed) ──▶ its output, exactly
      ──fixed grammar (only an output that BEGINS with observe / what is happening / predict ‹action› /
          act ‹action›)──▶ a request, or none ──(only when the person sends it)──▶ the loop
```

It reads only finished records, so the simulator's truth reaches the brain only after the loop has predicted,
chosen and acted. It adds no fact and no interpretation, and the brain is not told what it is or what to want.
`http://localhost:8770/bridge` shows the exact data at every boundary.
