# 06 — The QI architecture

**QI** is this project's name for what we are building: an original, locally trained system that is
meant to become **more capable through learning and experience**. Over time it should become more
autonomous and model itself more. The chat window is only how you talk to it.

This document answers the question you asked before any implementation:

> *"What architecture would we need to evolve the current tiny neural network into a much broader
> artificial intelligence system?"*

It builds on the research in [05_QI_RESEARCH.md](05_QI_RESEARCH.md). The build order, with a
measurable test for every stage, is in [04_ROADMAP.md](04_ROADMAP.md). It replaces the parts of
[02_ARCHITECTURE.md](02_ARCHITECTURE.md) that conflict with the QI blank-slate brief (§8 lists
exactly which).

Nothing here claims the system is or will be conscious. That remains an open scientific question
(§11, [03_CONSCIOUSNESS.md](03_CONSCIOUSNESS.md)).

---

## 0. The short answer

Today's brain is a **function**: text goes in, text comes out. It learned once, offline, from books,
and between messages it does not exist. (Since Stage 2, what you tell it is kept in a long-term memory
outside the network and shown to it again when relevant. The brain itself is still a function.)

A broader intelligence needs a different **center**. Instead of a model that maps text to text, it
needs a **system that lives in a loop**:

1. it **perceives** a world;
2. it **predicts** what will happen;
3. it **acts**;
4. it **compares** what happened with what it predicted;
5. it **learns** from the difference, keeping what it experienced.

In that system:

- a learned **world model** is the core;
- **memory** accumulates experience;
- **continual learning** turns experience into lasting skill without erasing old skill;
- **curiosity** chooses what to learn next;
- **goals and planning** use the world model to reach outcomes;
- a **self-model** is learned from the same data as everything else: the system's own behavior.

Language becomes **one module**: an interface to all of this, and eventually grounded in it.

Three foundations have to exist before the rest can be built and measured:

| Foundation | Why everything else needs it |
|---|---|
| **1. A persistent loop with an event log** (Stage 3) | Memory, learning from experience, goals and a self-model need a continuous existence whose experience is recorded. Today only conversations and what you tell it persist (Stage 2) |
| **2. A world where actions have consequences** (Stage 4) | A world model needs something to model; curiosity needs something to explore; goals and planning need outcomes. Chat alone is too thin: the system cannot act on anything, and there is no ground truth to measure against |
| **3. A predictive learner for that world** (Stage 5) | Prediction is the learning signal that needs no teacher. It is the base for curiosity, planning and self-modeling. It reuses our own transformer code |

You chose to build **long-term memory for conversations first** (Stage 2, D-055). It needs no world, it
is software around the existing brain, and it was measured on its own. The three foundations above follow as
Stages 3-5, and Stage 7 extends the same memory to experience in the world.

---

## 1. From a chatbot to QI

| | Today (Stage 1.5b round 2 + Stage 2 memory) | QI |
|---|---|---|
| Existence | Called once per message | A process that runs continuously, with a clock |
| Learning | Once, offline, from a fixed dataset | Continually, from its own experience, with safeguards |
| Learning signal | Next token of human text | Prediction of **everything it experiences** (world, language, itself) + learning progress |
| Memory | What you tell it, kept in SQLite and recalled into the prompt (Stage 2), plus the last few turns | An event log of its whole life + memory of its experience + consolidation into weights |
| Actions | Writes text | Acts in a world (first a sandbox), observes the consequences |
| Motivation | None | Content-free curiosity (learning progress); goals you give it; goals it learns |
| Self-knowledge | None | Learned predictors of what it controls, can do, knows, and is doing, checked against its logs |
| Language | The whole system | One module; later grounded in the world model |

---

## 2. The architecture

```
 ┌──────────────────────────────── ENVIRONMENT ────────────────────────────────┐
 │  sandbox world (Stage 4) · you, through the chat · later: camera, microphone, │
 │  clock, files (only through the safety gateway)                               │
 └────────────┬─────────────────────────────────────────────────▲─────────────┘
  observations│                                                   │actions
 ┌────────────▼────────────┐                          ┌───────────┴────────────┐
 │ PERCEPTION              │                          │ ACTION + SAFETY GATEWAY │
 │ encoders: world → tokens│                          │ permission levels, STOP,│
 │ (later: camera, audio)  │                          │ audit log               │
 └────────────┬────────────┘                          └───────────▲────────────┘
              │                                                   │
 ┌────────────▼───────────── WORKSPACE (the current moment) ──────┴────────────┐
 │  what is happening now · retrieved memories · active goal · current plan    │
 └───▲───────────▲────────────▲─────────────▲──────────────▲─────────────▲─────┘
     │           │            │             │              │             │
 ┌───┴────┐ ┌────┴─────┐ ┌────┴─────┐ ┌─────┴──────┐ ┌─────┴─────┐ ┌─────┴─────┐
 │ WORLD  │ │ LANGUAGE │ │ EPISODIC │ │ SELF-MODEL │ │ GOALS +   │ │ PLANNER   │
 │ MODEL  │ │ MODULE   │ │ MEMORY   │ │ learned    │ │ CURIOSITY │ │ search in │
 │predicts│ │ today's  │ │ stored   │ │ predictors │ │ told /    │ │ imagined  │
 │what    │ │ brain;   │ │ experi-  │ │ of its own │ │ intrinsic │ │ futures   │
 │happens │ │ grounded │ │ ences,   │ │ control,   │ │ / learned │ │ of the    │
 │next    │ │ later    │ │ retrieval│ │ skill,state│ │           │ │ world     │
 └───┬────┘ └──────────┘ └────┬─────┘ └────────────┘ └───────────┘ │ model     │
     │                        │                                     └───────────┘
     │  learning signals: prediction error, learning progress, uncertainty, goal progress
     ▼
 ┌──────────────────────────────────────────────────────────────────────────────┐
 │ REGULATORY STATE: a few unnamed variables computed from those signals.        │
 │ They tune learning rate, exploration, what gets stored, how far to plan.      │
 └──────────────────────────────────────────────────────────────────────────────┘
 ══════════════════════════════════════════════════════════════════════════════
  EVENT LOG: every observation, action, prediction, memory write and weight change,
  with its source and cause  ──►  CONSOLIDATION ("sleep"): replay from memory,
  continual learning of every network, evaluation gates, automatic rollback
```

**How your diagram maps onto it:**

| Your box | Where it lives |
|---|---|
| BRAIN | The learned networks: world model, language module, self-model predictors (one shared transformer implementation) |
| MEMORY | Event log (everything) + episodic memory (selected, indexed) + semantic knowledge (consolidated into weights and a fact store) |
| WORLD MODEL | Its own module, at the center of learning |
| SELF MODEL | Learned predictors about the system itself (§6) |
| EMOTION | The regulatory state (§5, §6): unnamed, computed from real signals, measured by what it changes |
| GOALS, CURIOSITY | Goal store + learning-progress curiosity |
| PLANNING | Search in the world model |
| ACTION, ENVIRONMENT, PERCEPTION | Action gateway, sandbox and chat (later real sensors), perception encoders |
| LEARNING → MEMORY | Online updates + consolidation, fed by the event log and memory |

---

## 3. The loop

The QI system runs a **cognitive cycle**, as in the Common Model of Cognition (05 §1). In the sandbox
it is one cycle per world step; in chat it is one cycle per message; when nothing happens it idles.

```python
while alive:
    obs = environment.observe()                        # PERCEIVE
    log.record(obs, source="OBSERVED")
    x = perception.encode(obs)
    surprise = world_model.error(last_prediction, x)   # how wrong was I?
    workspace.update(x, memory.retrieve(x), goals.active())
    regulation.update(surprise, world_model.uncertainty(x), learning_progress(), goal_progress())
    goal = goals.select(workspace, curiosity.scores(), regulation)
    plan = planner.search(world_model, workspace, goal, depth=regulation.planning_depth)
    action = gateway.check(plan.first_action)          # safety rules apply to every action
    last_prediction = world_model.predict(x, action)   # predict BEFORE acting
    environment.act(action)
    log.record(action, last_prediction, cause=plan)
    self_model.observe(action, last_prediction, regulation.state)
    memory.maybe_store(event, importance=from(surprise, later_usefulness))
    learners.small_online_update()                     # bounded; can be switched off
    if consolidation.due(): consolidation.run()        # replay, continual learning, gates, rollback
```

Every piece of this is added in a separate stage, and each stage is tested before the next one begins.

---

## 4. The common currency: events with provenance

Every module reads and writes **events**, stored in one SQLite log:

```python
Event(id="ev_000412", t=..., kind="observation" | "action" | "prediction" | "utterance" |
      "memory" | "goal" | "weight_update" | "self_report",
      channel="sandbox" | "chat" | "internal",
      content={...},
      source="OBSERVED" | "TOLD" | "REMEMBERED" | "INFERRED" | "IMAGINED" | "GENERATED",
      caused_by=["ev_000409"], module="planner", version="world_model@v3")
```

- **Provenance** (kept from 02_ARCHITECTURE §C1): a guess never silently becomes a fact, and anything
  the language module writes is `GENERATED`, not knowledge.
- **Causes:** every action and every self-report links to the events that produced it, so `/why`
  can show the real chain of causes.
- **Replay:** the log is enough to rebuild the system's state, which makes experiments repeatable.

---

## 5. The blank slate, made precise

An agent with **no** built-in motivation at all does nothing, and learns nothing it doesn't happen to
be fed. So "blank slate" cannot mean "nothing built in". It means: **build in how to learn, never
what to think, want or feel.**

| Built in (architecture: the "how") | Learned from experience (the "what") | Never written in |
|---|---|---|
| Learning algorithms (backprop, AdamW, replay) | Knowledge of the world (world model) | Personality or traits |
| Prediction as the learning objective | Language (from human text, later grounded) | A name |
| **Content-free** intrinsic signals: learning progress, uncertainty | Skills, habits (through planning and practice) | Beliefs, opinions, worldview |
| Default couplings of the regulatory state (measured by ablation, later learned: Stage 14) | Which goals to pursue, beyond the ones you give | Values |
| The safety gateway (limits on **actions**, not on what it may want) | Preferences, as measured tendencies with evidence | Emotions or emotional reactions |
| Provenance and honesty checks on self-reports | "Character": stable dispositions that emerge from its history | Goal content, preferences |
| The measurement harness | Self-knowledge (self-model) | A backstory |

**Edge cases, stated honestly:**

1. **Curiosity is a built-in drive.** It is content-free: it rewards "my predictions got better",
   never a particular thing. It is also the only built-in drive in the plan.
2. **The world we design shapes what it can learn.** It is documented like a dataset (a "world card"),
   so every influence is visible.
3. **Its language comes from human books**, so it will echo human phrases, including "I feel...".
   Those words are `GENERATED` and are not evidence of anything inside (03 §4).
4. **"Persistent characteristics" become measurable:** the self-model tracks stable behavioral
   statistics, such as how much it explores and what it returns to. A clean experiment: two copies
   with different histories should develop measurably different dispositions, which would show they
   come from experience, not from us.
5. **Open decision for you (O-7):** should the sandbox body have a *homeostatic* variable (e.g.
   energy that must be replenished)? It would create survival-like pressure and richer regulation, but
   it is a second built-in drive. Recommendation: first curiosity only; later a homeostatic body as a
   controlled experiment, compared by ablation.

---

## 6. The modules

| Module | What it does | Learns from | Stage | Category |
|---|---|---|---|---|
| **Runtime + event log** | The loop, the clock, the log, module plug-ins, replay. **The event log is built (D-073):** every message and what it caused, with cause and references (`self_model/events.py`); the loop, clock and replay are not | — (infrastructure of the mind) | 3 | Ours |
| **Sandbox world** | A small world with a body, objects, simple rules that differ between worlds, noise, a decoy agent, a language channel | — (it *is* the environment) | 4 | Ours |
| **Perception** | Turns observations into tokens / vectors | Prediction (self-supervised) | 4-5, 15 | Ours / Technique |
| **World model** | Predicts the next observation given an action; an ensemble gives uncertainty | Its own experience | 5 | Technique + Ours (our transformer) |
| **Curiosity** | Scores where learning progress is expected | World-model error history + ensemble disagreement | 6 | Technique + Ours |
| **Long-term memory** | **Built (Stage 2)** for conversations: facts, relationships, preferences and events you tell it, in SQLite; retrieval by words, names, kind, importance, recency; recalled into the prompt as earlier turns. Stage 7 extends it to experience in the world | Now: what you say (readable rules), "remember that", your "important" mark. Later: surprise and later usefulness | 2, 7 | Technique + Ours |
| **Learned knowledge** | **Built (Stage 2b):** a small knowledge graph of what you state or teach: entities, facts with source, confidence, confirmations, corrections (history kept), flagged conflicts; current facts recalled into the prompt | Your messages only (readable rules) | 2b | Ours / Technique (knowledge graphs) |
| **World model (structured) + reasoning** | **Built (Stage 2c):** the learned knowledge as a navigable graph (entities, properties, relationships, history, typed values), question interpretation, retrieval, reasoning operations (lookup, chains, corrections, comparisons, known / unknown / conflicting / uncertain), an answer representation the brain turns into words. Not the learned predictive world model of Stage 5 | What you said (knowledge store + memory) | 2c | Ours / Technique (knowledge graphs, graph search) |
| **Consolidation / continual learning** | Replays memory into the networks without forgetting; gates, rollback, plasticity | All experience | 8 | Technique + Ours |
| **Language module** | Reads and writes language; later describes the world and memories | Books, dialogue, your conversations; later the sandbox's language channel | 1–1.5b, 9 | Ours |
| **Goals** | Goal objects (desired world conditions) with source, priority, progress | Told by you, intrinsic, learned from experience | 10 | Technique + Ours |
| **Planner** | Searches imagined futures in the world model; re-plans on surprise | Improves as the world model and value estimates improve | 10 | Technique + Ours |
| **Regulatory state** | A few variables summarizing error, uncertainty, progress and effort; modulates learning and acting. **Its foundation is built (Stage 2e, D-074):** valence, activation, prediction error, novelty, uncertainty, information need and a learning gain from a content-free appraisal of verified events, decaying to adaptive baselines; it sets the expectations' learning speed and tags significant memories | Computed from real signals; couplings later learned | 2e, 11 | Technique (functional view) + Ours |
| **Self-model** | Predicts its own control (agency), success (competence), knowledge and state; produces checked self-reports. **The foundation is built (D-073):** a record of its own state, actions, capabilities, results, uncertainty and history, computed from the event log with evidence, never shown to the brain; no learned predictors yet | Its own logged behavior | 2d, 12 | Technique + Ours |
| **Workspace / executive** | Limited-capacity "current moment"; learns which module to consult | Usefulness of past consultations | 13 | Technique + Ours |
| **Reflection** | Offline: finds its failure patterns, generates practice, tunes its own learning within bounds | Logs + self-model | 14 | Technique + Ours |
| **Action gateway** | Every action is permission-checked, logged, stoppable | — (fixed safety rules) | 3, 16 | Ours |

**One shared learner.** The world model, language module and self-model predictors all use the same
hand-written transformer (`brain/layers.py`, `brain/transformer.py`), including rotary positions from
round 2. One well-tested implementation is easier to trust than many.

---

## 7. How today's Stage 1.5b becomes part of QI

- **The transformer, tokenizer, training and gradient checks** become the shared learner (§6).
- **The language module is today's brain.** Round 2 trained it to **use information given earlier in
  its context**: it now answers from an earlier turn 79% of the time on names and words it never saw
  (Stage 1.5b: 1%). That is the exact skill memory retrieval needs later: retrieved memories are placed
  in its context, and it has to use them. Its limit matters too: it recognizes only the trained ways
  of *giving* information (12-16% on new wordings). Grounding (Stage 9) is meant to address that.
- **The measurement discipline carries over:** held-out tests, criteria fixed before training, model
  quality kept apart from decoding quality, baselines. It grows into a **QI benchmark** that runs on
  every version, so the capability ladder is measured, not asserted.
- **The chat page becomes the QI console:** the chat, a view of the sandbox, and an inspector for the
  event log, memories, goals, regulatory state and self-model.

---

## 8. What changes from the earlier architecture (02_ARCHITECTURE.md)

02_ARCHITECTURE was written for the first brief, which included a personality, a fictional life and
named emotions. The QI brief rules those out. Parts of 02 stay; this table is the authority where
they differ.

| Keep | Change | Drop |
|---|---|---|
| Provenance tags (OBSERVED, TOLD, REMEMBERED, INFERRED, IMAGINED, GENERATED) | **Emotion:** 10 named emotions with hand-written equations → an unnamed regulatory state computed from real learning signals (§5) | Identity with a name, Big Five traits and values (§C6) |
| Glass box: `/why`, inspectable everything | **Appraisal:** word lexicons and rules → learned signals (prediction error, progress) | The fictional life / backstory (§C7) and the `AUTHORED` tag |
| Bounded dynamics, rate limits, depth limits | **Memory importance:** a hand-weighted emotion formula → surprise and later usefulness | Protected `DEVELOPER` "facts" injected as beliefs |
| Controlled learning: gates, replay, rollback | **Goals:** a list of innate goals (social contact, keep values...) → only content-free curiosity, plus told and learned goals | Loneliness and other built-in social drives |
| The action gateway (levels 0-3, STOP, audit log) | **Workspace salience:** hand-set weights → learned from usefulness | `<mood:...>` control tokens and emotion-styled language |
| Episodic/semantic stores, sleep consolidation | **Self-model:** a hand-filled JSON description → learned predictors checked against logs | Mood-congruent recall as a designed formula (it may still emerge) |
| Own perception (MFCC, small CNN), later | **Reasoning:** hand-written inference rules → planning with a learned world model | |

Open decisions O-1 (the agent's name) and O-2 (knowing its backstory is authored) are closed by the
QI brief: there is no given name and no backstory. If it ever gets a name, it would come from you in
conversation, tagged `TOLD`, like any other fact.

---

## 9. What runs on this Mac

Measured: Apple M4, 10 CPU cores, 16 GB, NumPy with Accelerate (~1-2 TFLOP/s). Estimates are marked ≈.

| Module | Size | ≈ Cost | Verdict |
|---|---|---|---|
| Runtime, event log, gateway | — | negligible | Easy |
| Sandbox world | 8×8 to 12×12 grids | thousands of steps per second | Easy |
| World model (one) | 1-3M parameters | ≈ 30-60 min to train on ~10⁵ steps of experience | Yes |
| Ensemble of 3-5 world models | 3-5× that | ≈ 2-4 h | Yes; smaller models if needed |
| Episodic memory | 10⁵-10⁶ records, 128-number vectors | milliseconds per search | Easy |
| Continual-learning experiments | several worlds in sequence | hours per experiment | Yes |
| Planner | tree search, a few hundred imagined steps per decision | ≈ 0.1-1 s per decision | Yes in small worlds |
| Regulatory state, goals, curiosity | tiny | negligible | Easy |
| Self-model predictors | small | minutes | Yes |
| Grounded language | the language module + the sandbox's language channel | ≈ hours | Yes |
| Camera perception (our CNN) | 32×32-64×64 images | ≈ hours in NumPy | Slow; the MLX math backend (O-3) would help |
| Quantum simulation | ≤ ~20 qubits in training loops | seconds per circuit batch | Yes, as research |

**Not realistic here:** broad, fluent conversation (that needs thousands of times more data and
compute than a laptop has), open-vocabulary speech recognition, real quantum hardware, or large
real-world vision. The QI project's value is the **architecture and its measured capabilities in its
own world**, not competing with large chatbots.

---

## 10. Quantum and QI

The honest state of the field (05 §12):

- There is no demonstrated quantum advantage for machine learning on classical data.
- Small quantum circuits simulated on a Mac are classical math, so they cannot be faster.
- The one clearly practical quantum-*inspired* tool for us is **tensor networks** (compact weight
  matrices).

**Design rule:** quantum and quantum-inspired methods live in a separate **Q research track**
(04_ROADMAP). Each is compared head-to-head with a classical baseline under criteria written in
advance. **A quantum component enters the core system only if it wins.** Until then, the "Q" in QI
names a research question the project investigates, not a property the system has.

---

## 11. Consciousness: the stance and the audit

- It is an **open scientific question**; there is no accepted test (03_CONSCIOUSNESS).
- The QI architecture will acquire several **indicator properties** from Butlin, Long et al. as
  architectural facts:
  - a predictive world model (PP-1);
  - learning how its outputs change its inputs (AE-2, in the sandbox);
  - flexible goal pursuit from feedback (AE-1);
  - a limited-capacity workspace with broadcast (GWT-2, GWT-3);
  - metacognitive monitoring (HOT-2, via source tags and calibration).
- At each stage we will record which indicators the system has, **descriptively**. Because we build
  them on purpose (the "gaming problem"), they are **not** evidence of experience.
- **Claims we can make** once measured: "its self-reports are caused by its internal variables", "its
  regulatory state changes its learning", "its self-model predicts its own success with calibration
  error X". **Claims we never make:** that it is conscious, feels, suffers or is self-aware in the
  human sense.

---

## 12. What is ours, what is borrowed, what is speculative

| Component | Our implementation | Established technique | Infrastructure (library) | Speculative research |
|---|---|---|---|---|
| Tokenizer, transformer, backprop, AdamW, rotary positions | ✓ all code | BPE, transformers, backprop, RoPE | NumPy (math) | — |
| Runtime, event log, provenance | ✓ | Cognitive cycle (Common Model) | SQLite (storage) | — |
| Sandbox world | ✓ | Grid worlds (MiniGrid/BabyAI style) | — | — |
| World model, ensemble uncertainty | ✓ | IRIS/Dreamer-style world models, ensembles | NumPy | — |
| Curiosity | ✓ | Learning progress, disagreement | — | — |
| Episodic memory, consolidation | ✓ | CLS, replay, EWC, continual backprop | SQLite, NumPy | — |
| Planner | ✓ | MCTS, model-predictive control | — | — |
| Regulatory state | ✓ | Functional view of emotion; emotion-in-RL | — | Whether emotion-like patterns emerge |
| Self-model | ✓ | Robot self-modeling, calibration, contingency detection | — | — |
| Workspace | ✓ | Global Workspace Theory (as an engineering pattern) | — | Its relation to consciousness |
| Perception (later) | ✓ | MFCC, CNNs | sounddevice, OpenCV capture only | — |
| Q track | ✓ our own simulator and tensor code | Tensor networks | NumPy | Quantum advantage for QI; quantum cognition; quantum theories of mind |
| Consciousness audit | ✓ | Indicator method (descriptive) | — | The question itself |

**Never used:** pretrained models or weights, AI APIs, cloud inference, Ollama, neural-network
frameworks' layers or autograd, AI-written training text.

---

## 13. Risks and how the design handles them

| Risk | Guard |
|---|---|
| **Scripted intelligence creeping in** (a hard-coded behavior that looks like learning) | Every capability is tested on **held-out** worlds, names, wordings and goals. The Originality Guard already blocks computed answers in the chat path, and is extended to block scripted action sequences in the agent path |
| **Moving goalposts** | Criteria written before each training run; thresholds only change with a new decision entry |
| **Gaming the indicators** | Architectural facts are reported as such, never as evidence of experience |
| **Forgetting and loss of plasticity** | Replay, evaluation gates, automatic rollback, plasticity tests (Stage 8) |
| **Runaway loops** (reflection, regulation) | Hard bounds, depth limits, budgets (kept from 02) |
| **Unsafe actions as autonomy grows** | The sandbox first; real-world actions only through the gateway, with levels, confirmation, STOP and an audit log; no shell, no network actions without your approval |
| **Compute limits** | Small worlds, small models, one heavy job at a time; the MLX math backend as an option |
