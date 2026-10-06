# 04 — The QI roadmap

The build plan for QI (the architecture is in [06_QI_ARCHITECTURE.md](06_QI_ARCHITECTURE.md), the
research behind it in [05_QI_RESEARCH.md](05_QI_RESEARCH.md)). It replaces the earlier 20-stage plan,
which was built around a scripted personality and a fictional life. That plan is kept in git history
(commit `d7af8c8`).

**Every stage is described the same way:**

- **Capability:** the one new ability it adds
- **Why:** why the system needs it, and why at this point
- **Depends on:** earlier stages it needs
- **Measured by:** the pass test (its exact thresholds are written down *before* the stage is built
  and never adjusted afterwards)
- **Still missing:** what it does not do yet

Every stage is delivered in the beginner format:

1. what we are building;
2. why;
3. folders;
4. complete files;
5. exact commands;
6. how to run;
7. how to test;
8. what should happen;
9. common errors;
10. next.

Numbers marked ≈ are estimates; we replace them with measurements.

---

## The dependency map

```
 DONE                         PHASE I: FOUNDATIONS         PHASE II: LEARNING FROM EXPERIENCE
 1   language learning        3  runtime + event log ──┬─► 7  memory of world experience ──► 8 continual
 1.5 conditional replies      4  sandbox world ────────┤        ▲ (extends Stage 2)             learning
 1.5b measured learning       5  world model ◄─────────┘        │                                ("sleep")
     (+ round 2: copying,         │                              │                                   │
      earlier turns)              ├─► 6  curiosity ──────────────┘                                   │
 2   long-term memory ✅                                                                              │
     (conversations)                                                                                 │
 2b  learning from experience ✅                                                                      │
     (knowledge, no training)                                                                        │
 2c  world model + reasoning ✅                                                                       │
     (facts as a graph; answers derived, words by the brain)                                          │
 PHASE III: ACTING WITH PURPOSE                                                                      │
 9  grounded language ◄── 5, 7                                                                       │
 10 goals + planning  ◄── 5, 6, 9                                                                    │
 11 regulatory state  ◄── 6, 8, 10                                                                   │
 12 self-model        ◄── 5, 7, 10, 11  ◄────────────────────────────────────────────────────────────┘

 PHASE IV: INTEGRATION AND AUTONOMY
 13 workspace/executive ◄── 7-12      14 reflection + adaptation ◄── 8, 12, 13
 15 real perception     ◄── 5, 8      16 real-world actions ◄── 3, 10, 15
 17 open-ended development ◄── everything

 RESEARCH TRACKS (parallel, optional, each with its own measured test)
 Q1-Q4 quantum and quantum-inspired · C consciousness-indicator audit · R1 faster math (MLX)
 L more language data · L2 teach the brain to use more kinds of recalled memories
```

---

## Done

| Stage | Capability | Why | Measured result | Still missing |
|---|---|---|---|---|
| **0** Setup + data | 32 public-domain books, cleaning, the Originality Guard | Every learned number must come from data we can name | 21 tests; dataset card lists every source | Any learning |
| **1** Neural language learning | Our first network (character MLP) with hand-written backprop; a local chat | Prove we can build and train a network entirely ourselves | Validation loss 1.510 vs 1.943 for the best counting method | Ignores your message (32-character view) |
| **1.5** Conditional conversation | Our transformer, trained on message → reply examples | Replies must depend on the message | Replies depend measurably on the message; 31% on never-seen arithmetic | Weak use of the message; earlier turns steer replies |
| **1.5b** Measured input → output learning | Diagnostics, our subword tokenizer, five data categories, honest labels | Find and fix the real causes of bad replies, with evidence | Never-seen arithmetic 99%; tasks 53% (target 70%: not met) | Copying new names; using earlier turns (1%) |
| **1.5b round 2** Copying and using earlier turns | Copy practice on ~16,000 words; answers from earlier turns; rotary positions + tied output (D-044–D-049) | Using information from the context is what memory retrieval will need | New names copied 93%; answers from an earlier turn 79%; tasks 72%; 9 of 11 criteria met (D-054) | New wordings of statements (12%); "your" vs "my"; memory beyond one conversation; any world to act in |

---

## Phase I: Foundations

### Stage 2: Long-term memory ✅ done (your choice, D-055)

- **Capability:** *remembering what you tell it, across restarts.*
  - Facts, relationships, preferences and events you tell it are stored in SQLite (`state/memory.db`),
    apart from the network.
  - For each new message, at most 3 relevant memories are recalled into the prompt as earlier turns, in
    your own words.
  - You control it: a memory panel, "remember that ...", "forget that ...".
- **Why:** you asked for it before serious chatting. It is software around the existing brain (no
  training), it needs no world, and it is the base that Stage 7 extends to experience in the world.
  Round 2 had just trained the brain to use information from earlier turns.
- **Depends on:** 1.5b round 2 (using earlier turns).
- **Measured by** (`python -m memory.evaluate`, 40 simulated users, names never trained):
  - formation 100% on 40 labeled messages (optimistic: the same person wrote the rules and the examples);
  - the needed memory given to the brain after a restart: **99.1%**; precision 88.8%;
  - unrelated messages recalling nothing: 100%;
  - the brain using a recalled name: 92%; a friend's name: 95%;
  - a pet's name, a sister's name, where you live, favorites, likes: **0%** (a limit of the brain);
  - all of these 0% with memory off.
- **Still missing:**
  - the brain can use only some kinds of recalled memories (track L2 below);
  - formation uses readable rules, not learning;
  - relevance is word-level, not learned meaning;
  - no memory of anything but conversation.

### Stage 2b: Learning from experience ✅ done (your request, D-061)

- **Capability:** *revisable knowledge from conversation.* A small knowledge graph (`learning/`) of what
  you state or teach:
  - entities, facts and connections;
  - a source and confidence for every fact;
  - confirmations;
  - corrections that keep history;
  - conflicts that are flagged.

  Recalled into the brain's prompt as your own sentences, current values only. No training.
- **Why:** experience -> learning is the core of the QI loop; memory gave the system a past, this gives it
  knowledge it can revise.
- **Depends on:** 2.
- **Measured by** (`python -m learning.evaluate`, 30 simulated users, made-up names):
  - the knowledge system 100% on every check (learn, persist, connect, correct, no duplicates, nothing
    from small talk or the model);
  - the right current fact delivered 100%, an old value never;
  - the brain used it 100% for names, pets, corrections and favorites, but 0-3% for capitals, "learn
    that" facts, "what do you know about", "how do I".
- **Still missing:**
  - the brain has never practised most knowledge question forms (a training round would teach them);
  - extraction is readable rules;
  - no learned meaning;
  - no checking of claims beyond source and confidence;
  - no outside sources.

### Stage 2c: A structured world model and basic reasoning ✅ done (your request, D-064 to D-067)

- **Capability:** *answers derived from what is known.* The learned knowledge as a navigable graph
  (`world/`):
  - entities, properties and relationships;
  - current and previous values, sources and confidence;
  - typed values (numbers with units).

  A pipeline in which each step is measured apart: interpretation -> retrieval -> reasoning (lookup,
  chains, corrections, comparisons, known / unknown / conflicting / uncertain) -> the answer as data
  -> the brain writes the reply. Not to be confused with Stage 5: this is a model of what was *told*,
  not a learned predictor of what happens next.
- **Why:** memory and learning store facts but cannot derive anything; planning, curiosity and self-checks
  later need explicit knowledge and an explicit "unknown".
- **Depends on:** 2, 2b.
- **Measured by** (`python -m world.evaluate`, 30 simulated users, 690 questions, made-up worlds):
  - retrieval 100%, reasoning 100% for every kind;
  - the brain expressed the result 30.4% (Stage 2b recall 29.4%, brain alone 5.8%);
  - names, corrections and "from memory" 93-100%;
  - chains 13-23% (from 0-3%);
  - direct world facts, capitals, conflicts, comparisons and skills 0%.
- **Still missing:**
  - the brain has never practised expressing most answers (a training round would teach it);
  - interpretation is readable rules (only the question forms they know);
  - no inference beyond what was said (it doesn't know a golden retriever is a dog).

### Stage 2d: The self-model foundation and the event log ✅ done (your request, D-073)

- **Capability:** *a record of its own state, built only from what happened.* An event log (Stage 3's
  foundation): every message and what it caused, with cause and references, never copies. A self-model
  computed from it and the stores, each line with its evidence:
  - current state;
  - capabilities (demonstrated, failed, available, unavailable, measured);
  - experience;
  - uncertainty (knows, does not know, unsure, inferred);
  - history;
  - environment.

  Questions about recorded state ("What did you learn recently?", "What have you failed at?") get a report.
  Identity questions ("What are you?", "Are you conscious?") are outside it.
- **Why:** curiosity, goals and Stage 12's learned self-model need a record of what the system did and how it
  went, without anyone deciding what the system is.
- **Depends on:** 2, 2b, 2c.
- **Measured by** (`python -m self_model.evaluate`, the real brain, two runs, three made-up people): 22 of
  22 checks each:
  - blank start;
  - evidence resolves (188 of 188);
  - every report matches what happened;
  - the brain's prompt is identical with it on or off (46 of 46);
  - identity questions get no report;
  - forgotten values vanish;
  - replay gives the identical self-model.

  Baselines: the brain alone states what is recorded 0 of 13 times; the expected calibration error of
  "expected success" is 0.09-0.10.
- **Still missing:**
  - it is never shown to the brain (a decision for later, with its own criteria);
  - it records and reports, but predicts nothing about itself (Stage 12);
  - question forms are readable rules.

### Stage 2e: The internal state ✅ done (your request, D-074)

- **Capability:** *a computational state that changes with experience.* Valence, activation, prediction error,
  novelty (each decaying to a baseline with its own half-life), uncertainty, information need and a learning gain,
  updated by a content-free appraisal of verified events, with every change recorded with its evidence. It is the
  foundation of Stage 11 (the regulatory state), built with the signals that exist now.
- **Why:** later stages need a mechanism that changes from experience and changes how the system learns, with no
  personality or emotions written in.
- **Depends on:** 2, 2b, 2c, 2d.
- **Measured by** (`python -m internal_state.evaluate`): 24 of 26 fixed criteria:
  - decay;
  - context;
  - memory tags;
  - recall effects;
  - learning gain;
  - reasoning;
  - persistence;
  - baselines;
  - false-state checks;
  - the blank-slate scan.

  The 2 misses decided that the two memory effects that change what the brain is shown stay off by default. As
  shipped: the memory benchmark is unchanged (0.00 points), and the brain's prompts and replies are identical.
- **Still missing:**
  - it chooses no actions;
  - the appraisal weights are hand-set;
  - expectations are small context tables;
  - no goals, body, social or threat dimensions (deliberately);
  - the brain does not see it.

### Stage 3: The QI runtime (the loop and the event log)

*The event log is built (Stage 2d, D-073). Still to come: the loop, the clock, module plug-ins and replay.*

- **Capability:** *persistence.* QI becomes a process that exists over time, instead of a function
  called per message. A clock, a cognitive cycle and module plug-ins. Every observation, action,
  prediction and utterance goes into one SQLite **event log** with its source and cause. The chat
  becomes one input channel. The page gets an inspector for the log.
- **Why:** every later capability (memory, learning from experience, goals, self-model) needs a
  continuous existence whose experience is recorded. Today only conversations and what you tell it
  survive (Stage 2); everything else happens per message.
- **Depends on:** 2 (its conversation log is where the event log starts); the Stage 1.5b round 2 brain plugs in as the language module.
- **Measured by:**
  - **replay:** re-running a recorded log rebuilds identical state;
  - **provenance:** 100% of outputs trace back to the events and model versions that caused them;
  - **crash safety:** kill and restart with nothing lost;
  - **the chat still passes all its tests.**
- **Still missing:** it learns nothing new from experience, and it has nothing to act on except
  words. **Size:** ≈ 1-2 sittings.

### Stage 4: The sandbox world

- **Capability:** *consequences.* A small grid world we write:
  - a body that can move, pick up, drop, push and toggle;
  - objects (keys, doors, boxes, switches);
  - rules that differ from world to world (which switch opens which door);
  - a *wanderer* that moves on its own;
  - a *noise tile* that flickers randomly (it can never be predicted);
  - a *decoy* that sometimes copies the agent's moves;
  - a simple **language channel** narrating events ("the red door opens").

  New worlds are generated from seeds, so there are always worlds it has never seen. The chat page
  shows the world, and you can play it yourself.
- **Why:** world models, curiosity, goals, planning and agency all need an environment where actions
  change what happens next, and where we know the true rules so we can measure learning exactly.
- **Depends on:** 3 (it plugs in as a channel).
- **Measured by:**
  - rule tests: every rule holds;
  - determinism: the same seed gives the same world;
  - a random-agent baseline recorded (coverage, how often doors get opened by chance);
  - a **world card** documenting everything the world contains, like a dataset card.
- **Still missing:** the agent acts at random. **Size:** ≈ 1-2 sittings.

### Stage 5: The world model (predictive learning)

- **Capability:** *prediction of consequences.* Our transformer, trained on the agent's own
  experience (observation and action tokens), predicts the next observation and narration. An
  ensemble of 3-5 small models gives an **uncertainty** estimate: where they disagree, it doesn't know.
- **Why:** this is the core of QI. Prediction needs no teacher, so every step of experience is a
  training example. It is also the base for curiosity, planning and self-modeling, and the most direct
  measure of functional "understanding": predicting what happens in situations never seen.
- **Depends on:** 3, 4; reuses `brain/` (transformer, rotary positions, training code).
- **Measured by:**
  - next-step accuracy on **never-seen worlds**, vs baselines ("nothing changes", "most frequent
    change");
  - 5- and 10-step imagined rollouts vs what really happens;
  - uncertainty is informative: higher disagreement ⇒ higher error (correlation);
  - it learns the hidden per-world rules from a few steps of experience in that world.
- **Still missing:** its experience comes from random actions; it learns offline, in one go.
  **Size:** ≈ 2-3 sittings.

---

## Phase II: Learning from experience

### Stage 6: Curiosity and active exploration

- **Capability:** *choosing what to learn.* The agent picks actions where its world model is expected
  to improve the most (**learning progress** and ensemble disagreement). It is the first and only
  built-in drive, and it is content-free.
- **Why:** autonomy starts with learning without a teacher choosing the examples. It is also the
  blank-slate-compatible answer to "why would it do anything at all?".
- **Depends on:** 5.
- **Measured by:**
  - **world-model accuracy per step of experience vs a random agent** (same number of steps);
  - coverage of the world;
  - the **noisy-TV test**: time spent staring at the random noise tile must fall, while a pure-surprise
    agent's rises;
  - ablation: curiosity off → slower learning.
- **Still missing:** everything it experiences is either in the training buffer or lost; no lasting
  memory; no goals. **Size:** ≈ 1-2 sittings.

### Stage 7: Memory of experience in the world (extends Stage 2)

- **Capability:** *remembering specific experiences in the world* ("where was the key in the blue
  room?"), in the same store Stage 2 built for conversations. Retrieval adds similarity from our own
  networks' vectors to Stage 2's word-level relevance. Importance comes from surprise and later
  usefulness instead of Stage 2's readable rules. Retrieved memories go into the language module's
  context, as in Stage 2.
- **Why:** learning from experience needs stored experience to replay. Continuity with you needs
  recall. The self-model needs a history.
- **Depends on:** 2 (the memory store), 3 (the log), 5 (vectors), 1.5b round 2 (using context).
- **Measured by:**
  - recall of **planted facts** after delays, other conversations and interference;
  - retrieval precision;
  - **memory on vs off**, in chat and sandbox tasks that need the past;
  - no invented memories: every recalled item links to a real event.
- **Still missing:** memories do not change the networks; nothing is generalized from them.
  **Size:** ≈ 2 sittings.

### Stage 8: Continual learning and consolidation ("sleep")

- **Capability:** *getting better from its own experience without forgetting.* Regular
  consolidation sessions retrain the networks on new experience mixed with replayed memories, with
  evaluation gates, automatic rollback and plasticity protection. Facts that repeat across episodes
  become consolidated knowledge.
- **Why:** without it, anything learned after training lives only in stored records; real
  adaptation means the networks themselves change, safely.
- **Depends on:** 5, 7.
- **Measured by**, on a sequence of different worlds (the standard continual-learning measures):
  - average accuracy over all worlds seen so far;
  - **backward transfer** (forgetting of old worlds must stay small);
  - **forward transfer** (old worlds make new ones faster to learn);
  - **plasticity** (it still learns world 20 as fast as world 1).
- **Still missing:** it learns about the world but has no purposes of its own beyond curiosity.
  **Size:** ≈ 2-3 sittings.

---

## Phase III: Acting with purpose

### Stage 9: Grounded language

- **Capability:** *words that refer to something.* The language module is trained on the sandbox's
  language channel alongside the world model, so it can describe the current state, answer questions
  about the world and about its memories, and follow simple instructions. Today its words only know
  other words.
- **Why:** this makes the chat about QI's actual world and experience instead of book statistics (the
  symbol-grounding problem). It is also needed to give goals in words.
- **Depends on:** 5, 7.
- **Measured by:**
  - question answering about **never-seen** world states vs a text-only model that cannot see the
    world (it should be at chance);
  - accuracy of its descriptions;
  - instruction-following success.
- **Still missing:** multi-step goals need planning. **Size:** ≈ 2 sittings.

### Stage 10: Goals and planning

- **Capability:** *pursuing outcomes over many steps in new situations.* A goal is a desired condition
  in world terms ("the red door is open"). Goals come from three sources, each tagged:
  - **you** (TOLD);
  - **curiosity** (intrinsic);
  - **experience** (states that proved useful before).

  The planner searches imagined futures in the world model (tree search / model-predictive control)
  and re-plans when a prediction fails.
- **Why:** autonomy means getting from here to a goal in situations it has never seen. Planning with a
  model is far more sample-efficient than trial and error, which matters on a laptop.
- **Depends on:** 5, 6, 9.
- **Measured by:**
  - success rate and steps on **never-seen worlds and goals**, vs random and vs a model-free learner
    given the same experience;
  - the effect of planning depth;
  - recovery when the world changes mid-plan.
- **Still missing:** it cannot adjust *how* it learns and acts to the situation. **Size:** ≈ 2-3
  sittings.

### Stage 11: The regulatory state (the emotional-state model, rebuilt for the blank slate)

- **Capability:** *self-regulation.* A few unnamed global variables computed from real signals:
  - surprise;
  - uncertainty;
  - learning progress;
  - goal progress;
  - effort.

  They modulate learning rate, exploration, what gets stored and how far to plan. No emotion names and
  no emotional personality are written in. Whether its state trajectories form stable, emotion-like
  patterns is analyzed, not assumed.
- **Why:** in animals, emotions work partly as control signals that set priorities under limited
  resources (the functional view; 05 §7). A system that adapts *how* it learns should recover faster
  when the world changes.
- **Depends on:** 6, 8, 10 (the signals must exist).
- **Measured by:**
  - **ablation:** with vs without regulation on worlds whose rules change suddenly (recovery speed)
    and under effort limits;
  - clamping a variable must change behavior in the predicted direction (causal test);
  - everything logged and inspectable.
- **Still missing:** it does not model or report its own state. **Size:** ≈ 1-2 sittings.

### Stage 12: The self-model

- **Capability:** *knowing what it is, functionally.* Learned predictors about itself:
  - **agency:** which things it controls (tested with the decoy that copies it);
  - **competence:** "how likely am I to reach this goal?";
  - **knowledge:** "do I know this / can I answer this?";
  - **state:** its own next regulatory state and actions.

  Its self-reports come from these predictors and are checked against the event log.
- **Why:** deciding when to explore, ask for help or refuse needs a model of its own abilities. It is
  the base for self-reflection, and it makes self-reports *caused by* real internal variables.
- **Depends on:** 5, 7, 10, 11.
- **Measured by:**
  - agency attribution accuracy, including never-seen decoys;
  - **calibration** of its success predictions (expected calibration error);
  - self-report accuracy vs the log;
  - ablation: without the self-model, it asks for help and explores less sensibly.
- **Still missing:** it does not yet use its self-model to change itself. **Size:** ≈ 2 sittings.

---

## Phase IV: Integration and autonomy

### Stage 13: The workspace (the executive)

- **Capability:** *combining abilities flexibly.* A limited-capacity "current moment" where the
  modules' outputs compete. The system learns which module to consult next, from how useful past
  consultations were. Example: "where did I leave the key yesterday, and how do I get there?" needs
  memory, then the world model, then the planner, then language.
- **Why:** until now modules are wired in fixed pipelines; broader intelligence needs new combinations.
- **Depends on:** 7-12.
- **Measured by:** success on **never-seen combinations** of sub-tasks vs a fixed pipeline; ablation of
  workspace capacity.
- **Still missing:** it improves what it knows, not how it learns. **Size:** ≈ 2 sittings.

### Stage 14: Reflection and adaptation

- **Capability:** *improving its own learning.* Offline, bounded routines:
  - find its own failure patterns in the log;
  - practice them (in the sandbox or in imagination);
  - tune its own learning settings within safe limits;
  - update the self-model.

  Every change is gated and reversible.
- **Why:** the step from "it learns" to "it learns how to learn"; autonomy in its own development.
- **Depends on:** 8, 12, 13.
- **Measured by:** learning speed on new worlds with vs without reflection, **against a baseline given
  the same extra compute** (so it isn't just "more training").
- **Still missing:** its world is still the sandbox. **Size:** ≈ 2-3 sittings.

### Stage 15: Real perception

- **Capability:** camera and microphone through our own encoders, feeding the same world-model
  machinery: predict the next sensory features. Self-supervised first; your labels later.
- **Why:** grounding in your physical surroundings, using machinery already proven in the sandbox.
- **Depends on:** 5, 8.
- **Measured by:** prediction of real sensory streams vs baselines; recognition of categories you label;
  detecting unusual events.
- **Still missing:** it can see and hear but not act in the room. **Size:** ≈ 3-4 sittings; faster
  with MLX (O-3).

### Stage 16: Real-world actions

- **Capability:** acting beyond the sandbox: the clock, a folder of its own, notifications, perhaps a
  small robot or Arduino. Every action goes through the gateway: permission levels, confirmation,
  STOP, audit log.
- **Depends on:** 3, 10, 15.
- **Measured by:** task success, and **zero safety violations under adversarial tests** (it must be
  impossible to reach a forbidden action).
- **Still missing:** long, unsupervised stretches of life. **Size:** ≈ 2 sittings.

### Stage 17: Open-ended development

- **Capability:** QI runs for days, choosing its own curriculum through curiosity. We measure how its
  capabilities grow over time, and run the scientific experiments:
  - ablations;
  - two copies with different histories: do they develop different dispositions?;
  - a descriptive consciousness-indicator audit.
- **Depends on:** everything.
- **Measured by:** the QI benchmark over time: a capability curve, not a claim.

---

## Research tracks (parallel, optional)

| Track | What | Test that decides |
|---|---|---|
| **Q1** Our quantum simulator | A NumPy state-vector simulator (≤ ~20 qubits), with tests against known results | Correctness tests (e.g. Bell states, interference) |
| **Q2** Tensor-network compression (quantum-inspired) | Replace a world-model layer with a tensor-train layer | Accuracy per parameter vs the normal layer, fixed in advance |
| **Q3** Quantum-probability belief models | Order-dependent belief updates in the self-model | Beats a Bayesian baseline on held-out belief-update tasks, or is dropped |
| **Q4** Variational circuits | A small circuit as a function learner in our simulator | Accuracy and trainability vs a classical network of equal size (no advantage expected) |
| **C** Consciousness-indicator audit | At each stage: which indicator properties the architecture has | Descriptive only; never a claim of experience |
| **R1** Faster math | Apple MLX *array math only* (O-3) | Same results, faster |
| **L** More language | Your own conversations; more public-domain dialogue | Per-category validation, as in Stage 1.5b |
| **L2** Use more kinds of memories ✅ round 3 | Generated MEMORY examples (favorites, likes, pets, family, places, "not told"); 5,000-step continuation into `models/stage1_5b3.npz` | Done: favorites 97.5%, pets 90%, family 100%, places 85%, not-told 71% (from 0-28%); likes 69% NOT MET, arithmetic 92.7% and repetition 6.0% regressed (D-060) |

**No quantum component enters the core system unless it wins its test.**

---

## What comes next (proposed, not started)

Stages 2, 2b, 2c, 2d and 2e are done. Sensible next steps, for you to choose:

- ~~Teach the brain to express reasoning results~~: done in round 4 (D-068, D-069). The brain says the result
  98.1% (from 30.4%). All 31 criteria were met.
- ~~Round 4b~~: done (D-070, D-071). Questions about you through reasoning are 100%, and everything
  together is 98.8%.
- **Memory round 4 (small).** Track L2 is done (round 3, D-060), and the retrieval fixes of D-063 are done.
  What remains:
  - places practised with varied real words;
  - more not-told practice;
  - arithmetic and repetition back within their limits.
- ~~The self-model foundation and the event log~~: done (Stage 2d, D-073).
- ~~The internal state~~: done (Stage 2e, D-074).
- ~~Round 4c~~: done (D-076, D-077). The brain says negative answers 100% (was 55.2%); copying some real words
  slipped (112 -> 108 of 112, e.g. "blue"). A possible follow-up: a redesigned encoding effect (salience from
  prediction error and conflict only) that passes the memory benchmark, so it can be on by default.
- **Stage 3: the rest of the QI runtime** (the loop, the clock, replay), then the sandbox world and the
  world model.
  - The event log (Stage 2d) already records every message and what it caused.
- **Showing recorded-state reports to the brain** (only with a measured reason and your approval). The brain
  alone says what is recorded 0 of 13 times. It would need a training round like round 4, with its own
  criteria.
