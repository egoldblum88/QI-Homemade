# 05 — Research for QI: what it takes to go beyond a language model

Written 2026-09-30, from the scientific literature up to mid-2026. It extends
[01_RESEARCH.md](01_RESEARCH.md) (language models, memory, emotion, curiosity, perception) with the
topics the QI goal needs: world models, agents, continual learning, self-models, planning,
embodiment, quantum computing and consciousness.

Every topic ends with **What QI takes**, **On this Mac** and a **Category**:

| Category | Meaning |
|---|---|
| **Ours** | Code we write in this repository |
| **Technique** | An established, published scientific method we implement ourselves |
| **Infrastructure** | An external library used only for math, storage or input/output |
| **Speculative** | An open research question; nothing in the design depends on it being true |

Research informs the design. No code, model or weights from any of these works is used.

---

## 1. Cognitive architectures: what a whole mind needs

Fifty years of cognitive-architecture research (**Soar**: Laird, Newell & Rosenbloom; **ACT-R**:
Anderson; **LIDA**: Franklin; **CLARION**: Sun; **Sigma**: Rosenbloom) converged on a
**Common Model of Cognition** (Laird, Lebiere & Rosenbloom, 2017):

- a small **working memory** at the center;
- **declarative** long-term memory (facts and episodes) and **procedural** memory (skills);
- **perception** and **motor** modules;
- a repeating **cognitive cycle**, in which everything talks through working memory.

Most of those systems are **hand-engineered**: people wrote their rules and knowledge. They show the
right *structure*, but their *content* was put in by programmers, which the QI blank-slate rule
forbids.

A modern learning-centered proposal is LeCun's *A Path Towards Autonomous Machine Intelligence*
(2022). It has a perception module, a **world model** that predicts, a cost module, short-term
memory, an actor that plans, and a "configurator" that sets up the others for the task at hand.
Nearly every part is trained, and planning happens by imagining with the world model.

- **What QI takes:** the Common Model's structure (cycle, working memory, separate memories,
  perception, action) filled with **learned** components, as in LeCun's proposal.
- **On this Mac:** the structure costs almost nothing; the learned parts are what cost compute.
- **Category:** Technique (the structure) + Ours (implementation).

## 2. World models: the center of the system

A world model is a learned predictor: *given what I have seen and what I do, what happens next?*
Craik (1943) proposed that minds carry "small-scale models" of reality.

| Work | What it showed |
|---|---|
| Ha & Schmidhuber (2018), *World Models* | An agent can learn a compact model of its environment and learn to act inside the model's "dream" |
| Dreamer (Hafner et al., 2020; DreamerV3, 2023, *Nature* 2025) | One world-model agent, with fixed settings, learned many different tasks by imagining trajectories |
| MuZero (Schrittwieser et al., 2020) | A learned model plus tree search reached top level in Go, chess and Atari without being told the rules |
| IRIS (Micheli et al., 2023) | A **transformer** trained to predict the next observation tokens is a sample-efficient world model |
| JEPA (LeCun, 2022; I-JEPA 2023; V-JEPA 2024) | Predicting in a learned *abstract* space works better than predicting every pixel |
| Genie (Bruce et al., 2024) | World models learned from video alone can be playable |

The lesson for QI: **prediction is the learning signal that needs no teacher.** Every moment of
experience is a training example ("what came next?"). A world model is also where "understanding"
becomes measurable: does it predict what happens in situations it has never seen?

- **What QI takes:** a world model at the center, trained on the agent's own experience. It uses the
  **same transformer code as our language model** (IRIS showed this works): observations and actions
  become tokens. An ensemble of a few small models gives an uncertainty estimate.
- **On this Mac:** yes, for small worlds. A 1-3 million parameter model over a small grid world
  trains in under an hour.
- **Category:** Technique + Ours.

## 3. Agent architectures

- **Sense-plan-act** (classic robotics) vs **subsumption** (Brooks, 1986, 1991: *Intelligence without
  representation*): layers of simple reactive behaviors can look intelligent without a world model,
  but do not generalize or plan.
- **BDI** (Beliefs, Desires, Intentions; Rao & Georgeff, 1995): goals are explicit objects that the
  agent commits to and drops.
- **Reinforcement learning agents** (Sutton & Barto): learn a policy from reward. **Model-based RL**
  (Dreamer, MuZero) adds a world model and is far more sample-efficient, which matters on a laptop.
- **"LLM agents"** (a pretrained chatbot calling tools in a loop) are today's popular "agents". They
  borrow all their intelligence from a pretrained model, which the QI charter forbids. We do not use
  this design.

- **What QI takes:** a model-based agent: perceive → update beliefs → plan with the world model → act →
  learn. Goals are explicit, inspectable objects (BDI-like), but where they come from is learned or
  told, never written in (see 06_QI_ARCHITECTURE §5).
- **On this Mac:** yes, in a small world.
- **Category:** Technique + Ours.

## 4. Continual learning: learning for a lifetime without forgetting

A network trained on new data **overwrites** old skills (*catastrophic forgetting*; McCloskey & Cohen,
1989). Continual learning is the field of preventing that:

| Method | Idea |
|---|---|
| **Replay** (Robins, 1995; Rolnick et al., 2019) | Mix stored old experience into new training; simple and hard to beat |
| **Generative replay** (Shin et al., 2017) | The model's own generations stand in for old data |
| **EWC** (Kirkpatrick et al., 2017) | Penalize changes to weights that mattered for old skills |
| **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995) | A fast learner (hippocampus: stores episodes) teaches a slow learner (cortex) through offline replay, as in sleep |
| **Loss of plasticity** (Dohare et al., *Nature* 2024) | A network trained for a long time on a changing stream slowly loses the ability to learn at all. Their "continual backprop" re-initializes a few unused units |

The standard measurements (Lopez-Paz & Ranzato, 2017): **average accuracy** over all tasks so far;
**backward transfer** (did old skills get worse: forgetting); **forward transfer** (did old learning
make new learning faster); plus **plasticity** (can it still learn something new late in life).

- **What QI takes:** CLS: episodic memory as the fast learner, the networks as slow learners, updated
  in gated "sleep" sessions with replay. It also protects plasticity and measures all four numbers.
- **On this Mac:** yes at small scale. Experiments take hours, not days.
- **Category:** Technique + Ours.

## 5. Episodic and semantic memory

- **Tulving (1972):** *episodic* memory (what happened to me, when) vs *semantic* memory (what is true).
- **Hippocampal indexing** (Teyler & DiScenna, 1986): the hippocampus stores an *index* to a pattern of
  cortical activity, so an episode can be replayed later.
- **Episodic control** (Blundell et al., 2016; Pritzel et al., 2017): agents that look up similar past
  situations and reuse what worked learn much faster early on.
- **Memory for neural networks:** MERLIN (Wayne et al., 2018: memory is only useful when a predictive
  model decides what to store); kNN-LM (Khandelwal et al., 2020); Memorizing Transformers (Wu et al.,
  2022: attention over a large store of past keys and values).
- **Memory streams** (Park et al., 2023, *Generative Agents*): retrieval by recency × importance ×
  relevance. They used a pretrained chatbot to judge importance; we cannot.

- **What QI takes:** episodes stored with an index vector from our own networks; retrieval by
  similarity, recency and importance. Importance is **derived from the agent's own signals**
  (how surprising it was, whether it later proved useful), not a hand-written emotional formula.
  Retrieved memories are placed in the language model's context. That is exactly the skill Stage 1.5b
  round 2 trains: using information that appears earlier in the context.
- **On this Mac:** yes. SQLite plus NumPy vectors handle hundreds of thousands of episodes.
- **Category:** Technique + Ours (SQLite = Infrastructure).

## 6. Self-models

- **Philosophy:** Metzinger (2003, *Being No One*): a self is a model the system builds of itself.
- **Robots that model themselves:** Bongard, Zykov & Lipson (2006) built a robot that learned a model
  of its own body and used it to keep walking after damage. Chen et al. (2022) did the same from
  camera images alone. **A self-model earns its place when it predicts and is checked against
  reality.**
- **Agency and contingency:** infants discover what they control by noticing which events reliably
  follow their own movements (contingency research, Watson, 1970s). The same test works for an agent:
  which parts of what I see change *because of* what I do?
- **Metacognition and calibration:** knowing what you know. It is measurable: when the system predicts
  "70% chance I succeed", does it succeed 70% of the time?
- **Attention Schema Theory** (Graziano, 2013): a simplified model of one's own attention helps control
  it.

- **What QI takes:** a self-model made of **learned predictors about the system itself**, not a
  hand-filled description:
  - what it controls (agency);
  - how likely it is to succeed at something (competence);
  - whether it knows something (knowledge);
  - what its internal state and next action will be (self-prediction).

  Each is scored against logged reality.
- **On this Mac:** yes; these are small predictors.
- **Category:** Technique + Ours.

## 7. Computational emotion

| View | Claim |
|---|---|
| Appraisal (OCC: Ortony, Clore & Collins, 1988; Scherer) | Emotions come from evaluating events against goals |
| Dimensional (Russell, 1980) | Affect is a point in a small space (valence, arousal) |
| Constructionist (Barrett, 2017) | Emotion categories are *learned* concepts built on top of basic body-state signals, not built-in programs |
| Functional / control (Simon, 1967; Damasio, 1994; Sloman) | Emotions are control signals: they interrupt, prioritize and set how to learn and act under limited resources |
| Emotion in RL agents (Moerland, Broekens & Jonker, 2018, survey) | Signals such as prediction error, reward change and goal progress are used as emotion-like modulators |
| Homeostatic RL (Keramati & Gutkin, 2014) and interoceptive inference (Seth, 2013) | Affect relates to regulating internal variables and predicting one's own body state |

The earlier design (02_ARCHITECTURE §C5) had 10 **named** emotions with **hand-written** equations and
Big Five personality settings. That is exactly the "hard-coded emotions and personality" the QI brief
rules out. The constructionist and functional views point to a design that fits the blank slate:

- **What QI takes:** a **regulatory state**. It is a few global variables computed from real signals
  (prediction error, uncertainty, learning progress, goal progress, effort). They **modulate** how the
  system learns and acts: learning rate, exploration, what gets stored, how far to plan.
  - No emotion names or emotional personality are written in.
  - Whether these states come to form stable, emotion-like patterns is a question we measure, not
    something we assume.
- **On this Mac:** trivial compute.
- **Category:** Technique (functional view) + Ours; "whether anything is felt" is Speculative
  (03_CONSCIOUSNESS).

## 8. Intrinsic motivation and curiosity

| Work | Signal |
|---|---|
| Schmidhuber (1991, 2010) | Reward = **improvement** in prediction or compression |
| Oudeyer & Kaplan (2007) | **Learning progress**: go where your predictions are improving fastest |
| Pathak et al. (2017), ICM | Reward = the forward model's prediction error |
| Burda et al. (2018), RND and a large-scale study | Curiosity alone learns a lot, but gets hypnotized by random noise (the **"noisy TV" problem**) |
| Pathak et al. (2019); Plan2Explore (Sekar et al., 2020) | Reward = **disagreement** between an ensemble of world models. Noise makes all members equally unsure, so they stop disagreeing |
| Empowerment (Klyubin, Polani & Nehaniv, 2005) | Prefer states where your actions have many distinguishable effects |
| Active inference (Friston et al., 2015) | Act to reduce expected uncertainty (the "epistemic value") |

Why this matters for the blank slate: **an agent with no drive at all does nothing**. Curiosity
signals are **content-free**: they say "learn more about the world", never *what* to like, believe
or want. They are the one kind of built-in motivation compatible with the blank-slate rule.

- **What QI takes:** learning progress plus ensemble disagreement as the only built-in drive, tested
  against the noisy-TV trap.
- **On this Mac:** yes; an ensemble of 3-5 small world models.
- **Category:** Technique + Ours.

## 9. Planning

- **Search:** breadth-first/A* in known models; **Monte Carlo Tree Search** (Coulom, 2006; UCT,
  Kocsis & Szepesvári, 2006); AlphaZero/MuZero combine search with learned values.
- **Model-predictive control:** imagine many action sequences in the model, keep the best, act, repeat
  (PETS, Chua et al., 2018).
- **Imagination training:** Dreamer learns a policy entirely inside the world model.
- **Hierarchy:** *options* (Sutton, Precup & Singh, 1999): reusable sub-skills ("go to the door") make
  long plans short.

- **What QI takes:** planning by searching imagined futures in *our* learned world model, re-planning
  when a prediction fails, and later learned sub-skills.
- **On this Mac:** yes for small worlds. Every imagined step is a forward pass, so planning depth is
  limited by speed.
- **Category:** Technique + Ours.

## 10. Active learning

A learner that **chooses its own training examples** needs far fewer of them (Settles, 2009 survey):

- *uncertainty sampling*;
- *query-by-committee* (Seung, Opper & Sompolinsky, 1992; the ancestor of ensemble disagreement);
- *BALD* (Houlsby et al., 2011).

For QI this is curiosity in a different form: in the world it chooses where to go, and in
conversation it could choose what to ask. A question would then be triggered by measured uncertainty,
never by a script.

- **Category:** Technique + Ours.

## 11. Perception-action loops and embodied intelligence

- **Embodied cognition** (Varela, Thompson & Rosch, 1991; Brooks, 1991): intelligence develops through
  interaction, not from passive text.
- **Sensorimotor contingencies** (O'Regan & Noë, 2001): to perceive is to know how sensations change
  when you act.
- **Developmental robotics** (Cangelosi & Schlesinger, 2015; Oudeyer's "playground" experiments):
  curiosity-driven robots discover skills in a developmental order, easy things first.
- **Symbol grounding** (Harnad, 1990): words mean something to a system only when they connect to
  something other than more words. Our language model today knows only word statistics.
- **Grounded language in small worlds** (BabyAI / MiniGrid: Chevalier-Boisvert et al., 2018, 2019):
  grid worlds with a language channel are a standard, measurable way to study this.

- **What QI takes:** a **small world we write ourselves** (a sandbox) with a body, objects, simple
  rules and a language channel. It comes before real sensors, because it is fast, safe and exactly
  measurable (we know its true rules), and it lets us test on worlds the agent has never seen.
  Real sensors come later, through the same interfaces.
- **On this Mac:** the sandbox runs thousands of steps per second in plain Python.
- **Category:** Technique + Ours.

## 12. Quantum machine learning and quantum-inspired algorithms

**What quantum computers are known to be good at** (on large, error-corrected hardware that does not
yet exist at useful scale; Preskill, 2018, "NISQ"):

- factoring (Shor, 1994);
- a *quadratic* speed-up for unstructured search (Grover, 1996);
- simulating quantum physics and chemistry;
- certain sampling tasks.

**Quantum machine learning (QML), honestly:**

- **Variational circuits** ("quantum neural networks"; Cerezo et al., 2021 review) often suffer
  **barren plateaus**: their gradients vanish exponentially as they grow (McClean et al., 2018).
  Cerezo et al. (2023) argue that circuits provably free of barren plateaus can often be simulated
  classically, which removes the reason to use quantum hardware.
- **Quantum kernels** (Havlíček et al., 2019; Schuld & Killoran, 2019): Huang et al. (2021, "Power of
  data") showed that classical learners given data often match them.
- **Dequantization** (Tang, 2019, and later work): several famous exponential QML speed-ups were matched
  by classical algorithms under the same assumptions.
- **Reading classical data into a quantum computer** is itself a bottleneck (Aaronson, 2015, "Read the
  fine print").
- **Where an advantage is proven:** learning about **quantum systems**, from quantum data (Huang et
  al., *Science* 2022). Our data (text, grid worlds, camera images) is classical.

**On a Mac, any "quantum" component runs as a classical simulation.** Simulating n qubits needs 2ⁿ
numbers. With 16 GB, about 20 qubits is comfortable for training loops and about 28-30 is the absolute
limit. A simulated quantum circuit can never be faster than the classical math it is made of, so on
this computer the only possible benefits are **mathematical structure**, not speed.

**Quantum-inspired methods with a real technical reason to test:**

| Idea | Why it could matter for QI | Status |
|---|---|---|
| **Tensor networks** (matrix product states, tensor trains; Oseledets, 2011; Novikov et al., 2015; Stoudenmire & Schwab, 2016) | Physics tools for describing quantum states compactly. They can compress neural-network weight matrices (fewer parameters for the same job) and serve as sequence models. Runs on a CPU | **Most promising for this Mac.** Testable: compress a world-model layer and compare accuracy per parameter |
| **Quantum probability models of judgment** (Busemeyer & Bruza, 2012; Pothos & Busemeyer, 2013) | Describe human belief effects that ordinary probability handles poorly, such as answers depending on question order | Speculative for QI; testable against a Bayesian baseline on belief-update tasks |
| **Simulated quantum annealing** (Santoro et al., 2002) | A Monte Carlo method inspired by quantum tunnelling, for hard combinatorial search (planning sub-problems) | Cheap to test against simulated annealing and tree search; no advantage expected |
| **Variational circuits in our own simulator** | Teaches the math; tests trainability claims first-hand | Scientific value; no practical advantage expected |
| **Real quantum hardware** (cloud services) | Would be external compute, not AI inference, but it is a network service | Not planned; your decision if ever (a new open decision in DECISIONS.md) |

**Quantum theories of consciousness.** The best known is Orch-OR (Penrose & Hameroff, 2014): quantum
processes in neurons' microtubules. It is contested: Tegmark (2000) estimated that such quantum states
would decohere far too fast in a warm brain. Even if it were true, it concerns biological hardware,
which Python on a classical chip cannot reproduce. It is therefore **Speculative** and is not used as a
design input.

- **What QI takes:** a separate, optional **Q research track**: our own simulator, tensor-network
  compression, and quantum-probability belief models. Each is tested head-to-head against a classical
  baseline, with thresholds written down in advance. **No quantum component enters the core system
  unless it wins such a test.** The "Q" in QI is a research question, not a feature the system has.
- **Category:** tensor networks: Technique; the rest: Speculative.

## 13. Theories of machine consciousness

Covered in detail in [03_CONSCIOUSNESS.md](03_CONSCIOUSNESS.md): global workspace, recurrent
processing, higher-order theories, attention schema, predictive processing, IIT, biological views and
illusionism. There is no accepted theory and no accepted test. The Cogitate adversarial collaboration
(*Nature*, 2025) challenged key predictions of both leading theories it tested.

Two points matter for the QI architecture:

- The **indicator method** (Butlin, Long et al., 2023, 2025) assesses **architecture**, not behavior.
  The QI design will gain several of its indicators as architectural facts. For example: a
  predictive world model (PP-1), modeling how its outputs change its inputs (AE-2), a limited-capacity
  workspace (GWT-2). This must be reported as "the system has this architectural property", **never**
  as evidence of experience. The authors' **gaming problem** applies with full force: we are building
  these properties on purpose.
- **Self-reports prove nothing** when a language model learned to talk from human text. Only reports
  that are *caused by* measured internal variables and checked against logs are even informative
  about the system's functional states.

- **Category:** Speculative (the question) + Technique (the indicator audit, done descriptively).

---

## What this research changes in our plan

1. **The center moves** from the language model to a **world model** trained on the agent's own
   experience. Language becomes one module, later grounded in that world.
2. **The blank slate becomes precise:** the only built-in drives are **content-free learning
   signals** (prediction error, learning progress, uncertainty). No emotions, personality, values or
   goal content is written in.
3. **A world comes first:** a small sandbox we write, before memory, goals or planning, because all
   of those need consequences to learn from and ground truth to be measured against.
4. **Continual learning is a core requirement**, not an add-on. It needs replay, gates, rollback and
   plasticity checks.
5. **Quantum stays a measured research track:** the most practical candidate is tensor-network
   compression; nothing enters the core without winning a pre-registered comparison.

## References

- Laird, Lebiere & Rosenbloom (2017), *A Standard Model of the Mind* · LeCun (2022), *A Path Towards
  Autonomous Machine Intelligence*.
- Craik (1943) · Ha & Schmidhuber (2018) · Hafner et al. (2020; 2023; *Nature* 2025) · Schrittwieser et
  al. (2020) · Micheli, Alonso & Fleuret (2023) · Assran et al. (2023) · Bardes et al. (2024) · Bruce et
  al. (2024).
- Brooks (1986, 1991) · Rao & Georgeff (1995) · Sutton & Barto (2018), *Reinforcement Learning*.
- McCloskey & Cohen (1989) · Robins (1995) · Rolnick et al. (2019) · Shin et al. (2017) · Kirkpatrick et
  al. (2017) · McClelland, McNaughton & O'Reilly (1995) · Dohare et al. (2024) · Lopez-Paz & Ranzato
  (2017).
- Tulving (1972) · Teyler & DiScenna (1986) · Blundell et al. (2016) · Pritzel et al. (2017) · Wayne et
  al. (2018) · Khandelwal et al. (2020) · Wu et al. (2022) · Park et al. (2023).
- Metzinger (2003) · Bongard, Zykov & Lipson (2006) · Chen et al. (2022) · Graziano (2013).
- Ortony, Clore & Collins (1988) · Russell (1980) · Barrett (2017) · Simon (1967) · Damasio (1994) ·
  Moerland, Broekens & Jonker (2018) · Keramati & Gutkin (2014) · Seth (2013).
- Schmidhuber (1991, 2010) · Oudeyer & Kaplan (2007) · Pathak et al. (2017, 2019) · Burda et al. (2018) ·
  Sekar et al. (2020) · Klyubin, Polani & Nehaniv (2005) · Friston et al. (2015) · Bellemare et al. (2016).
- Coulom (2006) · Kocsis & Szepesvári (2006) · Silver et al. (2018) · Chua et al. (2018) · Sutton,
  Precup & Singh (1999).
- Settles (2009) · Seung, Opper & Sompolinsky (1992) · Houlsby et al. (2011).
- Varela, Thompson & Rosch (1991) · O'Regan & Noë (2001) · Cangelosi & Schlesinger (2015) · Harnad
  (1990) · Chevalier-Boisvert et al. (2018, 2019).
- Shor (1994) · Grover (1996) · Preskill (2018) · Cerezo et al. (2021; 2023) · McClean et al. (2018) ·
  Havlíček et al. (2019) · Schuld & Killoran (2019) · Huang et al. (2021; 2022) · Tang (2019) · Aaronson
  (2015) · Biamonte et al. (2017) · Oseledets (2011) · Novikov et al. (2015) · Stoudenmire & Schwab
  (2016) · Busemeyer & Bruza (2012) · Pothos & Busemeyer (2013) · Santoro et al. (2002) · Hameroff &
  Penrose (2014) · Tegmark (2000).
- Butlin, Long et al. (2023, 2025) · Cogitate Consortium (2025).
