# 02 — Architecture of the whole system

> **Updated 2026-09-30 for the QI brief.** The current architecture is
> [06_QI_ARCHITECTURE.md](06_QI_ARCHITECTURE.md). Where the two differ, 06 wins (its §8 lists exactly
> what is kept, changed and dropped). In short: the provenance tags, glass box, bounded dynamics,
> gated learning with rollback, memory stores and action gateway are kept. The named emotions with
> hand-written equations, the Big Five personality, the name and values (§C6), the fictional life
> (§C7), built-in social drives and injected "developer facts" are **dropped**: the QI blank slate
> builds in how to learn, never what to think, want or feel. This file is kept as a record of the
> first design.

This is the blueprint. We will **not** build it all at once. [04_ROADMAP.md](04_ROADMAP.md) splits it
into small stages. Read §A first. The rest is reference you can return to as each stage arrives.

The system has three parts:

| Part | Folder | What it is | Analogy (loose, not a claim) |
|---|---|---|---|
| **Brain** | `brain/` | Our neural language model: tokenizer, transformer, training, generation | The language areas: turns plans into words and notices surprising input |
| **Mind** | `mind/` | Structured systems: memory, emotion state, identity, self-model, goals, curiosity, reflection, learning, workspace | Everything else a mind keeps track of |
| **World** | `world/` | Senses in, actions out, with source tags and safety rules | Eyes, ears, voice, hands |

---

## A. Overview

### A1. The big picture

```
 ┌──────────────────────────────────── THE WORLD ─────────────────────────────────────┐
 │   keyboard · microphone · camera · clock · computer state · sensors · you          │
 └──────────────┬──────────────────────────────────────────────────▲─────────────────┘
                │ raw input                                          │ approved actions
      ┌─────────▼──────────┐                              ┌──────────┴───────────┐
      │ SENSES  (world/)   │                              │ ACTION GATEWAY       │
      │ text, voice, vision│                              │ allowlist · risk     │
      │ → Observations     │                              │ levels · confirm     │
      │   tagged OBSERVED  │                              │ · audit log · STOP   │
      └─────────┬──────────┘                              └──────────▲───────────┘
                │                                                    │
 ┌──────────────▼───────────────────── MIND (mind/) ─────────────────┴────────────────┐
 │                                                                                    │
 │   Appraisal ───► Emotion state ◄──────► Goals & drives ◄──── Curiosity             │
 │      ▲         (10 variables, mood)         │                   ▲                  │
 │      │                │                     │                   │                  │
 │  ┌───┴────────────────▼─────────────────────▼───────────────────┴──────────────┐   │
 │  │          GLOBAL WORKSPACE  (≈5 slots · items compete · winners broadcast)   │   │
 │  └───▲───────────────▲──────────────────▲─────────────────▲─────────────────▲──┘   │
 │      │               │                  │                 │                 │      │
 │   Memory         Self-model         Reflection        Decision         Learning    │
 │   STM · episodic  (structured,      (bounded,         (action          (fast ·     │
 │   semantic ·      attention         triggered)        selection,       medium ·    │
 │   autobiographical schema)                            content plan)    slow/sleep) │
 │                                                            │                       │
 └────────────────────────────────────────────────────────────┼───────────────────────┘
                     ▲ surprise, embeddings                   │ content plan + state
          ┌──────────┴────────────────────────────────────────▼──────────┐
          │ BRAIN (brain/): OUR neural language model                     │
          │ text → tokenizer → embeddings → transformer blocks →          │
          │ output distribution → sampler → text                          │
          └───────────────────────────────────────────────────────────────┘
```

The brain does **not** contain the personality, the memories or the emotions. They live in the mind,
as inspectable data. The brain turns the mind's decisions into language. The mind uses the brain's
signals (surprise, similarity) as part of understanding.

### A2. Design principles

1. **Glass box.** Every decision can be explained: `/why` shows which memories, emotions, goals and
   candidate responses led to the last output.
2. **Provenance everywhere.** Every item carries an epistemic tag (§C1). A guess never silently
   becomes a fact.
3. **Separate systems, one workspace.** Modules are independent Python packages that talk through the
   workspace and a few clear APIs, so each can be tested and replaced on its own.
4. **No hidden scripts.** The agent's words come from the neural model. Any text produced by plain code
   (errors, debug views) is labeled `[system]`, never passed off as the agent speaking.
5. **Bounded dynamics.** Every variable has limits, decay and rate limits. Reflection has a depth
   limit. Nothing can loop forever or spiral.
6. **Controlled learning.** Instant learning happens in memory. Weight changes happen only offline,
   gated and reversible.
7. **Everything local.** No network calls for intelligence. All data stays on this computer.
8. **Measured, not claimed.** Capabilities in the self-model come from tests (validation loss,
   calibration), not from hard-coded boasts.

### A3. Eventual folder structure (it grows stage by stage)

```
ai homemade/
├── README.md
├── docs/                         research, architecture, roadmap, ledger, decisions
├── data/
│   ├── books.csv                 the list of books the AI may learn from
│   ├── raw/                      downloaded public-domain books (unmodified)
│   ├── own/                      YOUR writing (stories, dialogues)
│   ├── processed/                cleaned text, train/val splits, token files
│   ├── backstory/                the fictional life (JSONL), written by you
│   └── DATASET_CARD.md           source + license of every file (generated)
├── tools/                        data scripts: manifest.py, download_books.py, prepare_data.py,
│                                 prepare_dialogues.py
├── brain/                        ── the neural language core (Phases 1–3)   [as built, Stage 1.5]
│   ├── tokenizer.py              characters + special markers <user> <ai> <end>
│   ├── model.py                  Stage 1: MLP language model (forward+backward)
│   ├── layers.py                 Linear, Embedding, LayerNorm, GELU, causal attention, Block
│   ├── transformer.py            Stage 1.5: the transformer language model, masked loss
│   ├── dialogue.py               conversation format, dialogue extraction, arithmetic practice
│   ├── train.py                  Stage 1 training, AdamW, LR schedule, grad clipping
│   ├── train_chat.py             Stage 1.5 training: reading, then replies
│   ├── generate.py               generation: temperature, top-k, loop penalty, stop tokens
│   ├── evaluate.py               Stage 1 vs 1.5: losses, input use, reply selection, arithmetic
│   └── (later) tokenizer_bpe.py, api.py (the interface the mind uses, §B6)
├── mind/                         ── cognition (Phases 4–11)
│   ├── items.py                  Item + epistemic source tags
│   ├── workspace.py              global workspace
│   ├── memory/                   store.py (SQLite), short_term.py, episodic.py,
│   │                             semantic.py, autobiographical.py, retrieval.py, consolidation.py
│   ├── appraisal.py              event → appraisal vector
│   ├── emotion.py                state dynamics
│   ├── identity.py               name, traits, values, preferences, relationships
│   ├── backstory.py              fictional-life importer
│   ├── self_model.py
│   ├── reflection.py
│   ├── goals.py                  goals, drives
│   ├── curiosity.py
│   ├── decision.py               action selection + content planning
│   ├── speech_planner.py         content plan → brain → reranked response
│   └── learning.py               feedback, preferences, sleep consolidation
├── world/                        ── environment (Phases 12–14)
│   ├── observation.py            Observation, Action types
│   ├── channels/                 text.py, clock.py, system.py, audio.py, camera.py, serial_io.py
│   ├── gateway.py                safe action layer
│   ├── voice/                    mfcc.py, vad.py, recognizer.py, formant_tts.py
│   └── vision/                   features.py, motion.py, cnn.py
├── agent.py                      ── the complete loop (Phase 15)
├── chat.py                       text chat interface with inspection commands
├── runs/                         checkpoints, logs, loss plots (not committed)
├── state/                        the agent's live SQLite database + snapshots
└── tests/                        unit tests + gradient checks
```

### A4. Technology

| Need | Choice | Status under the Charter |
|---|---|---|
| Language | Python 3 (you have 3.14.7) | — |
| Matrix math | **NumPy** (uses Apple Accelerate) | Allowed: math infrastructure |
| Storage | **SQLite** (`sqlite3`, built into Python), JSON, NumPy `.npz` for weights | Allowed |
| Tests | `unittest` (built in) | Allowed |
| Plots | matplotlib (optional) | Allowed: graphics |
| Audio I/O (later) | `sounddevice` | Allowed: audio I/O only |
| Camera I/O (later) | OpenCV **capture only**, with none of its bundled models | Allowed: CV infrastructure |
| Hardware (later) | `pyserial` | Allowed |
| GPU speed-up (research) | Apple MLX *array math only*, never `mlx.nn` or its autograd | Allowed as math acceleration, if you approve |
| **Not used** | PyTorch/TensorFlow/JAX layers and autograd, Hugging Face models, any pretrained weights, any AI API | Forbidden |

---

## B. The Brain: our neural language core (Phases 1–3)

### B1. How information moves through it

Shapes are shown for a batch of `B` text windows, each `T` tokens long, model width `d`,
vocabulary size `V`.

```
"Once upon a time"                       text
      │  tokenizer.encode
      ▼
[ 34, 71, 12, 88, ... ]                  token IDs                 shape [B, T]
      │  token embedding lookup  (E: [V, d])
      │  + position embedding    (P: [T_max, d])
      ▼
x                                          vectors                   shape [B, T, d]
      │  ┌──────── transformer block × L ────────┐
      │  │ x = x + Attention(LayerNorm(x))       │   mix information across positions
      │  │ x = x + MLP(LayerNorm(x))             │   process each position
      │  └───────────────────────────────────────┘
      ▼
LayerNorm → x · Eᵀ   (output layer shares E)  logits                  shape [B, T, V]
      │  softmax
      ▼
probabilities for the NEXT token at every position                    shape [B, T, V]
      │
      ├── training:   cross-entropy against the real next tokens → one loss number → backprop
      └── generation: take the last position's distribution → sample → append → repeat
```

### B2. Tokenizer

- **Stage 1–4: character tokenizer.** Vocabulary = every distinct character in the training text
  (~70–100). `encode("hi") → [40, 41]`.
- **Stage 5+: our byte-level BPE.** Start from the 256 byte values, learn ~4,000 merges on the
  training corpus, and save the merges to `tokenizer.json`. Any text, including emoji and typos, can
  be encoded, because the worst case is raw bytes.
- **Special tokens** (reserved IDs, added in the dialogue phase):
  `<pad> <bos> <eos> <user> <ai> <mem> </mem> <sys>`, plus control tokens such as
  `<intent:ask> <intent:share_memory> <mood:low> <mood:high> <arousal:calm> ...` used to *condition*
  generation (§C11).

### B3. The model and what its parameters mean

| Parameter group | Shape (config A) | Meaning |
|---|---|---|
| Token embeddings `E` | [96, 128] | One learned 128-number vector per token. Also reused as the output layer ("weight tying"). |
| Position embeddings `P` | [128, 128] | One learned vector per position 0…127, so the model knows word order. |
| Attention `Wq, Wk, Wv, Wo` (per layer) | 4 × [128, 128] | How each position forms its question, its advertisement, its content, and how head outputs are recombined. |
| MLP `W1, W2` (per layer) | [128, 512], [512, 128] | The per-position "thinking" layer. Much of the pattern knowledge ends up here. |
| LayerNorm `γ, β` (2 per layer + final) | [128] each | Learned rescaling after normalization. |

Configs (from [01_RESEARCH.md §A6](01_RESEARCH.md)):

| Name | Layers | Width d | Heads | Context T | Vocab | Params |
|---|---|---|---|---|---|---|
| A: char-tiny | 4 | 128 | 4 | 128 | ~96 | ~0.8M |
| **B: bpe-small (main)** | 6 | 256 | 8 | 256 | 4,096 | ~5.8M |
| C: bpe-medium (research) | 8 | 384 | 6 | 512 | 8,192 | ~17.5M |

Implementation style: every layer is a small Python class with `forward()`, `backward()`,
`params()` and `grads()`. There is no automatic differentiation. Each backward pass is derived by hand,
explained in comments, and verified by `gradcheck.py`.

### B4. Training pipeline

```
data/raw/*.txt ──clean──► data/processed/{train,val}.txt ──tokenize──► train_ids.npy / val_ids.npy (uint16 IDs)
                                                                                    │
     ┌──────────────────────────────────────────────────────────────────────────────┘
     ▼
 for step in 1..N:
   batch     = B random windows of T+1 tokens           (inputs = first T, targets = shifted by one)
   logits    = model.forward(inputs)                     (forward pass)
   loss      = cross_entropy(logits, targets)            (how surprised were we?)
   grads     = model.backward(dloss)                     (backpropagation)
   clip(grads, max_norm=1.0)                             (safety)
   adamw.step(params, grads, lr=schedule(step))          (learning)
   every 100 steps:  log train loss
   every 500 steps:  validation loss on unseen text; print a generated sample
   every 1000 steps: checkpoint → runs/<run>/ckpt_<step>.npz  (+ config.json, tokenizer, optimizer state)
 after training:  evaluate.py → baselines, novelty, minimal pairs → runs/<run>/report.md
```

Training on **your own data** uses the same pipeline. Put `.txt` files in `data/own/`, list them in
the dataset card, and choose either (a) mix them into pretraining, or (b) fine-tune a finished
checkpoint on them at a low learning rate with replay (§C12).

### B5. Generation

`sample.py` implements temperature, top-k, top-p, repetition penalty, no-repeat-n-gram, max length and
stop tokens ([01_RESEARCH.md §A10](01_RESEARCH.md)). It returns the text **plus** the probability of
each chosen token, which feeds the confidence and surprise signals.

### B6. The Brain API (how the mind uses the brain)

```python
brain.encode(text) -> list[int]
brain.decode(ids) -> str
brain.surprisal(text, context="") -> list[float]  # −log p per token: novelty / "how unexpected"
brain.score(text, context="") -> float             # total log-probability: fluency, reranking
brain.embed(text) -> np.ndarray                    # mean hidden state: similarity for memory search
brain.generate(context, controls, sampling) -> Generation(text, token_probs, stopped_by)
```

**Research extension: the state-conditioned core.** The mind's state vector `s` (10 emotions,
valence, arousal, intent) is turned by two small learned matrices into a scale `γ_l` and shift `β_l`
for each layer `l`: `h ← LayerNorm(h) · (1 + γ_l) + β_l`. Training labels come from running *our own*
appraisal system over the training passages. Emotion then shapes language inside the network itself.
Limitation: those automatic labels are noisy, so the effect may be weak at first. We measure it.

### B7. Built / borrowed / limitation

- **We build:** tokenizers, every layer's forward and backward, loss, AdamW, schedule, training loop,
  checkpoints, sampler, evaluation.
- **Borrowed technique:** backpropagation, embeddings, attention/transformer, LayerNorm, residuals,
  Adam, BPE, sampling methods (all published math, cited in 01_RESEARCH).
- **Limitation:** 1–6M parameters trained on a few million tokens. Local fluency only, no world
  knowledge, weak long-range coherence, CPU-bound training (hours, not minutes, for the main model).

---

## C. The Mind (Phases 4–11)

### C1. The shared currency: `Item` with epistemic source tags

Everything that moves between modules is an `Item`:

```python
Item(
    id="it_20261002_191400_07",
    kind="utterance" | "percept" | "memory" | "fact" | "goal" | "emotion" | "question" | "plan" | "reflection",
    content="The user said they have an exam tomorrow.",
    data={...},                       # structured fields
    source="TOLD",                    # epistemic tag (below)
    confidence=0.9,                   # 0..1
    salience=0.6,                     # how strongly it competes for the workspace
    t_created=..., emotion_tag={"valence": -0.2, "arousal": 0.5},
    links=["ep_000139"],
)
```

| Tag | Meaning | Rule |
|---|---|---|
| `OBSERVED` | Came directly from a sensor or channel | Belief confidence = sensor reliability |
| `TOLD` | Someone said it | Stored as *"user said X"*. Belief in X = trust(speaker) × plausibility |
| `REMEMBERED` | Retrieved from memory | Keeps its original tag; confidence decays with memory strength |
| `INFERRED` | Derived by a rule or model | Confidence ≤ weakest premise × rule reliability; labeled as inference |
| `IMAGINED` | A hypothesis or simulation | **Never** stored as a belief. May be stored as "I imagined X" |
| `GENERATED` | Text our language model produced | Treated like IMAGINED: the model's words are not facts |
| `AUTHORED` | Fictional backstory written by the developer | The agent's in-simulation past; the origin is always visible to you and to `/why` |
| `DEVELOPER` | Facts you set about its nature (e.g. "my emotions are computational variables") | High confidence, protected |

**Contradiction handling:** a new belief that conflicts with an existing one (same subject and
relation, different value) creates an *uncertainty item*. That feeds curiosity (§C10), and may lead
the agent to ask.

### C2. Global workspace and cognitive cycle

- Modules **propose** items with a salience score.
- The workspace keeps the top **K ≈ 5** (capacity limit). A small stickiness bonus stops the focus
  from flickering.
- Winning items are **broadcast**: each module's `on_broadcast(items)` runs, e.g. memory retrieves
  related episodes, appraisal evaluates new percepts, curiosity looks for gaps.
- **Salience** = base × (1 + κ_threat·threat + κ_novel·novelty + κ_goal·goal_relevance + κ_emo·|valence|).
  Emotion changes the κ's (fear raises κ_threat, curiosity raises κ_novel).
- **Attention schema:** the workspace records *what* is in focus and *why* (which term dominated its
  salience). The self-model reads this to answer "what am I paying attention to?".

### C3. Memory system (Phase 4)

Stored in one SQLite database `state/mind.db`. Short-term memory also lives in RAM.

| Store | Contents | Lifetime |
|---|---|---|
| **Short-term** | Last ~20 turns + current workspace items; fades within the session | minutes–session |
| **Episodic** | Specific events: what, when, who, feelings, importance, source | long; decays unless used |
| **Semantic** | Facts and beliefs as triples `(subject, relation, object)` + confidence + evidence (episode IDs) | long; confidence updated |
| **Autobiographical** | Life story: life periods → general events → key episodes; self-defining memories; themes | very long; protected |
| **Procedural** (later) | Learned action preferences (from the bandit) | long; slow change |

**Episode record**

```json
{
  "id": "ep_000142",
  "text": "The user told me they were nervous about an exam.",
  "t": "2026-10-02T19:14:00", "life_period": "present",
  "participants": ["user"],
  "source": "TOLD", "origin": "lived", "confidence": 0.9,
  "importance": 0.62, "strength": 1.0, "access_count": 0, "last_access": null,
  "emotion": {"valence": -0.2, "arousal": 0.5, "top": {"curiosity": 0.4, "fear": 0.2}},
  "appraisal": {"novelty": 0.3, "goal_relevance": 0.6, "...": "..."},
  "consequences": [], "lessons": [], "tags": ["exam", "user_feelings"],
  "protected": false
}
```

Links live in a separate table: `(from_id, to_id, type, weight)`, where type is one of
`same_topic | caused | same_person | same_period | contradicts | summarized_by`.

**Forming memories (encoding).** Every workspace broadcast of an event may create an episode:

```
importance = clip( 0.30·arousal + 0.20·|valence| + 0.20·goal_relevance
                 + 0.15·novelty + 0.10·self_relevance + 0.05·social , 0, 1 )
             + 0.3 if you explicitly say "remember this"
store only if importance > θ_store (default 0.15); otherwise it stays in short-term and fades
```

**Retrieving memories.** Given a cue (current workspace items):

```
score(m) = w_rel·relevance(m, cue)          # our BM25 keyword match; later + cosine of brain.embed
         + w_act·activation(m)              # ACT-R style: ln Σ_j (t_now − t_j)^−0.5 over past uses, squashed
         + w_imp·importance(m)
         + w_mood·congruence(m, mood)       # 1 − |valence_m − mood_valence| / 2
         + w_link·spreading(m)              # sum of link weights from items in the workspace
defaults: w_rel .45, w_act .15, w_imp .20, w_mood .10, w_link .10
w_mood grows with mood intensity (capped at .25), which models mood-congruent recall without runaway rumination
```

Each retrieval strengthens the memory: `strength ← strength + 0.2·(1 − strength)`.

**Forgetting and compression ("sleep" consolidation, run between sessions).**
1. Decay: `strength ← strength · (1 + Δt/τ)^−ψ` (a power law, like human forgetting curves).
2. Weak + unimportant + unprotected episodes are **clustered** by topic, time and people, then
   **compressed** into one gist episode (e.g. "During early October we often talked about weather";
   count = 7). The originals move to an `archive` table. They leave the search index but are not
   destroyed, so forgetting is reversible for you.
3. Repeated facts across episodes are promoted to **semantic** memory with evidence links.
4. Autobiographical structure is updated: periods, key episodes, themes.
5. A **replay set** is selected for optional neural consolidation (§C12).

### C4. Understanding and appraisal

The agent "understands" input through several of our own signals combined. None of them is a
borrowed model.

| Signal | How we compute it |
|---|---|
| Novelty | brain.surprisal z-score vs running average; unseen words and entities; new channel events |
| Pleasantness | A human-made valence lexicon: either one **you** write, or a published human-rated word-norms dataset (e.g. Warriner et al. 2013, NRC VAD; license checked at Stage 8). Not one written by an AI. Later: our own small classifier trained on sentences **you** label, using brain.embed features |
| Dialogue act | question / statement / request / greeting / feedback: rules first, then our own classifier |
| Facts | Pattern extraction ("my name is X", "I like X", "X is Y") → semantic triples tagged TOLD |
| Goal relevance / congruence | Match against active goals (topic overlap + event types such as "question answered" or "user leaving") |
| Agency | Who caused it (self / user / world): subject heuristics |
| Certainty | From source tag + confidence + hedging words ("maybe", "I think") |
| Coping potential | From self-model capabilities: can I do anything about this? |
| Value compatibility | Compare with the identity's values |
| Social | Contact events: greeting, attention, leaving, silence |

Output: an **appraisal vector** `a` (Scherer-style stimulus checks). Honest limitation: understanding
from lexicons, patterns and small classifiers is shallow and brittle. It improves as our own models
improve, and every step is inspectable.

### C5. Emotion state (Phase 6)

**State:** 10 variables in [0,1]: `happiness, sadness, fear, frustration, curiosity, excitement,
loneliness, trust, calmness, surprise`. Derived from them: **core affect** (valence ∈ [−1,1],
arousal ∈ [0,1]) and a slow **mood**. Trust is also kept **per person** in the relationships table.

**Personality → dynamics, not scripts.** Big Five traits set each variable's baseline `b_e`,
sensitivity (gain) `g_e` and time constant `τ_e`. For example, high neuroticism raises the gain on
fear and sadness, and high openness raises the gain on curiosity.

**Update each tick (Δt seconds):**

```
1. decay toward baseline:      x_e ← b_e + (x_e − b_e)·exp(−Δt/τ_e)
       τ: surprise 30 s · excitement 5 min · fear 10 min · frustration 20 min · curiosity 30 min
          happiness 1 h · calmness 1 h · sadness 3 h · (trust: days)
2. drives:                     loneliness ← loneliness + r_L·Δt·(no social contact)     (bounded)
3. appraisal impulses u = g ⊙ M(a):   (M = our appraisal→emotion rules, OCC/Scherer-inspired)
       threat      = relevance·max(−congruence,0)·(1−coping)
       happiness  += relevance·max(congruence,0)·0.6 + max(pleasantness,0)·0.3
       sadness    += relevance·max(−congruence,0)·certainty·(1−coping)·0.6       (loss, irreversible)
       fear       += threat·(1−certainty)·0.7                                    (uncertain future harm)
       frustration+= relevance·max(−congruence,0)·coping·[agency≠self]·0.6       (blocked goal)
       surprise   += novelty·0.8
       curiosity  += novelty·(1−threat)·0.5 + info_gap·0.5
       excitement += relevance·max(congruence,0)·(1−certainty)·0.5               (good thing may happen)
       loneliness −= social·0.5
       calmness   −= (novelty + threat)·0.3
       trust[p]   += 0.05·reliability_evidence − 0.15·violation                  (lost faster than gained)
4. bounded (saturating) add:   x ← x + u·(1−x) if u>0 else x + u·x          → always stays in [0,1]
5. coupling (small matrix C):  fear ⊣ curiosity, fear ⊣ calmness, loneliness → sadness, excitement ⊣ calmness …
6. core affect:                valence = Σ v_e·x_e (normalized), arousal = Σ α_e·x_e (normalized)
7. mood:                       mood ← mood + (Δt/τ_mood)·(core_affect − mood),  τ_mood ≈ 1 day
```

**How the state influences everything else**

| Influences | Mechanism |
|---|---|
| Attention | κ weights in workspace salience (§C2): fear → threats win; curiosity → novelty wins |
| Memory formation | arousal and \|valence\| raise importance (§C3 encoding) |
| Memory retrieval | mood-congruence weight (§C3 retrieval) |
| Goals | fear raises safety/uncertainty-resolution goals; loneliness raises social goals; curiosity raises exploration |
| Decisions | risk aversion = r₀ + k·fear; threshold to ask a question drops with curiosity; frustration raises "change approach" |
| Language | `<mood:…>` control tokens and (research) state-conditioning of every layer; temperature × (1 + 0.3·(arousal − 0.5)); response length |
| Voice (later) | pitch, speed, volume from arousal and valence |

**Grounded self-report.** When the agent talks about its feelings, the content plan must contain the
*actual* top variables. A checker rejects candidate responses that name an emotion whose variable is
below threshold. What it says about its state is then *derived from* the state, not invented by the
language model. This makes the reports accurate readings of a simulation. It does not make them
evidence of experience. The agent also holds a protected `DEVELOPER` fact: "my emotions are
computational state variables; whether anything is felt is unknown."

**Stability safeguards:** hard bounds, decay to baseline, capped mood-congruence, and a cap on how
long any negative variable can stay above 0.8 before recovery dynamics strengthen. This is good
engineering, and it is a precaution given the tragic backstory.

### C6. Identity and personality (Phase 5)

```json
{
  "name": "<you choose>",
  "created": "2026-…", "origin": "An experimental program built from scratch by <you>",
  "traits": {"openness": 0.8, "conscientiousness": 0.6, "extraversion": 0.4, "agreeableness": 0.7, "neuroticism": 0.55},
  "values": [{"value": "honesty", "weight": 0.9, "origin": "developer", "evidence": []}],
  "preferences": [{"about": "stories about the sea", "valence": 0.6, "strength": 0.3, "evidence": ["ep_0031", "ep_0107"], "origin": "learned"}],
  "relationships": [{"person": "user", "familiarity": 0.4, "trust": 0.6, "affection": 0.5, "n_interactions": 12}],
  "long_term_goals": ["understand my past", "learn about the world"]
}
```

**How identity emerges (nothing scripted):**
- **Traits** shape *dynamics* (baselines, gains, decay), which shape behavior indirectly.
- **Preferences are learned:** things repeatedly paired with positive or negative state gain
  preference weight (evaluative conditioning, counted with evidence).
- **Values change slowly**, only through reflection, and only when evidence crosses a threshold.
  Developer-set core values are protected.
- **History** comes from autobiographical memory.
- **The voice** comes from the language model, conditioned on state and on identity features.
- Protected fields (name, origin) can only be changed by you.

### C7. The fictional life (Phase 7)

You write the life story as **individual memory records** in `data/backstory/life.jsonl`
(one JSON object per line), plus `people.json`. The format example below is only an illustration of
the fields. The real content should be written by you, not by an AI (Charter rule 5).

```json
{"id": "bs_007", "title": "(short title)", "event": "(what happened, in your words)",
 "time": {"age": 9, "period": "childhood"}, "location": "(place)",
 "people": ["(person id)"], "importance": 0.9,
 "emotion": {"valence": -0.8, "arousal": 0.7, "sadness": 0.9, "fear": 0.4},
 "consequences": ["(what changed afterwards)"], "lessons": ["(what was learned)"],
 "confidence": 0.8, "cues": ["(words that should bring it to mind)"], "links": ["bs_006"]}
```

**Import pipeline:**
1. **Validate**: schema, chronology, people exist, values in range.
2. **Insert** as episodic memories: `source = REMEMBERED`, `origin = AUTHORED`.
3. **People** → relationships (with history links).
4. **Life periods** → autobiographical structure.
5. **Lessons** → candidate beliefs/values with the memory's confidence.
6. **Gradual integration** (your "coherent history over time"): memories start with accessibility
   proportional to importance. Consolidation passes add links and period summaries. Cue-based recall in
   conversation raises accessibility. The autobiography becomes more connected through use, instead of
   being dumped in all at once.
7. **Optional imprint**: the aggregate emotional weight of the past can shift baselines slightly,
   e.g. a small rise in the fear baseline (bounded, configurable). The past then shapes the
   temperament.

Open decision for you: does the agent *know* its past is authored? (See [DECISIONS.md](DECISIONS.md).)
Either way, the provenance is always visible to you in `/why` and in the database.

### C8. Computational self-model (Phase 9)

A structured object recomputed each cycle **from real subsystem data**, saved as snapshots:

```json
{
  "who_am_i":        {"name": "…", "origin": "…", "age_days": 41, "traits": {"…": 0.0}, "core_values": ["honesty"]},
  "what_i_know":     {"domains": {"animals": {"facts": 23, "avg_confidence": 0.71}}, "vocabulary": 4096,
                      "language_model": {"checkpoint": "B_ep4", "val_loss": 3.12}},
  "what_i_dont_know":[{"question": "What is the user's job?", "source": "info_gap", "priority": 0.4}],
  "what_i_experienced": {"episodes": 412, "life_periods": 5, "key_memories": ["bs_007", "ep_0231"], "recent": "…"},
  "what_is_happening": {"now": "2026-10-02T19:20", "present": ["user"], "channel": "text",
                        "attending_to": [{"item": "it_…", "because": "novelty"}], "topic": "exams"},
  "what_i_want":     [{"goal": "understand the user's exam worry", "priority": 0.7, "progress": 0.2}],
  "uncertain_about": [{"belief": "user likes jazz", "confidence": 0.45, "reason": "single mention"}],
  "what_i_learned":  [{"lesson": "…", "from": ["ep_0199"], "t": "…"}],
  "how_my_state_changed": {"now": {"curiosity": 0.62, "sadness": 0.31}, "vs_1h_ago": {"curiosity": 0.21},
                           "causes": ["ep_0412: user shared news"]},
  "capabilities":    {"can_see": false, "can_hear": false, "can_speak_aloud": false},
  "calibration":     {"stated_80pct_correct_rate": 0.66}
}
```

The self-model is **tested**. Calibration checks whether its confidence matches reality, and
capability flags come from which channels actually work.

### C9. Reflection (Phase 10)

**Triggers:** an event with importance > 0.7; accumulated importance since the last reflection > 2.0;
a large prediction error; end of session.

**Structure** (each field is computed by our code; the language model only phrases the summary):

| Question | Computed from |
|---|---|
| What happened? | the episode(s) + participants + time |
| Why was it important? | the importance factors that were largest |
| How did it change my state? | state snapshot before vs after |
| What did I learn? | prediction vs outcome; repeated patterns; confirmed or refuted beliefs |
| Should I remember it? | importance + novelty + goal relevance → protect, keep, or let decay |
| Did a belief or preference change? | evidence counts crossing thresholds → proposed update, applied with a logged reason |

**Limits:** depth 1 during conversation (a reflection cannot trigger another reflection), depth ≤ 2
during sleep consolidation, at most N reflections per hour, and a fixed compute budget.
Results are stored as `kind="reflection"` memories linked to their sources.

### C10. Goals, drives and curiosity (Phase 11)

```json
{"id": "g_12", "description": "find out what the user's job is", "type": "curiosity",
 "priority": 0.4, "urgency": 0.2, "progress": 0.0, "status": "active",
 "satisfied_when": {"fact": ["user", "job", "*"]}, "parent": "g_learn", "created": "…", "expires": "…"}
```

- **Innate goals:** learn · explore · understand · maintain important memories · complete tasks ·
  resolve uncertainty · keep social contact · keep its values.
- **Drives** (homeostatic levels that create goals when low): knowledge, social contact, coherence
  (memory maintenance), competence (task success).
- **Arbitration:** `U(action) = Σ_g priority_g·urgency_g·E[progress_g | action] − cost(action)
  + emotion_bias(action) + learned_preference(action, context)`.

**Curiosity pipeline** (your diagram, made concrete):

```
UNKNOWN INFORMATION   ← surprisal spikes · unknown words/entities · low-confidence beliefs ·
        │               contradictions · open questions · gaps in the self-model's knowledge map
        ▼
CURIOSITY (score)     = gap_size × relevance × expected_learning_progress   (noisy-TV guard)
        ▼
QUESTION/EXPLORATION  → candidate action "ask", rate-limited (max 1 question per few turns)
        ▼
NEW INFORMATION       → tagged TOLD / OBSERVED
        ▼
LEARNING              → belief confidence up · uncertainty item closed · intrinsic reward
        ▼
MEMORY                → episode + semantic fact + learning-progress statistics updated
```

### C11. Decision and response generation

**Actions:** `respond · ask_question · share_memory · express_state · clarify · stay_silent ·
reflect · (later) world actions via the gateway`.

Pipeline for speaking:

```
1. decision.py picks the intent (highest utility, see C10)
2. content plan = {intent, key items from workspace: memory snippets, facts, the question,
                   the actual emotion readings, style: mood/arousal}
3. context for the brain (fits in the context window, e.g. 256 tokens):
      <bos> <mood:low> <arousal:calm> <intent:share_memory>
      <mem> (1–2 short retrieved snippets) </mem>
      (last few turns)  <user> (message) <ai>
4. brain.generate × N candidates (N = 4–8), sampling settings adjusted by state
5. rerank (our own scoring):  fluency (brain.score) + plan coverage + grounded-self-report check
                              − repetition − contradiction with memory
6. output the best one, tagged GENERATED; log plan, candidates and scores for /why
```

This is not "a giant personality prompt". The context holds a few control tokens and one or two
memory snippets chosen *by the mind*. Personality lives in state, memory and learned preferences.

Limitation: a small model will often ignore the plan or drift. The reranker helps. Dialogue
fine-tuning (§C12, Stage 9) is required before it can converse at all.

### C12. Learning from experience (Phase 8): three speeds

| Speed | What changes | When | Safety |
|---|---|---|---|
| **Fast** | Memories, belief confidences, relationship values, emotion state | Instantly, every event | Bounded values; provenance |
| **Medium** | Preference weights; action-selection bandit (which intents work in which contexts) from feedback (`/good`, `/bad`, or appraised reactions) | Every interaction, small learning rate | Logged; reversible; decays toward prior |
| **Slow** | **Neural network weights** | Only in an offline **sleep consolidation** you start manually | Gated: see below |

**Sleep consolidation (neural) protocol:**
1. Select a curated set: important conversations and reflections. You approve the list.
2. Mix in **replay** of the original corpus (e.g. 80% old / 20% new) to prevent catastrophic forgetting.
3. Fine-tune at a low learning rate for a few hundred steps (research option: train only small
   low-rank "experience adapters", so the base weights never change).
4. Evaluate: validation loss on the original corpus, a probe set of the new material, and samples.
5. **Accept** only if old-skill loss worsens by less than ε. Otherwise **roll back** automatically.
6. Version everything: `runs/brain/v1 → v2 → …`, with a changelog in the self-model ("what I learned").

---

## D. The World (Phases 12–14)

### D1. Environment interface

```python
Observation(channel="camera", t=..., kind="motion", data={"region": "left", "amount": 0.7},
            source="OBSERVED", confidence=0.8)
Action(kind="speak", params={"text": "..."}, risk=0, requested_by="decision", reason="...")
```

| Channel | Input | Built by us | Phase |
|---|---|---|---|
| text | keyboard | yes | 9 |
| clock | time of day, time since events | yes | 12 |
| system | battery, CPU, active hours | yes (small library for readings) | 12 |
| audio | microphone → our VAD → our recognizer | yes | 13 |
| camera | frames → our feature extractors | yes | 14 |
| serial | Arduino/sensors via a message protocol we define | yes | 12+ |

### D2. Action gateway (safety)

| Level | Examples | Rule |
|---|---|---|
| 0 | print text, speak, update its own memory | automatic |
| 1 | write a file inside `state/sandbox/`, move a servo within limits | automatic, logged |
| 2 | open an app, play a sound file, any network access | **asks you every time** |
| 3 | delete files, send messages, purchases, system settings | **forbidden**: not in the allowlist |

The agent never gets a shell. Every action is logged with its reason. A `STOP` file or keyboard
command halts all actions immediately. There is a dry-run mode for testing.

### D3. Voice (Phase 13)

```
MICROPHONE → sounddevice → our VAD (energy + zero-crossings) → our MFCC features
  → recognizer v1: DTW templates of YOUR voice (a few dozen words)
    recognizer v2: our small CNN keyword model (Speech Commands + your recordings)
    recognizer v3 (stretch): our CTC acoustic model + our language model as prior
  → text (tagged OBSERVED with recognition confidence) → MIND → … → response text
  → our formant synthesizer (Klatt-style rules; pitch/rate/volume from emotion state) → SPEAKER
```

### D4. Vision (Phase 14)

```
CAMERA → capture (OpenCV I/O only) → our processing:
   brightness · dominant colors · motion (frame differencing) · background model · blobs & tracking
   → later: our small CNN trained on images YOU label
 → STRUCTURED OBSERVATIONS  e.g. {"lights": "off", "motion": "left, strong", "someone_present": 0.7}
 → MIND (appraisal · memory · goals) → DECISION
```

Vision stays **separate** from language. It produces structured observations, not text for the language model to imagine over.

---

## E. The complete loop (Phase 15)

```python
while running:
    obs     = world.perceive()                         # PERCEIVE (all channels, tagged OBSERVED)
    items   = interpret(obs)                           # INTERPRET: appraisal, dialogue act, facts, surprisal
    ws.propose(items); ws.propose(memory.retrieve(ws.focus()))   # RETRIEVE MEMORY
    self_model.update(ws, emotion, goals, memory)      # UPDATE SELF MODEL
    emotion.update(dt, appraisals)                     # UPDATE EMOTIONAL STATE
    goals.update(emotion, self_model, ws)              # CHECK GOALS (drives, curiosity)
    options = reason(ws, memory, goals)                # REASON: rule inference, analogies, IMAGINED hypotheses
    action  = decide(options, goals, emotion)          # DECIDE: utility + learned preferences
    result  = gateway.execute(action)                  # ACT (safety-checked)
    outcome = world.observe_result(result)             # OBSERVE RESULT
    learning.fast_update(outcome); learning.medium_update(outcome)   # LEARN
    memory.encode(ws, outcome, emotion)                # STORE MEMORY
    if reflection.triggered(): reflection.run(limit=1) # REFLECT WHEN NECESSARY (bounded)
    sleep_until_next_tick()                            # event-driven for chat; ~1–4 Hz for senses
```

When nothing is happening, the loop keeps running slowly. Loneliness rises, curiosity may queue a
question for when you return, and at night (or on command) it runs sleep consolidation.

**"Reasoning", honestly:** our small language model cannot reason step by step the way large
models appear to. Reasoning here is (1) rule-based inference over semantic memory (e.g. "A is part of
B, B is in C → A is in C", tagged INFERRED), (2) analogy: "last time something similar happened,
X followed", (3) utility-based planning over goals, and (4) the language model as an *imagination*
engine whose outputs are tagged IMAGINED until checked. Research track: train our own model on
reasoning tasks generated by our own programs (arithmetic, logic puzzles).

---

## F. Persistence

`state/mind.db` (SQLite) tables: `episodes, links, facts, periods, people, identity, values,
preferences, goals, questions, reflections, state_log, self_model_snapshots, feedback, observations_log,
actions_log, archive`. The brain lives in `runs/brain/<version>/` (`weights.npz`, `config.json`,
`tokenizer.json`, `train_log.csv`, `report.md`).

## G. Inspecting it (the glass box)

Commands in `chat.py`: `/state` (emotion bars and mood) · `/self` (self-model) · `/memories <query>`
(retrieval with score breakdown) · `/goals` · `/why` (full trace of the last response) · `/good`, `/bad`
(feedback) · `/remember` · `/sleep` (run consolidation) · `/brain` (model info, sample). Later a small
local web dashboard (graphics library only).

## H. Your phases → where they live

| Phase | Module(s) | Roadmap stages |
|---|---|---|
| 1 First brain | `brain/` layers, model, tokenizer | 1–4 |
| 2 Training | `brain/data.py, train.py, evaluate.py` | 2–5 |
| 3 Generation | `brain/sample.py` | 6 |
| 4 Memory | `mind/memory/` | 7, 11 |
| 5 Identity | `mind/identity.py` | 10 |
| 6 Emotion | `mind/appraisal.py, emotion.py` | 8 |
| 7 Fictional life | `mind/backstory.py` | 10 |
| 8 Learning | `mind/learning.py` | 14 |
| 9 Self-model | `mind/self_model.py` | 12 |
| 10 Reflection | `mind/reflection.py` | 12 |
| 11 Curiosity & goals | `mind/goals.py, curiosity.py, decision.py` | 13 |
| 12 Grounding | `world/`, epistemic tags in `mind/items.py` | 11, 15 |
| 13 Voice | `world/voice/` | 16–17 |
| 14 Vision | `world/vision/` | 18 |
| 15 Complete loop | `agent.py`, `mind/workspace.py` | 15, 19 |

## I. The honest summary of limitations

1. **Language:** fluent-ish for short stretches, often incoherent beyond that; no broad knowledge;
   conversation only after dialogue fine-tuning, and still simple.
2. **Understanding:** lexicons, patterns and small classifiers: shallow but inspectable.
3. **Reasoning:** mostly explicit code, not emergent in the network.
4. **Emotions:** simulated state variables whose initial equations we chose by hand, not discovered.
5. **Perception:** weak when fully from scratch; recognizes little at first.
6. **Compute:** CPU training means hours per model. Big jumps in quality need much more data and
   compute than a laptop provides.

What it **will** have that most chatbots do not: a persistent memory that it curates, a real
state system that measurably changes its behavior, goals, curiosity from real uncertainty signals, a
structured self-model checked against reality, provenance on everything, and a brain whose every
weight was learned here from data you can name.
