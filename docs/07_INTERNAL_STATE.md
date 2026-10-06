# 07 — Internal state (computational emotion): research and design

> **Status: built (2026-10-02, D-074)**, as approved, with your three choices:
> - valence = own verified checks + information gained;
> - recall reordering kept on only if the memory benchmark stays within 1 point of state-off;
> - slowly shifting baselines on, capped and resettable.
>
> It brings the roadmap's Stage 11 ("the regulatory state") forward, using the signals that exist today:
> reasoning, learning, memory and the replies' verified checks.
>
> **Changed while building, each with a measured reason:**
> 1. **Valence is a moving average of each episode's signal**, not a sum (5.2). With the sum, the first
>    measurement showed valence pinned at +1.0 after 8 messages: a new system beats its 0.5 priors on every
>    check, and frequent messages add up faster than decay removes them. As a moving average it stays within
>    the range of its inputs. The information weight became 0.6 relative to the operation errors (the same ratio
>    as 0.3 / 0.5).
> 2. **"Before" values exclude the episode's own new facts.** Learning stores a message's facts before the
>    appraisal runs, so the information need "before" a contradiction already contained it. It is now computed
>    without them (and with the facts they replaced).
> 3. **The code scan skips the two measurement tools** (`evaluate.py`, `simulate.py`). They never run in the
>    chat, and they must contain test inputs such as an emotional reply, like the `tests/` folder. Every module
>    that runs in the program is scanned.
>
> 4. **The two memory effects that change what the brain is shown are OFF by default** (your rule, measured).
>    With importance at encoding (M1) on, the memory benchmark's "What do I like / love?" found the right memory
>    28.7% instead of 100%. The first memory of each kind is the most novel, so its boost pushed later, equally
>    relevant ones out of the 3 recalled. The recall-order bonus (M2) changed nothing further. Both stay built,
>    tested and switchable (`python -m chat.server --state-memory-effects`). Tags (the record of significance at
>    encoding) and M3 (recall moves the state, never what the brain sees) are on. As shipped, the memory benchmark
>    differs from state-off by 0.00 points, and the real brain's prompts and replies are identical with the state
>    on and off.
>
> Results (24 of 26 criteria; the 2 misses are the M1/M2 decisions above):
> [models/internal_state_evaluation.md](../models/internal_state_evaluation.md).

**What this is:** a small set of numbers that change when real events happen, return toward a baseline over
time, and measurably change three things:
- what gets stored as important;
- which memories are recalled first;
- how fast the system's own expectations learn.

**What this is not:**
- not a personality;
- not a list of emotions;
- not feelings;
- not a prompt.

The numbers are never turned into sentences like "I feel sad", and the brain never sees them. Calling one of
them "valence" names a computed quantity. It is not evidence that anything is felt ([03](03_CONSCIOUSNESS.md)).

Sources: the established literature named below, from my own knowledge (no web search was used for this
document). Each source is a well-known paper or book you can look up.

---

## 1. Research summary

### 1.1 Appraisal theory
- **The claim:** emotions follow from *evaluating an event's significance*, not from the event itself. So the
  same event can produce different states in different contexts.
  - Lazarus (1991), *Emotion and Adaptation*.
  - Scherer's component process model: a sequence of "stimulus evaluation checks" (novelty, intrinsic
    pleasantness, goal conduciveness, coping potential, norm compatibility). Scherer (2009), *Cognition &
    Emotion*.
  - The OCC model: emotions as valenced reactions to events, agents and objects, relative to goals,
    standards and attitudes. Ortony, Clore & Collins (1988).
- **Evidence:** appraisal dimensions predict which emotion people report (Smith & Ellsworth, 1985, and many
  replications). This is correlational. Whether appraisal *causes* emotion in the brain is debated.
- **For us:** appraisal is what gives context dependence. But the classic checks need *goals, standards
  and attitudes*, and those would be hard-coded values here. Only the **content-free** checks fit the
  blank slate:
  - novelty;
  - discrepancy from expectation;
  - expected success ("coping potential");
  - information change.

### 1.2 Computational emotion models (engineering)
- **EMA** (Gratch & Marsella, 2004): appraisal over the agent's own plans and beliefs, with coping strategies.
- **WASABI** (Becker-Asano & Wachsmuth, 2010): appraisal pushes a continuous mood/affect space (PAD) that
  decays back over time.
- **FAtiMA** (Dias, Mascarenhas & Paiva, 2014): a modular appraisal architecture.
- Survey: Marsella, Gratch & Petta (2010), "Computational models of emotion".
- **Lesson:** the successful engineering pattern is **hybrid**: discrete appraisal events drive a continuous
  state that decays. All of these models, however, were built to *display* human-like emotion. They hard-code
  human goals, personality parameters and emotion labels, which is what we must avoid.

### 1.3 Affective computing
- Picard (1997), *Affective Computing*. The field is mostly about *recognizing* and *expressing* human emotion
  (faces, voice, text). Recent work with language models mostly concerns generating emotional *text*.
- **For us:** almost none of it applies. We are not recognizing your emotions or styling replies. The
  relevant part is Picard's argument that affect-like signals can serve as *control* in a machine.

### 1.4 Homeostasis, allostasis and internal drives
- **Homeostatic models:** variables with set points, and behavior that reduces deviation from them.
  - Cañamero (1997): motivated agents with internal variables.
  - Keramati & Gutkin (2014, *eLife*): reward defined as reduction of deviation from a set point
    ("homeostatic reinforcement learning").
- **Allostasis:** predictive regulation, where set points themselves adapt. Sterling (2012). Barrett's "body
  budget".
- **Interoceptive inference:** affect as the brain predicting its own body state. Seth (2013, *TICS*).
- **For us:** we have no body and no real resource that runs out, so a built-in "energy" or "social" need
  would be invented. It would be a predetermined motive, which the blank slate forbids.
  - What we take is the **mechanism**: baselines (set points), decay toward them, and slow adaptation of the
    baselines.
  - The open decision O-7 (a homeostatic variable for the future sandbox body) stays open.

### 1.5 Reinforcement learning, reward and state
- **Prediction error:** dopamine neurons signal reward prediction error, the difference between what was
  expected and what happened (Schultz, Dayan & Montague, 1997). Temporal-difference learning: Sutton & Barto.
- **Emotion in RL** (survey: Moerland, Broekens & Jonker, 2018, *Machine Learning*): emotion-like signals are
  derived from prediction error, value and its change. Broekens (2018), "A temporal difference
  reinforcement learning theory of emotion": joy and distress map onto signed TD errors.
- **Mood as momentum:** two closely related models.
  - Rutledge, Skandali, Dayan & Dolan (2014, *PNAS*): momentary happiness is well fitted by an exponentially
    decaying sum of recent expectations and reward prediction errors.
  - Eldar, Rutledge, Dolan & Niv (2016, *TICS*): mood integrates recent prediction errors and biases how new
    outcomes are perceived.
- **For us:** a well-supported, computational definition of a valence-like variable is **the decaying sum of
  recent signed prediction errors**. It needs no content-specific liking: only an expectation and an outcome.

### 1.6 Intrinsic motivation, novelty, uncertainty, information
- **Learning progress and compression progress:** Schmidhuber (1991; 2010). Oudeyer, Kaplan & Hafner (2007).
- **Prediction-error curiosity:** Pathak et al. (2017), ICM. Burda et al. (2018), RND.
- **Count-based novelty:** Bellemare et al. (2016). Novelty is about 1/√(1 + times seen).
- **Bayesian surprise attracts attention:** Itti & Baldi (2009).
- **Valence as change in uncertainty:**
  - Joffily & Coricelli (2013, *PLoS Comput Biol*): valence as the rate of change of free energy
    (uncertainty).
  - Hesp et al. (2021, *Neural Computation*): valence as change in confidence.
  - Kiverstein, Miller & Rietveld (2019): "error dynamics", where improving prediction feels good.
- **For us:** novelty, surprise (unsigned prediction error) and information gained are content-free signals
  that can be computed from our own records. **Information gain as positive** is the one evaluative assumption
  this design makes (see 5.4). It is mainstream in intrinsic-motivation research, but it *is* an assumption.

### 1.7 Dimensional models
- **Core affect:** two broad dimensions (valence and arousal) organize self-reported human affect.
  - Russell (1980): the circumplex model.
  - Russell (2003, *Psych. Review*): "core affect".
  - Mehrabian & Russell (1974): PAD adds dominance.
- **Evidence:** the two-dimensional structure is one of the most replicated findings in affect research.
- **For us:** a small continuous state is the right *shape*. But the dimensions must be **defined by
  computation**, not borrowed as words, because "arousal" has no meaning for a program until we say what it
  measures.

### 1.8 Categorical emotions
- **Basic emotions as built-in programs:** Ekman (1992).
- **Counter-evidence:** a large review finds that facial movements do not reliably map onto emotion categories
  across people and situations (Barrett, Adolphs, Marsella, Martinez & Pollak, 2019, *Psych. Science in the
  Public Interest*).
- **The constructionist view** (Barrett, 2017): categories are *learned concepts* built on top of core affect.
- **For us:** a fixed list of human emotions is (a) scientifically contested and (b) exactly the predetermined
  emotional profile the blank slate rules out. If categories ever appear, they should be *found* as recurring
  patterns in the logged state (an analysis), never written in.

### 1.9 Emotion as a control mechanism
- Simon (1967): emotions interrupt and re-prioritize processing. Damasio (1994): somatic markers.
- Doya (2002, *Neural Networks*): neuromodulators as meta-parameters.
  - dopamine: prediction error;
  - serotonin: time horizon;
  - noradrenaline: exploration;
  - acetylcholine: learning rate.
- Yu & Dayan (2005, *Neuron*): expected vs unexpected uncertainty set how much to trust new evidence.
- **Learning rate from surprise:**
  - Pearce & Hall (1980): "associability". After surprising outcomes, the learning rate is high.
  - Behrens et al. (2007, *Nat. Neurosci.*): humans raise their learning rate when the world is volatile.
- **For us:** this is the functional core. The state should **change how the system learns and what it
  stores**. That is measurable, causal and needs no claim about feelings.

### 1.10 Memory and emotion
- **Arousal at encoding strengthens consolidation:** McGaugh (2004, *Annu. Rev. Neurosci.*), via the
  amygdala's modulation of the hippocampus. Strong animal and human evidence. It is about *arousal*
  (significance), not positivity.
- **Mood-congruent recall:** Bower (1981). A meta-analysis finds a real but moderate effect (Matt, Vázquez &
  Campbell, 1992). **Mood-dependent memory** (recall is better in the same state as encoding) is weaker and
  less reliable (Eich, 1995).
- **Rumination:** congruent recall can feed back into the state.
- **For us:**
  - significance at encoding raises importance (well supported);
  - state-congruent recall gets only a *small, bounded* ordering effect (weaker evidence);
  - recall feeding back into the state gets explicit guards against loops.

### 1.11 Persistence, decay, inertia, adaptation
- **Emotions are short, moods are longer:** different time scales (Frijda; Russell 2003).
- **Emotional inertia** (Kuppens, Allen & Sheeber, 2010): affect is autocorrelated over time. A partial update
  plus exponential decay produces it.
- **Hedonic adaptation:** responses to repeated events shrink, and set points move (Brickman & Campbell, 1971;
  Frederick & Loewenstein, 1999). Diener, Lucas & Scollon (2006) show the set point is not fixed: lasting
  changes do occur.
- **For us:**
  - a **separate decay rate per dimension**;
  - partial updates (inertia);
  - habituation through learned expectations and count-based novelty;
  - an optional, slow, bounded **baseline adaptation**: the experimental "individual development"
    mechanism.

### 1.12 How strong is each idea? (supported / engineering approximation / speculative)

| Idea | Status |
|---|---|
| A low-dimensional (valence, arousal) structure in human self-reported affect | **Supported** (decades of replication) |
| Signed prediction errors drive momentary mood, with exponential decay | **Supported** in lab and app studies (Rutledge 2014), for humans |
| Significance/arousal at encoding strengthens memory | **Supported** (McGaugh) |
| Learning rate rises after surprise or in volatile conditions | **Supported** (Pearce-Hall; Behrens 2007) |
| Habituation to repeated events; slow set-point change | **Supported**, with individual variation |
| Appraisal dimensions predict emotion reports | **Supported** (correlational) |
| Fixed basic-emotion categories as natural kinds | **Contested** |
| Mood-congruent recall | **Moderate**; mood-dependent memory is weaker |
| Using "outcome vs expectation of the system's own checks" as valence | **Engineering approximation** (ours) |
| Count-based novelty, table-based expectations, our half-lives and gains | **Engineering approximation**, hand-set and then measured |
| Valence as the rate of change of uncertainty (free energy) as a general theory | **Speculative / theoretical** |
| Baseline drift as "personality development" | **Speculative**: we only measure whether the dynamics change |
| That any of these variables is felt | **Unknown; never claimed** |

---

## 2. Model comparison

| | 1 Categorical | 2 Dimensional | 3 Appraisal | 4 Homeostatic / drives | 5 Hybrid |
|---|---|---|---|---|---|
| **Representation** | A label from a fixed list (joy, fear, ...) | A point in a small continuous space | Event evaluations along checks | Internal variables with set points and needs | Appraisal → continuous state → decay to (adaptive) baselines |
| **Scientific support** | Contested as natural kinds | Strong for structure | Strong (correlational) | Strong in physiology; we have no physiology | Combines the supported parts |
| **Context dependence** | Only via rules we write | None by itself | **Yes**, the core idea | Via the current deficit | **Yes** |
| **Blank-slate fit** | **Poor**: a predetermined emotional profile | Good, if the dimensions are defined by computation | Classic versions need goals and values: only content-free checks fit | Built-in needs are predetermined motives: only the mechanism fits | **Good**, when restricted to content-free checks and mechanisms |
| **Traceable to evidence** | Weak (why this label?) | Medium | **High** (each check has an input) | Medium | **High** |
| **Influences behavior** | Via scripts (risky: becomes a response script) | Via couplings | Via coping | Via drive reduction | Via explicit, bounded couplings |
| **Risk** | Anthropomorphic claims; scripts | Meaningless numbers if undefined | Hidden values | Invented needs | More parts to test |

**Choice: 5, hybrid, restricted to the blank slate.**
- **Appraisal:** of *verified* events, with content-free checks only (expectation discrepancy, novelty,
  information change, conflict).
- **State:** a small **dimensional** state whose every dimension is defined by a computation.
- **Homeostatic-style dynamics:** per-dimension decay toward baselines that can adapt slowly.
- **No categories, no drives, no goals.**

This was chosen because it is the only option that is context-dependent, traceable, measurable and free of
predetermined motives. It was not chosen because it looks human.

---

## 3. Recommended architecture

```
 VERIFIED EVENTS (the event log: your messages' meta-data, reasoning results, verified reply checks,
 knowledge changes, memories formed, recalls)          <- never the words of generated text
        |
        v
 APPRAISAL (internal_state/appraisal.py)       content-free checks, each with its evidence:
        |   discrepancy (outcome vs learned expectation), novelty, information change, conflict
        v
 EXPECTATIONS (internal_state/expectations.py) small learned predictors of the system's own outcomes;
        |   their learning rate is the "learning gain" (Pearce-Hall)
        v
 INTERNAL STATE (internal_state/state.py)      valence, activation, surprise, novelty   (decaying)
        |                                      uncertainty, information need            (computed)
        |                                      learning gain                             (control signal)
        |                                      baselines                                 (slow, experimental)
        v
 EFFECTS (explicit, bounded, switchable, logged):
   M1 memory importance at encoding   M2 recall ORDER among relevant memories   M3 recall -> state
   L1 learning rate of expectations   (nothing else; never the reply's words)
        |
        v
 STATE HISTORY (state/internal_state.db): every change, with the event ids that caused it
 SELF MODEL: reports the state, its changes, causes and evidence      BRAIN: never sees any of it
```

**Separation rules:**
- **Internal state** is its own package and its own database.
- **Memory, learning, reasoning:** their rules are unchanged. It talks to them only through three named
  couplings (M1, M2, L1).
- **Self-model:** reads it (like memory and knowledge) and never writes it.
- **Brain:** gets nothing.
- **Off switch:** `--no-state` turns it off. With it off, everything behaves exactly as today.
- **Depends on the event log:** with the self-model off, there is no evidence, so internal state is off too.

---

## 4. Exact state representation

### 4.1 The dimensions

| Dimension | Range | At start | Computed as | Decay (half-life, configurable) |
|---|---|---|---|---|
| **valence** | -1 .. 1 | 0 | Recent outcomes of the system's own verified checks *relative to expectation* (mean signed prediction error), plus information gained minus contradictions added | 20 min |
| **activation** | 0 .. 1 | 0 | How much is changing: rises with the strongest of surprise, novelty and conflict, saturating (it can approach 1 but never exceed it) | 3 min |
| **surprise** (prediction error) | 0 .. 1 | 0 | Moving average of unsigned prediction errors | 5 min |
| **novelty** | 0 .. 1 | 0 | Moving average of how new recent input was (count-based: about 1/√(1 + times seen)) | 10 min |
| **uncertainty** | 0 .. 1 | 1.0 (no experience yet) | Recency-weighted mean entropy of the expectations used recently; falls as predictions become reliable | none: computed from the expectations |
| **information need** | 0 .. 1 | 0 | Pressure of open gaps: 1 − exp(−Σ weights) over unknown questions not yet resolved (each weight fades with a 1-day half-life), conflicting facts and unsure facts | none: computed from evidence |
| **learning gain** | 0.05 .. 0.6 | 0.3 | Pearce-Hall associability per expectation: rises after surprising outcomes, falls when outcomes are predicted | per update (γ = 0.3) |
| **baselines** (valence, activation) | ±0.25 / 0 .. 0.2 | 0 | Slowly move toward the recent state (η = 0.02 per appraised episode); **experimental, switchable** | none (they are the slow part) |

**Not included, deliberately:**
- **Social connection/disconnection, threat/safety, energy/load:** these would need built-in needs, a
  self-preservation value, or a body. Each is a predetermined motive.
- **Goal progress:** there are no goals (blank slate). Its place is taken by "outcome vs expectation of the
  system's own operations".
- **Named emotions:** none.

### 4.2 Expectations (what "expected" means)

Small learned predictors of binary outcomes, kept per context. For each context key k:
- p_k: the predicted probability that the outcome happens (start 0.5: no information);
- n_k: how many outcomes were seen;
- α_k: the learning gain (start 0.3).

Update (Rescorla-Wagner with Pearce-Hall gain):

```
δ   = outcome − p_k                          prediction error (signed); |δ| is the surprise
p_k ← p_k + α_k · δ
α_k ← clip((1 − γ) · α_k + γ · |δ|, 0.05, 0.6)
```

| Expectation | Context key | Outcome | Kind |
|---|---|---|---|
| a question is interpreted | `interpret:question` | the rules covered it (reasoning or recorded-state) | **operation** (signed, goes into valence) |
| the reply says the reasoning result | `express:<status>` | the expression check said it | operation |
| the reply is complete | `reply:complete` | ended by itself, first attempt | operation |
| a question is answerable | `answerable:<about you / your role / named thing / world>` | status known (vs unknown) | **information** (unsigned only: surprise; an unknown opens a gap) |
| stored knowledge holds | `consistent:<category>` | your statement confirmed what was stored (vs corrected or contradicted) | information |

**Operation** outcomes are the system's own components working as built: interpretation, expression,
generation. They are checked by measurements that already exist. Only these signed errors enter valence.
**Information** outcomes say how the world turned out, with no "better" or "worse" side. They change
surprise, uncertainty and information need, never valence directly.

### 4.3 Memory tags (in the internal-state database, keyed by memory id)

When a memory is formed during an episode whose **salience** σ (the largest of that episode's surprise,
novelty, conflict and |information change|) is at least **θ = 0.4**, a tag is stored:
- memory id;
- σ;
- valence and activation at encoding;
- the event id;
- the time.

Below the threshold, no tag is stored, so most memories are not "emotional". A tag's strength fades with a
7-day half-life. It is renewed only by **new evidence** (you state the same thing again), never by being
recalled. That prevents loops.

### 4.4 Configuration

All numbers live in one place, `internal_state/config.py`:
- half-lives;
- gains (k_v = 0.5, information weight w_i = 0.6, k_a = 0.6, λ_s = λ_n = 0.5);
- bounds;
- θ;
- β (encoding boost 0.2);
- recall weights;
- refractory times;
- switches for M2, M3 and baseline adaptation.

Tests override them. They are engineering settings, measured, and changed only with a recorded reason.

---

## 5. The event → appraisal pipeline

### 5.1 Eligible inputs (and what is never an input)

| Event (from the event log) | Appraised as |
|---|---|
| your message: a question, and its reasoning result | interpretation (operation); answerability (information); novelty of the question form and of what it is about; unknown → a gap opens; conflicting → conflict |
| the reply's **verified checks** (said / not said, complete / not) | expression and completeness (operations) |
| knowledge changed: learned / confirmed / corrected / changed (flagged) | novelty (new things and relations); consistency (information); a flagged contradiction → conflict and information loss; a statement that **resolves an open gap** → information gain |
| recall (tagged memories put in the prompt) | reinstatement (M3) |
| memory formed | no state change by itself; it receives the encoding effect (M1) |
| **Not appraised** (recorded only) | the words of any reply; measurements; run start/stop; your panel controls; "forget" requests |

**Why controls and forgetting are not appraised:** treating deletion as bad would be a built-in
self-preservation value.

**Generated text can never create state.** The event log holds no reply words at all. Only the checks the
system itself measured about a reply are used.

### 5.2 One episode (a message and what it caused)

1. **Decay** every decaying dimension to now: `x ← b + (x − b) · 2^(−Δt / half-life)`.
2. **Appraise:** collect the checks (δ and |δ| for each expectation used), novelty n, information change g
   (+1 per gap resolved, −0.5 per contradiction added, scaled to −1 .. 1), and conflict c. Each item carries
   its event id.
3. **Update** (bounded):

   ```
   signal     = clip(mean signed δ of operations + w_i · g, −1, 1)          (w_i = 0.6)
   valence    ← valence + k_v · (signal − valence)            (a moving average; built version, see Status)
   surprise   ← surprise + λ_s · (max |δ| − surprise)
   novelty    ← novelty + λ_n · (n − novelty)                      (only if the episode had new input)
   activation ← activation + k_a · max(|δ|, n, c) · (1 − activation)   (saturating: a nonlinear effect)
   then the expectations learn (4.2); uncertainty and information need are recomputed
   ```

4. **Baselines** (if on): `b ← clip(b + η · (x − b))`.
5. **Effects:** M1 (encoding tag and importance) for memories formed in this episode.
6. **Record** one history row: before, after, the appraisal items, and every event id used.

### 5.3 Context dependence (built in, not scripted)

The same event gives different changes because:
- **expectations differ:** an expression failure after many successes means p ≈ 0.95, δ ≈ −0.95 and a large
  valence drop. After many failures, p ≈ 0.05, δ ≈ −0.05 and almost no change;
- **novelty depends on history:** a first mention is novel, the tenth is not;
- **an "unknown" answer** is surprising where answers are usually known, and unsurprising where they
  usually aren't;
- **a statement** can resolve an open gap (information gain) or not, depending on what was asked before.

### 5.4 The one evaluative assumption, stated openly

Some source of "better / worse than expected" is unavoidable in any valence-like variable. This design uses
only two, both content-free:
1. **The system's own components working as built** (verified checks).
2. **Information gained** (gaps resolved) or lost (contradictions).

It never uses any liking or disliking of content, people, topics or words. If you prefer, valence can be
restricted to (1) only. That is a one-line configuration change.

---

## 6. Connections

### 6.1 Memory
- **M1, significance → importance (encoding):** a tagged memory's importance rises by β · σ (at most +0.2,
  capped at 1.0). The change is logged in memory's own log with its cause ("salience 0.62 at encoding,
  event 41").
  - Importance only **orders** memories that are already relevant (memory/retrieval.py), so it can never
    bring an unrelated memory into the prompt.
- **M2, state → recall order** (small; switchable): a tagged memory's score is multiplied by
  (1 + b), with b = 0.1 · strength · (1 + 0.5 · congruence) and b ≤ 0.1.
  - Congruence = 1 − |valence now − valence at encoding| / 2.
  - It applies **only among memories that already passed the relevance thresholds**.
- **M3, recall → state:** when a tagged memory is recalled into the prompt, valence moves 20% of its
  strength toward its encoding valence, and activation rises a little.
  - **Refractory:** the same memory reinstates at most once per 30 minutes.
  - Reinstatement creates no new tags and does not renew the tag.
- **M4, repetition → baselines:** see 4.1 (experimental).
- **M5, decay:** tag strength fades unless you state the thing again.
- **Rumination guard:** bounded effects, the refractory period, renewal only by new evidence, and decay that
  always wins, so negative recall cannot feed itself. A test proves it.

### 6.2 Learning
- **L1, surprise → learning attention:** the learning gain α (Pearce-Hall) sets how fast the expectations
  move. M1 makes surprising episodes' memories more important.
- **L2, learning events → state:** new facts (novelty), confirmations (predictions met: uncertainty falls),
  corrections (surprise), flagged contradictions (conflict, information loss), and resolved gaps
  (information gain).
- **L3, repeated failure vs success:** the expectations learn it, so the same later event gives a different
  change (5.3).
- **L4, uncertainty → information seeking:** today the system has no action of its own to seek information
  (it cannot ask; the brain writes every reply). Information need is computed and reported, and the
  self-model lists the most pressing open questions first. Real information *seeking* belongs to curiosity
  (Stage 6).
- **The learning store's own rules are unchanged.** Fact confidence, corrections and conflicts are not
  altered by the state. The state does not secretly replace learning.

### 6.3 Reasoning and the world model
- **Reasoning outcomes:**
  - interpreted or not;
  - known / unknown / conflicting / uncertain;
  - expressed or not;
  - a gap resolved later.

  They become appraisal items through the expectations above.
- **World events:**
  - new entities and relations (novelty);
  - corrections (consistency);
  - contradictions (conflict).

  These come only from **your statements as learning stored them**, or from the world model's own results.
  Never from generated text.

### 6.4 Self-model
- A new section, **internal state**, contains:
  - the current values with their definitions;
  - recent changes with their causes and evidence;
  - the history;
  - the baselines and how they moved;
  - the expectations and their uncertainty.
- A new recorded-state question: "What is your internal state?" / "What changed your internal state?" gives
  **numbers with causes**.
- "How do you feel?" stays **outside** the self-model (it asks for a feeling claim).
- Still never shown to the brain.

### 6.5 Chat
- **A new Internal state panel** shows:
  - current values: a bar, the number and the one-line definition for each dimension;
  - recent changes: cause, change and evidence (event ids);
  - baselines;
  - expectations: context, p, count and learning gain;
  - memory tags;
  - a small history chart.
- **No first-person text** anywhere.
- **Details** under a reply shows that message's appraisal and state change.
- **The reply is never written, chosen or edited by the state.**

### 6.6 Persistence (current vs history)

`state/internal_state.db` holds:
- **current:** one row per dimension (value, baseline, time);
- **expectations;**
- **memory tags;**
- **history:** one row per appraised episode, holding before, after, the appraisal items and the event ids.

The event log stays the evidence. History rows *refer* to events and never copy them.

On restart, the saved state is loaded and **decayed for the time the program was off**. So after a night,
the fast dimensions are back at baseline, while expectations and baselines (the slow, learned parts)
remain. With `--no-memory` it lives in RAM only, like the event log.

---

## 7. Testing strategy

### 7.1 Unit tests (`tests/test_internal_state.py`, synthetic events, an injected clock, no brain)

| # | Test |
|---|---|
| 1 | **Neutral at startup:** valence, activation, surprise, novelty and information need are 0; uncertainty is the prior; no history; no tags |
| 2 | One event changes the state, by exactly the equations in 5.2 |
| 3 | Repeating the event: novelty and surprise shrink as predicted (closed form); uncertainty falls |
| 4 | Decay: each dimension's measured half-life is within 1% of its setting; after 7 half-lives it is within 0.01 of baseline; different dimensions decay at different rates |
| 5 | Different events (success, expression failure, unknown, contradiction, new fact, gap resolved) give different appraisal vectors |
| 6 | Context: the same expression failure after 10 successes vs after 10 failures gives \|Δvalence\| differing ≥ 3×; the same unknown answer in "usually known" vs "usually unknown" contexts gives surprise differing ≥ 0.3 |
| 7 | Memories affect the state: recalling a tagged memory moves valence toward its encoding sign; the refractory period holds |
| 8 | The state affects memory importance: high salience means a tag and higher importance; low salience means neither |
| 9 | Learning events affect the state (new fact, confirmation, correction, contradiction, gap resolved), each with its predicted sign |
| 10 | Reasoning outcomes: unexpected success raises valence, unexpected failure lowers it, unknown raises information need, conflicting raises activation |
| 11 | Restart: the state equals the saved state decayed by the downtime (error < 1e-6); expectations, baselines and tags are kept |
| 12 | Every history row's evidence resolves to real events (100%) |
| 13 | Generated text cannot create state: emotional words in a reply ("I am terrified, I love this") give a state identical to a neutral reply with the same checks; non-eligible events change nothing |
| 14 | No predefined personality or identity: the **blank-slate scan** (below) |
| 15 | The self-model reports exactly the stored state (values, causes, evidence) |
| 16 | **Not a response script:** with M2 off, the brain's prompt and reply are byte-identical with the state on and off; with M2 on, only the order of recalled memories can differ; no code maps state to text |
| + | **Rumination guard:** 50 repeated recalls of a strongly negative memory keep valence within bounds, and decay still returns it to baseline |
| + | Bounds: no value ever leaves its range under 10,000 random eligible events |

**Blank-slate scan** (automated, in the originality guard), over the code and the stored state:
- No trait-like definitions: emotions as traits ("is happy", "temperament", "mood_default"), personality
  parameters (Big Five names, "openness", "extraversion"...), likes and dislikes, opinions, values, goals,
  identity, names, favorites, consciousness claims.
- No first-person state phrases ("I feel", "I am happy/sad/scared") in any string the package can output.
- Stored state contains only the allowed keys (the dimensions, expectations, tags, history) and numbers with
  evidence.

### 7.2 Benchmark (`python -m internal_state.evaluate`)

It runs controlled experiments, mostly without the brain, plus one integrated run through the real chat
engine. **Criteria fixed now, before building:**

| Measure | Criterion |
|---|---|
| State-transition consistency | Identical event histories give identical states (100%) |
| Decay behavior | Half-lives within 1%; return to baseline within 0.01 after 7 half-lives |
| Context sensitivity | The two context probes of test 6 meet their thresholds |
| Memory interaction | High-salience episodes tagged ≥ 90%, low-salience ≤ 10%; among equally relevant memories the tagged one is recalled first 100%; reinstatement moves valence in the encoding direction 100%; the rumination guard holds |
| Learning interaction | Learning gain after a surprise > in a stable run (100% of probes); uncertainty strictly falls over 10 consistent outcomes |
| Reasoning interaction | Signs of Δvalence as predicted for unexpected success and failure (100%) |
| Persistence | Restart error < 1e-6 |
| Baseline adaptation (experimental) | 200 successes vs 200 unexpected failures give baselines ≥ 0.1 apart and different responses to the same probe event; with adaptation off, baselines stay 0 |
| False-state creation | Generated words: 0 change; non-eligible events: 0 change; unexplained changes: 0 (every change has evidence) |
| Blank-slate compliance | 0 findings |
| No regression | The memory benchmark (`python -m memory.evaluate`) with M2 on stays within 1 point of state-off; the world benchmark is unaffected (reasoning items do not use M2) |

The full test suite must still pass. **No neural training is needed:** the brain is not involved.

---

## 8. What remains missing after this stage

- **No action selection by the state.** It changes storage, recall order and learning speed, not what the
  system does. Information seeking and exploration need curiosity and goals (Stages 6 and 10).
- **The appraisal checks are hand-designed** (content-free) functions with hand-set weights. Learning these
  couplings and proving their value by ablation in a changing world is Stages 11 and 14.
- **Expectations are small context tables:** no generalization, and no learned world prediction (Stage 5).
- **Missing dimensions:** no goals (so no goal progress), no body (so no energy or homeostatic need), no
  social or threat dimensions. These are deliberate: each would be a predetermined motive.
- **The brain does not see the state.** Nothing turns it into language. A later stage could, only with a
  measured reason and your approval.
- **Whether recurring, emotion-like patterns emerge** is unanswered. It is an analysis of the logged history
  for later, not a claim.
- **Nothing here shows that anything is felt.** The claims ladder in [03](03_CONSCIOUSNESS.md) is unchanged.
