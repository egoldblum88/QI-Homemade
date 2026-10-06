# 01 — Research: how to build a thinking-and-talking system from nothing

This document is the research phase. It explains what is already known, compares approaches,
and says what is realistic **on this computer**. The architecture in
[02_ARCHITECTURE.md](02_ARCHITECTURE.md) follows from the conclusions here.

Each topic ends with a **"What we take"** box, which separates the *scientific idea* we learn from
and the *implementation* we write ourselves.

---

## Part A — The neural language core

### A1. What a language model actually is

A language model is a function that, given some text, outputs a **probability for every
possible next piece of text** (the next *token*).

```
"The cat sat on the"  →  model  →  { "mat": 0.21, "floor": 0.12, "bed": 0.07, ..., "banana": 0.0001 }
```

To *generate*, you pick a token from that distribution, append it, and repeat.

To *train*, you show it real text and adjust it so it gives **higher probability to the token that
really came next**. The measure of "how wrong" it was is the **cross-entropy loss**:

```
loss = −log( probability the model gave to the correct next token )
```

- If the model gave the correct token probability 1.0 → loss = 0 (perfect).
- If it gave 0.01 → loss = 4.6 (bad).
- A model that knows nothing and guesses uniformly over V tokens has loss = log(V).
  For ~96 characters that is log(96) ≈ **4.56**. Any learning must beat this number.
- **Perplexity** = e^loss, which you can read as "how many choices the model is effectively confused
  between". Uniform guessing over 96 characters gives perplexity 96. A good small character model
  gets to about 4–5.

The important point is that the model is never given sentences to repeat. It is only given
a number (the loss) that says how surprised it was, and it adjusts itself to be less surprised.
Everything it "knows" about spelling, grammar and style comes out of that process.

### A2. The building blocks

| Block | Plain-language meaning | The math (you will implement it) |
|---|---|---|
| **Tokenizer** | Turns text into a list of integers and back. | a lookup table / merge rules |
| **Embedding** | Each token ID gets a learned vector of numbers, its "meaning coordinates". Tokens used in similar ways drift close together. | row lookup in a matrix `E` of shape `[vocab, d]` |
| **Linear layer** | Mixes numbers: every output is a weighted sum of every input. The weights are what gets learned. | `y = x·W + b` |
| **Nonlinearity** | Bends the math so that stacked layers can represent more than a straight line. | `tanh`, `ReLU`, `GELU` |
| **Softmax** | Turns raw scores ("logits") into probabilities that sum to 1. | `p_i = e^{z_i} / Σ_j e^{z_j}` |
| **Loss** | One number for "how wrong". | cross-entropy `−log p_correct` |
| **Gradient** | For each weight, which direction to nudge it to reduce the loss, and how strongly. | `∂loss/∂w` |
| **Backpropagation** | An efficient way to get *all* gradients at once, using the chain rule from the output back to the input. | chain rule, layer by layer |
| **Optimizer** | The rule that uses gradients to update weights. | SGD: `w ← w − lr·g`; Adam: adaptive per-weight steps |

**How backpropagation works in our code.** Every layer will have two methods:

- `forward(x)` computes the output and remembers what it needs.
- `backward(grad_output)` receives "how much the loss changes if my output changes", computes
  (a) the gradients for its own weights and (b) `grad_input`, which it passes to the previous layer.

We will derive each `backward` by hand, then **verify it numerically** with *gradient checking*:
nudge one weight by ±ε, measure the change in loss, and compare with what `backward` claims.
If they match to ~6 digits, the math is right. This is how we can trust code we wrote ourselves.

**How the network changes when it learns.** Before training, every weight is a small random number
and the output probabilities are roughly uniform. Each training step: (1) run a batch of real text
forward, (2) measure loss, (3) backpropagate, (4) nudge every weight a tiny bit downhill. After
thousands of steps, some embedding directions come to encode things like "vowel vs consonant" or
"start of a word". Attention heads specialize (for example, "look at the previous quote mark").
The MLP layers store common patterns. We will be able to *inspect* this: embedding similarity lists,
attention maps, and loss curves.

### A3. Tokenization options

| Tokenizer | Vocabulary | Pros | Cons | Verdict |
|---|---|---|---|---|
| Character-level | ~70–100 | Trivial to build; no unknown words; perfect for learning | Long sequences; model must learn spelling; weaker per-compute | **Stage 1–4** |
| Word-level | 20k–100k+ | Short sequences | Huge vocab and embedding matrix; unknown words; bad with typos | No |
| **Byte-level BPE** (Byte-Pair Encoding) | 1k–8k (our choice) | Balance between the two; nothing is ever "unknown" (falls back to raw bytes) | Must implement the merge-learning algorithm | **Stage 5+** |
| Unigram LM tokenizer (SentencePiece-style) | similar | Slightly better compression | More complex (EM algorithm) | Maybe later, as an experiment |

BPE is a compression algorithm (Gage, 1994) adapted for language models (Sennrich et al., 2016).
Start with bytes, repeatedly find the most frequent adjacent pair, and merge it into a new token.
We will write it from scratch in ~150 lines of Python.

### A4. Comparing neural language-model architectures

| Approach | How it works | Memory of context | Quality at our tiny scale | Training speed in NumPy | Difficulty | Verdict |
|---|---|---|---|---|---|---|
| **Count n-gram** (not neural) | Count how often each token follows the previous n−1 tokens | n−1 tokens | Poor but a real baseline | Instant | Very easy | **Stage 1 baseline** |
| **MLP n-gram** (Bengio et al., 2003) | Embed last k tokens, concatenate, MLP → softmax | fixed k tokens | Beats counts; generalizes to unseen combinations | Fast | Easy | **Stage 2**, first real neural brain |
| **Vanilla RNN** | Hidden state carried step by step: `h_t = tanh(x_t·W + h_{t−1}·U)` | in principle unlimited; in practice short (vanishing gradients) | Moderate | **Slow**: must loop over time steps in Python | Medium | Optional experiment |
| **LSTM / GRU** | RNN with gates that protect memory | longer | Good; competitive with transformers at *very* small scale | Slow (same sequential problem) | Medium-hard backward pass | Optional experiment |
| **Transformer (decoder-only)** | Self-attention: every position directly looks at all earlier positions; stacked blocks | full context window (e.g. 128–512 tokens) | Best known quality per compute; scales smoothly | **Fast**: whole sequence in parallel as big matrix multiplies, which is where NumPy is fastest | Medium-hard; attention backward is the tricky part | **Main core, Stage 4+** |
| State-space / linear-recurrent (S4, Mamba, RWKV) | Recurrent in form, trainable in parallel | long | Strong at large scale | Needs special scan kernels to be efficient | Hard | Not now; research track |

**Why a transformer, and why it is not "copying another AI".** The transformer is a published
mathematical technique (Vaswani et al., 2017), in the same way a convolution or a Fourier transform is.
Using it no more copies ChatGPT than using wheels copies someone's car. What we do not take is
anyone's *weights*, *code* or *training data*. There is also a practical reason for our setup
specifically: NumPy on your M4 multiplies big matrices at 1–2 TFLOP/s, but a Python `for` loop over
time steps (needed by RNNs) is slow. Transformers turn almost all work into big matrix multiplies.

**Where our originality can go in the neural core.** Once the standard core works, we can run
experiments that are specific to this project. The main one is a *state-conditioned language core*:
the agent's emotion and goal vector is fed *into every layer* of the network, as a learned scale-and-shift
(an idea related to FiLM conditioning, Perez et al. 2018). Emotion then shapes language at the neural
level, not only through words placed in the prompt. See [02_ARCHITECTURE.md §B6](02_ARCHITECTURE.md).

### A5. Attention in plain language

For each position in the text, the network computes three vectors from that position's current
representation:

- **Query**: "what am I looking for?"
- **Key**: "what do I contain?" (advertised to later positions)
- **Value**: "what I will hand over if you pick me"

```
scores  = (Q · Kᵀ) / √d_head          # how well each query matches each key
scores  = mask_future(scores)           # causal mask: a position may not look at the future
weights = softmax(scores)               # attention weights, each row sums to 1
output  = weights · V                   # weighted average of the values
```

*Multi-head* attention runs several of these in parallel (e.g. 4 heads), each free to learn a
different kind of lookup. A **transformer block** is:

```
x = x + Attention(LayerNorm(x))     # gather information from other positions
x = x + MLP(LayerNorm(x))           # process it position by position
```

The `x + ...` (residual connection, He et al. 2015) lets information and gradients flow straight
through, which makes deep networks trainable. **LayerNorm** (Ba et al., 2016) keeps the numbers in a
healthy range. **Positional embeddings** tell the network *where* each token is, because attention by
itself ignores order.

### A6. What is realistic on this Mac

**Measured on this machine (2026-09-28):** NumPy float32 matrix multiplies of the shapes our model
uses ran at **0.83–2.07 TFLOP/s**, and batched attention-shaped multiplies at ~1.0 TFLOP/s.
Whole-model training also includes element-wise work (softmax, LayerNorm, GELU) and Python overhead,
so for planning we assume **~10–15% of peak ≈ 150 GFLOP/s effective**. We will measure the real
figure in Stage 4.

Training cost ≈ **6 × (number of parameters) FLOPs per token**, plus attention.

| Config | Tokenizer | Layers × width | Context | Parameters | FLOPs/token (train) | Est. speed | Est. time per 10M tokens |
|---|---|---|---|---|---|---|---|
| **A: char-tiny** | char (~96) | 4 × 128, 4 heads | 128 | **~0.8M** | ~6M | ~25k tok/s | **~7 min** |
| **B: bpe-small** | BPE 4096 | 6 × 256, 8 heads | 256 | **~5.8M** | ~40M | ~3.7k tok/s | **~45 min** |
| C: bpe-medium | BPE 8192 | 8 × 384, 6 heads | 512 | ~17.5M | ~125M | ~1.2k tok/s | ~2.3 h (CPU); maybe 3–5× faster on the Apple GPU (research track) |

Memory (RAM) is not a problem: config B needs roughly 1 GB during training.

**Our real bottleneck is data, not compute.** A rule of thumb from scaling research (Hoffmann et al.,
2022) is about 20 training tokens per parameter. Config B would like ~120M tokens. A curated set of
public-domain children's books gives roughly 3–20M tokens, so we will train for several passes, use
regularization, and grow the dataset over time. Public-domain text is abundant (Project Gutenberg has
~75,000 books). What limits us is compute per night, not legal data.

**Honest expectations for the language it will produce**

| Model | What it will probably do |
|---|---|
| Count bigram | Letter soup with realistic letter frequencies: `"thend wher heas an"` |
| MLP n-gram (char) | Mostly real short words, broken grammar |
| A: char-tiny transformer | Real words, local grammar, the *style* of old storybooks, little meaning beyond a clause or two |
| B: bpe-small transformer | Mostly grammatical sentences and story-like fragments; names and topics hold for a few sentences; drifts after that; **no general world knowledge; cannot answer questions or follow instructions** until trained on dialogue, and even then only short, simple replies |

For scale: GPT-2 small (2019) has 124M parameters and was trained on ~40 GB of text. Config B is
~20× smaller and sees ~500× less text. Modern commercial models are thousands of times beyond that.
The TinyStories study (Eldan & Li, 2023) showed that models of 1–30M parameters *can* write fluent,
coherent simple stories when the training text uses a small, simple vocabulary. That is encouraging
evidence about what small models can do. However, TinyStories itself was **written by GPT-3.5/GPT-4**,
so we cannot use it (see A8).

The practical consequence for the design is that the small neural network cannot be the *whole* mind.
It will be the agent's **language faculty**. It turns what the rest of the mind decides into words,
and it provides signals like "how surprising is this input". Memory, emotion, goals and reasoning
live in separate systems we write around it. This is Part B of the research, and it is where most of
the project's originality lives.

### A7. Training techniques we will implement

| Technique | Why it exists |
|---|---|
| **Train / validation split by document** | Validation text must be *unseen*, or we can't tell learning from memorizing. Splitting by whole books prevents leaks. |
| **Mini-batching** | Process e.g. 32 random text windows at once, which is faster and gives a less noisy gradient than one at a time. |
| **Careful initialization** | Random weights scaled by ~1/√(inputs) so signals neither explode nor vanish at the start. |
| **Adam / AdamW optimizer** (Kingma & Ba 2014; Loshchilov & Hutter 2017) | Keeps a running average of each weight's gradient and its size, and gives every weight its own step size. Much faster than plain SGD for transformers. "W" = weight decay, a gentle pull toward zero that reduces overfitting. |
| **Learning-rate warmup + cosine decay** | Small steps at the start (random weights are fragile), larger in the middle, tiny at the end to settle. |
| **Gradient clipping** | If a gradient is suddenly huge, scale it down, which prevents one bad batch from wrecking the model. |
| **Dropout** | Randomly zero some activations during training so the model can't rely on any single path. Important for us because our data is small. |
| **Checkpoints** | Save weights + optimizer state + config + tokenizer so training can stop and resume, and so we can roll back. |
| **Evaluation** | Validation loss, perplexity, bits-per-character, comparison against baselines, samples, and a **novelty test** (A9). |

### A8. Datasets and licenses

**Policy:** human-written, legally clear, and **no AI-generated text** (Charter rule 5).
Books published long ago have one more advantage: they cannot contain AI-generated text.

| Dataset | Human-written? | License | Role |
|---|---|---|---|
| **Curated Project Gutenberg books**: fairy tales, fables, classic children's novels (Grimm, Andersen, Aesop, Carroll, Baum, Burnett, Kipling, Grahame, Montgomery, Sewell, Potter, …) | Yes | Public domain in the US. Gutenberg's terms: strip the Project Gutenberg header/footer and trademark, and the remaining text is unrestricted. | **Main pretraining corpus** |
| **Your own original writing** | Yes (please write it yourself; do not paste AI output) | Yours | Pretraining, dialogue style, fictional backstory |
| Public-domain plays and dialogue-heavy books (Gutenberg) | Yes | Public domain | Dialogue fine-tuning |
| DailyDialog (Li et al., 2017) | Yes | CC BY-NC-SA 4.0: OK for a personal, non-commercial project; share-alike if redistributed | Optional dialogue fine-tuning |
| Simple English Wikipedia | Mostly (recent edits may include AI-assisted text) | CC BY-SA 4.0 (attribution) | Optional, later, for factual knowledge |
| ~~TinyStories~~ | **No: generated by GPT-3.5 and GPT-4** | CDLA-Sharing-1.0 | **Excluded.** Training on it would put another AI's knowledge into our weights. |
| Google Speech Commands | Yes (recordings) | CC BY 4.0 | Voice: keyword recognition (Phase 13) |
| LJ Speech | Yes (human reader) | Public domain | Voice: speech-synthesis research |
| LibriSpeech | Yes (LibriVox readings) | CC BY 4.0 | Voice: stretch goal, large-vocabulary recognition |

Practical notes:
- **Gutenberg forbids automated scraping of its main website.** For ~30 books, download by hand.
  For more, use an official mirror or their robot-harvest endpoint, as their policy describes.
- For readers outside the US: choosing authors who **died in 1955 or earlier** makes the originals
  public domain in "life + 70 years" countries as of 2026. For translations (e.g. Grimm, Andersen),
  the *translator's* dates matter too. Stage 0 will include a dataset card listing every file,
  its source and its license.
- Every dataset gets a **dataset card** (`data/DATASET_CARD.md`) recording where each file came from.

### A9. Proving the model really generates, and doesn't recite

You asked us to test whether it produces language it was *not* given as a complete sentence.
We will measure it directly:

1. **n-gram novelty.** For each generated text, what fraction of its 5-word sequences never appear
   anywhere in the training data? Memorized text scores near 0%. Real generation scores high.
2. **Longest copied span.** The longest substring shared with the training data. Short = composing.
   Long = reciting.
3. **Minimal pairs.** We write sentence pairs that differ in one grammatical detail
   ("the cats **are** hungry" vs "the cats **is** hungry"). A model that learned grammar gives the
   correct one higher probability. Neither sentence needs to appear in the training data.
4. **Baselines.** The trained model must beat the uniform, unigram and bigram baselines on *unseen*
   validation text.

### A10. Generation controls (implemented in Phase 3)

| Control | What it does | Why it exists |
|---|---|---|
| **Temperature** | Divide logits by T before softmax. T<1 sharpens, T>1 flattens. | Low T is safe but repetitive; high T is creative but chaotic. It is our main creativity dial, and later linked to arousal. |
| **Top-k** (Fan et al., 2018) | Only sample among the k most likely tokens. | Cuts off the long tail of very unlikely junk tokens. |
| **Top-p / nucleus** (Holtzman et al., 2019) | Sample among the smallest set of tokens whose probabilities sum to p. | Like top-k, but adapts: few options when the model is sure, more when unsure. |
| **Repetition penalty** (Keskar et al., 2019) | Lower the score of tokens already generated. | Small models love loops ("the the the"). |
| **No-repeat n-gram** | Forbid repeating any exact n-token sequence. | A harder guard against loops. |
| **Max length / stop tokens** | Stop after N tokens or at an end marker. | Prevents endless output. |

---

## Part B — Research for the mind around the brain

### B1. Memory architectures

| Idea | Source | Key point |
|---|---|---|
| Multi-store memory | Atkinson & Shiffrin (1968); Baddeley's working memory (1974, 2000) | Separate short-term (small, fast-fading) and long-term stores |
| Episodic vs semantic memory | Tulving (1972) | "What happened to me" is stored separately from "what is true" |
| Autobiographical memory | Conway & Pleydell-Pearce's Self-Memory System (2000) | Life periods → general events → specific episodes, organized around the self and its goals |
| Activation-based retrieval and forgetting | ACT-R (Anderson), Ebbinghaus forgetting curve | A memory's accessibility grows each time it is used and decays with time |
| Mood-congruent memory | Bower (1981) | Current mood makes similar-mood memories easier to recall |
| Emotional enhancement | McGaugh and others | Emotionally arousing events are encoded more strongly |
| External memory for neural nets | Memory Networks (Weston et al., 2014); Neural Turing Machine (Graves et al., 2014); retrieval-augmented generation (Lewis et al., 2020); kNN-LM (Khandelwal et al., 2020) | A network can consult a separate store instead of keeping everything in its weights |
| Memory stream + reflection for agents | Generative Agents (Park et al., 2023) | Retrieve by recency × importance × relevance; periodically "reflect" into higher-level memories. (They used ChatGPT for scoring and reflection. We compute these with our own code.) |

> **What we take:** separate stores (short-term, episodic, semantic, autobiographical) kept in SQLite,
> outside the neural net. Retrieval scored by relevance + recency/activation + importance + emotional
> congruence + links. Encoding strength boosted by arousal and goal relevance. Decay, plus consolidation
> that compresses weak memories into summaries. **We write every formula and every line of storage code.**

### B2. Continual learning: should the network keep learning after training?

The central danger is **catastrophic forgetting** (McCloskey & Cohen, 1989). Train a network on new
data alone and it overwrites old skills. Known remedies:

- **Rehearsal / replay**: mix old data with new data when fine-tuning (Robins, 1995).
- **Regularization**: penalize changing weights that were important for old tasks
  (Elastic Weight Consolidation, Kirkpatrick et al., 2017).
- **Parameter isolation**: put new learning into small extra weights, such as low-rank adapters
  (LoRA, Hu et al., 2021), which can be removed if they cause harm.
- **Complementary Learning Systems** (McClelland, McNaughton & O'Reilly, 1995): the brain seems to use a
  *fast* learner (hippocampus: stores episodes immediately) and a *slow* learner (cortex: gradually
  absorbs patterns during offline replay, e.g. sleep).

> **What we take:** a CLS-style design. The **memory system is the fast learner**, with no weight
> changes during conversation. The **neural network is the slow learner**: only during a deliberate,
> offline "sleep consolidation" run, on curated memories mixed with replayed original data, checked
> against validation, and **automatically rolled back** if the old skills get worse.

### B3. Reinforcement learning (RL)

RL is learning from rewards: an agent acts, gets a reward signal, and adjusts its behavior policy.
Methods range from simple **multi-armed bandits** to Q-learning and policy gradients, up to RLHF
(the method used to make chatbots helpful, which needs large reward models and a lot of compute).

> **What we take:** small, safe RL where it fits. A **contextual bandit** learns which *kinds* of actions
> work well in which situations (ask a question vs share a memory vs express state), from your
> feedback. Curiosity supplies an **intrinsic reward** (B7). We do **not** do RL on the neural language
> core early on; it is unstable and data-hungry.

### B4. Cognitive architectures

Several decades of research have built whole-mind architectures: **Soar** (Laird, Newell, Rosenbloom),
**ACT-R** (Anderson), **LIDA** (Franklin, based on Global Workspace Theory), **CLARION** (Sun) and
**Sigma** (Rosenbloom). They largely agree on a "Common Model of Cognition" (Laird, Lebiere &
Rosenbloom, 2017): a limited **working memory** at the center, **declarative** long-term memory
(facts, episodes), **procedural** memory (skills and habits), perception and motor modules, and a
repeating **cognitive cycle** (roughly 50–100 ms in humans).

**Global Workspace Theory** (Baars, 1988; Dehaene's neuronal version) proposes that many specialized
processes run in parallel and compete for a small shared "stage". Whatever wins is **broadcast** to all
processes. It is a clean engineering pattern for connecting many modules.

> **What we take:** a cognitive cycle, a small capacity-limited **global workspace** with competition
> and broadcast, and separate memory systems. We do not use or copy Soar/ACT-R/LIDA code. We implement
> the concepts in our own simple form.

### B5. Computational emotion models

| Family | Idea | Examples |
|---|---|---|
| **Appraisal theories** | Emotion comes from *evaluating* an event against your goals: Is it new? Good or bad for me? Who caused it? Can I cope? | OCC model (Ortony, Clore & Collins, 1988); Scherer's Component Process Model; Lazarus |
| **Dimensional** | Every affective state is a point in a small space | Russell's circumplex (valence × arousal, 1980); PAD (pleasure-arousal-dominance) |
| **Basic emotions** | A handful of distinct, recognizable emotion programs | Ekman |
| **Constructionist** | Emotions are constructed from core affect + concepts + context | Barrett |
| **Computational systems** | Working implementations | EMA (Marsella & Gratch), WASABI (Becker-Asano), FAtiMA, Kismet (Breazeal), homeostatic agents (Cañamero) |
| **Functional role** | Emotions prioritize, bias attention, memory and decisions | Damasio's somatic-marker hypothesis; Simon (1967) "motivational and emotional controls of cognition" |

> **What we take:** a hybrid of our own design. **Appraisal** (event → evaluation vector) drives
> **10 named emotion variables**. These feed **core affect** (valence, arousal) and a slow **mood**.
> **Personality** (Big Five traits) sets each variable's baseline, sensitivity and decay rate. The state
> then **modulates** attention, memory encoding, memory retrieval, goals, decisions and language
> generation. The equations are in 02_ARCHITECTURE.md §C5.

### B6. Self-models, agents, world models

- **Self-models in robots:** robots that learn a model of their own body and use it to adapt
  (Bongard, Zykov & Lipson, 2006; Kwiatkowski & Lipson, 2019). The lesson: a self-model is useful when
  it *predicts* and is *checked against reality*.
- **Metacognition and calibration:** knowing what you know. It can be measured: when the agent says
  "80% sure", is it right about 80% of the time?
- **Attention Schema Theory** (Graziano): a system benefits from a simplified model of *its own
  attention*.
- **Source / reality monitoring** (Johnson, Hashtroudi & Lindsay, 1993): humans tag memories by
  origin (perceived vs imagined vs told), and errors in that tagging cause false memories. This is
  the scientific basis for our epistemic tags (Phase 12).
- **Agent architectures:** sense-plan-act; Brooks' subsumption (1986); **BDI** (Beliefs, Desires,
  Intentions; Rao & Georgeff, 1995), which is close to our goal system.
- **World models:** Craik (1943) proposed that minds carry "small-scale models" of reality; Ha &
  Schmidhuber (2018) showed learned world models in neural agents; **predictive processing** (Friston,
  Clark, Seth) sees the brain as a prediction machine driven by prediction error.
- **Symbol grounding problem** (Harnad, 1990): words only mean something to a system if they connect to
  something beyond other words, such as perception and action. This is why Phase 12–14 matter.

> **What we take:** a **structured self-model** assembled from real subsystem data (not a paragraph),
> with calibration tests. An **attention schema**. **Epistemic source tags** on everything. A BDI-like
> goal system. Small per-channel **prediction models** whose errors produce surprise and curiosity.

### B7. Curiosity and intrinsic motivation

- Berlyne (1960): novelty, complexity and uncertainty drive exploration.
- Loewenstein (1994): the **information-gap** theory. Curiosity arises when you notice a gap in what you know.
- Schmidhuber (1991, 2010): reward = **improvement** in the ability to predict or compress.
- Oudeyer & Kaplan (2007): **learning progress**. Seek situations where you are learning fastest.
- Pathak et al. (2017): reward = prediction error of a learned forward model.
- The **"noisy TV" problem**: pure-surprise curiosity gets stuck staring at random noise. Learning
  progress fixes this, because noise never becomes more predictable.

> **What we take:** curiosity computed from **real signals**: our language model's surprise on new
> input, unknown words and entities, low-confidence or contradictory beliefs, and open questions.
> Weighted by **learning progress** to avoid the noisy-TV trap. Rate-limited, so the agent doesn't
> interrogate you.

### B8. Perception from scratch

| Task | Classic (no neural net) | Neural (trained by us) | Borrowed options we will **not** use by default |
|---|---|---|---|
| **Speech → text** | MFCC features (Davis & Mermelstein, 1980) + Dynamic Time Warping template matching (Sakoe & Chiba, 1978): recognizes a few dozen words of *your* voice | Small CNN keyword recognizer (Speech Commands, CC BY 4.0) → stretch: CTC acoustic model (Graves et al., 2006) on LibriSpeech with our language model as the prior | Whisper (OpenAI: **excluded**), Vosk/Kaldi models, Apple dictation |
| **Text → speech** | **Formant synthesizer** (Klatt, 1980), rules only: robotic but 100% ours; emotion controls pitch, rate, volume | Small spectrogram model trained on LJ Speech (public domain) + Griffin-Lim reconstruction (1984) | macOS `say`, pretrained neural TTS |
| **Vision** | Brightness, color, frame differencing, background subtraction, blob tracking | Small CNN (LeCun et al., 1989) trained on photos *you* label ("my desk", "someone present") | Pretrained detectors, OpenCV's bundled cascades/DNN models |

Honest assessment: from-scratch perception on a laptop will be **much weaker** than borrowed
pretrained models. Recognizing a small vocabulary of your voice is very achievable. Open-vocabulary
dictation from scratch is a serious research project with a high error rate. We will design perception
behind a clean interface, so any module can be upgraded later. If you ever decide to allow a borrowed
*perception* module, it would be plugged in, clearly labeled **BORROWED** in the ledger and in every
observation's provenance, and never used as the brain. That is your decision to make later
(see [DECISIONS.md](DECISIONS.md)).

---

## Part C — Conclusions

1. **Language core:** our own decoder-only transformer, reached step by step
   (counts → MLP → attention), written in NumPy with hand-derived and gradient-checked backprop.
   Char-level first, then our own byte-level BPE.
2. **Scale:** ~1M parameters first; ~6M as the main model; ~15–20M only if we add a GPU array backend
   (math only, never its neural-net or autograd features).
3. **Data:** curated public-domain human-written books + your own writing. TinyStories excluded
   (AI-generated). No AI-written text in training or memory.
4. **Division of labor:** the neural net is the **language faculty and a surprise sensor**. It is not
   the whole mind. Memory, emotion, goals, self-model, reflection and most reasoning are **separate
   systems we write**, connected through a global workspace.
5. **Learning:** fast (memory, instant) → medium (preferences, per session) → slow (neural weights,
   offline, gated, reversible).
6. **Perception:** our own classical + small neural modules behind a common interface; weak at first,
   upgradeable, always source-tagged.
7. **Consciousness:** see [03_CONSCIOUSNESS.md](03_CONSCIOUSNESS.md). We can build functional
   analogues of several theory-derived indicators. That does not demonstrate experience, and we will
   not claim it does.

---

## References (science we learn from; no code or weights taken)

- Rumelhart, Hinton & Williams (1986). *Learning representations by back-propagating errors.*
- Bengio, Ducharme, Vincent & Jauvin (2003). *A neural probabilistic language model.*
- Hochreiter & Schmidhuber (1997). *Long short-term memory.* · Cho et al. (2014), GRU.
- Bahdanau, Cho & Bengio (2014). *Neural machine translation by jointly learning to align and translate.*
- Vaswani et al. (2017). *Attention is all you need.* · Radford et al. (2018, 2019), decoder-only LMs.
- He et al. (2015), residual networks · Ba, Kiros & Hinton (2016), LayerNorm · Hendrycks & Gimpel (2016), GELU.
- Kingma & Ba (2014), Adam · Loshchilov & Hutter (2017), AdamW · Hoffmann et al. (2022), compute-optimal scaling.
- Gage (1994), BPE compression · Sennrich, Haddow & Birch (2016), subword BPE.
- Fan, Lewis & Dauphin (2018), top-k · Holtzman et al. (2019), nucleus sampling · Keskar et al. (2019), repetition penalty.
- Eldan & Li (2023). *TinyStories* ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759); [dataset card](https://huggingface.co/datasets/roneneldan/TinyStories)).
- Perez et al. (2018), FiLM conditioning · Khandelwal et al. (2020), kNN-LM · Lewis et al. (2020), RAG.
- Tulving (1972) · Atkinson & Shiffrin (1968) · Baddeley (2000) · Conway & Pleydell-Pearce (2000) · Bower (1981).
- Anderson et al., ACT-R · Laird, Newell & Rosenbloom, Soar · Franklin, LIDA · Laird, Lebiere & Rosenbloom (2017).
- Baars (1988), Global Workspace Theory · Dehaene & Changeux, global neuronal workspace.
- Ortony, Clore & Collins (1988), OCC · Scherer (2001, 2009) · Russell (1980) · Marsella & Gratch (2009), EMA · Damasio (1994).
- McCloskey & Cohen (1989) · Robins (1995) · Kirkpatrick et al. (2017), EWC · McClelland, McNaughton & O'Reilly (1995), CLS · Hu et al. (2021), LoRA.
- Berlyne (1960) · Loewenstein (1994) · Schmidhuber (1991, 2010) · Oudeyer & Kaplan (2007) · Pathak et al. (2017).
- Johnson, Hashtroudi & Lindsay (1993), source monitoring · Harnad (1990), symbol grounding · Graziano, attention schema.
- Bongard, Zykov & Lipson (2006) · Rao & Georgeff (1995), BDI · Brooks (1986) · Ha & Schmidhuber (2018).
- Park et al. (2023). *Generative Agents.* · Packer et al. (2023). *MemGPT.*
- Davis & Mermelstein (1980), MFCC · Sakoe & Chiba (1978), DTW · Graves et al. (2006), CTC · Klatt (1980) · Griffin & Lim (1984) · LeCun et al. (1989).
- Dataset licenses: [Project Gutenberg license](https://www.gutenberg.org/policy/license.html), [robot access policy](https://www.gutenberg.org/policy/robot_access.html), [DailyDialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog), [Speech Commands](https://research.google/blog/launching-the-speech-commands-dataset/), [LJ Speech](https://keithito.com/LJ-Speech-Dataset/).
