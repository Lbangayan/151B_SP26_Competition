# Project Plan — CSE 151B Math Reasoning Competition

This document captures the **phased workflow** for the rest of the project,
from milestone (done) through final submission. The core insight driving the
phasing is that **vLLM and transformers are complementary tools, not
substitutes**: vLLM gives 5–20× faster inference but cannot train; transformers
gives full training capability but slower inference. The optimal pipeline uses
each library for what it's good at.

For codebase-level guidance (file layout, eval pipeline, judger details), see
[CLAUDE.md](CLAUDE.md). For the design discussion of RL algorithms (MDP
formulation, RFT / DPO / GRPO walk-through), see [IDEA.md](IDEA.md).

## Tooling thesis

| Phase activity | Library |
|---|---|
| Inference / evaluation / prompt engineering | **vLLM** |
| Supervised fine-tuning (SFT) | **transformers + peft + trl** |
| RL fine-tuning (GRPO, DPO, rejection-sampling FT) | **transformers + peft + trl** |
| Final submission generation | **vLLM** (with the trained LoRA adapter loaded) |

The hand-off: train a LoRA adapter via `peft + trl`, save the adapter weights
(typically ~100 MB), then load the adapter into vLLM via
`LLM(enable_lora=True)` + `LoRARequest`. Adapter inference in vLLM is
first-class and well-documented.

## Why this ordering matters

Two reasons to **prompt-engineer before fine-tuning**, not the other way around:

1. **Prompt engineering is cheap.** A run on 20 questions takes ~50 minutes;
   on 100 questions, a few hours. Fine-tuning is much more expensive
   (multi-hour runs on a cluster GPU per checkpoint).
2. **Fine-tuning changes prompt response patterns.** A model SFT'd on
   CoT-style math data may make verbose few-shot exemplars redundant or even
   harmful. We expect to **re-validate prompts on the fine-tuned model**, but
   doing a first pass on the base model gives us a strong starting point and
   teaches us what kinds of interventions help on the eval set.

The corollary: don't over-invest in prompt engineering on the base model.
Lock in the top 2–3 candidates and move on; the post-FT re-validation is
where the final tuning happens.

---

## Phase 1 — Baseline + first-pass prompt engineering ✅ DONE (milestone)

**Goal**: establish a working pipeline; test three prompt-engineering ideas
on a small, fixed evaluation subset.

- [x] Stratified 20-question evaluation subset (`results/eval_subset.json`).
- [x] Prompt-hash--keyed response cache (`results/cache/*.jsonl`).
- [x] Baseline run (13/20 = 65%).
- [x] Few-shot static exemplars (12/20 = 60%).
- [x] Single-turn embedded reflection (11/20 = 55%).
- [x] Crash-safe mini-batched generation (`BATCH_SIZE=5`).
- [x] Milestone writeup.

**Honest milestone read**: at N=20, the three prompt variants are
statistically indistinguishable from baseline. The most actionable
sub-finding is that the long-question bucket (multi-part word problems)
improved with both interventions, while short-question buckets regressed —
suggesting the gains from "smarter" prompts can be undone by losses on
questions the baseline already solved. **Conditional prompt routing**
(applying the multi-part exemplar only when `[ANS]` appears multiple times,
or applying reflection only on long questions) is a promising direction
for Phase 2.

---

## Phase 2 — Scale up evaluation on DSMLP (in progress)

**Goal**: increase evaluation power; finish self-consistency; build the
**v2 prompt engineering pipeline** (conditional routing + LaTeX preprocessing
+ validation/retry); lock in the strongest base-model configuration for use
as the starting point in fine-tuning.

The laptop's 8 GB VRAM was too constrained for self-consistency (K parallel
samples per question exceeded the KV-cache budget). DSMLP's 24 GB MIG slice
removes that constraint.

**Status update — v2.1 is shipped in the baseline notebook.** The
preprocessing, routing, and MCQ stage-2 reconciliation (which is the
targeted-reflection realization of the "validation + retry" idea) all
work end-to-end. Remaining work in Phase 2 is filling in the specialized
free-form prompts, migrating the v2.1 scaffolding to the SC notebook,
expanding the subset, and running SC.

### Prompt engineering pipeline (v2) — design

The v1 milestone tested baseline / few-shot / reflection / SC as **independent
unconditional methods** — each variant applied the same prompt to every
question. The v1 result pattern (long bucket improved, short bucket
regressed) suggests universal application is the wrong default.

v2 is a **conditional pipeline**: it applies methods based on question
properties, then validates outputs after generation, with bounded retry on
failure.

```
preprocess(question)                        ← LaTeX repair (deterministic)
        ↓
prompt = select_prompt(question)            ← type/length-conditional
        ↓
samples = vllm.generate(prompt, n=K)        ← self-consistency
        ↓
extracted = [extract_answer(s) for s in samples]
        ↓
if validation_fails(extracted, question):
    samples = vllm.generate(prompt + retry_hint, n=K)   ← bounded retry
    extracted = [extract_answer(s) for s in samples]
        ↓
final = majority_vote(extracted) or extracted[0]
```

**Pre-inference: LaTeX typo repair.** Investigation revealed systematic
LaTeX transcription errors in the dataset — math commands like `int`,
`frac`, `infty` appear without their leading backslash. For example,
`int_{-infty}^{+infty} frac{a^{3/2}}{s^2+a^2}` should be
`\int_{-\infty}^{+\infty} \frac{a^{3/2}}{s^2+a^2}`. External LLMs
(GPT, Claude) confirmed that without seeing multiple-choice options,
even strong reasoners cannot recover the intended question — they
compute the literal-but-meaningless answer. A deterministic preprocessor
fixes this for known commands using a regex pass. **Implemented as
`repair_latex` / `preprocess_question` in the baseline notebook** — uses
a context-aware regex (command must be followed by `{ } _ ^ ( )`) to
avoid false positives on English words like "the tan of x".

**Prompt selection: conditional routing.** The system prompt is selected
per question. `select_prompt(item)` returns
`(system_prompt, user_prompt, options_visible: bool)`; the third return
value is the single source of truth for whether the MCQ took the
visible-options route (skip verify + stage-2):

| Question type | Prompt | Status |
|---|---|---|
| **MCQ — heuristic-flagged** ("which of the following ..." / "all of the above") | `SYSTEM_PROMPT_MCQ` with options inline (single-stage; minimalized + typo-aware in v2.6) | ✅ shipped (v2.1, refined v2.6) |
| **MCQ — derivation** (everything else, **slot 4** = 369 items) | `SYSTEM_PROMPT_MATH`, options hidden in stage-1 (intentionally shares prompt with slot 1) | ✅ shipped |
| Multi-part free-form (`[ANS]` count > 1, **slot 2** = 415 items) | `SYSTEM_PROMPT_MULTIPART` (Plan-and-Solve-style "identify each sub-question, solve in order, output all answers in ONE `\boxed{}`") | ✅ shipped (v2.3) |
| Long single-answer free-form (> 500 chars, **slot 3** = 23 items) | `SYSTEM_PROMPT_LONG` (CoRe-style: restate + enumerate conditions before solving) | ✅ shipped (v2.2) |
| Short single-answer free-form (**slot 1** = 302 items) | `SYSTEM_PROMPT_MATH` (Qwen-official minimal: `"Please reason step by step, and put your final answer within \boxed{}."` + multi-answer safety hint) | ✅ shipped (v2.4) |

The MCQ heuristic (`mcq_needs_options(question, options)`) catches roughly
1.6% of MCQs in the public dataset — small fraction, but those are exactly
the cases where the two-stage flow was wasted compute (model can't derive
"which option has property P" without seeing the options).

### Prompt design philosophy (v2.2)

All Phase-2 prompts follow Thinking-model best practices established by
the DeepSeek-R1 paper, the Qwen3-4B-Thinking-2507 model card, and
practitioner guides (Helicone, Together AI):

- **No few-shot exemplars.** Multiple sources independently report that
  exemplars *degrade* accuracy on Thinking-model variants (R1, o1,
  Qwen3-Thinking) because they compete with the model's internal `<think>`
  CoT. This invalidated the original `_FEWSHOT` plan for slots 2 and 3 —
  both now use exemplar-free structural prompts.
- **No "think step by step" preamble.** Redundant; the model already CoTs
  internally. The Qwen team's own recommended math prompt is just *"Please
  reason step by step, and put your final answer within `\boxed{}`."*
- **No verbose persona.** Minimal, declarative, output-contract-focused.

**Slot 3 (long-context, `SYSTEM_PROMPT_LONG`)** is the exception that
adds structural content beyond the output contract. Based on
[Xu et al. 2024 (E-GSM, "Can LLMs Solve Longer Math Word Problems
Better?")](https://arxiv.org/html/2405.14804v1), forcing the model to
**restate the problem and enumerate every given condition + the quantity
to find** before solving (the CoRe technique) gave +5.34pp on extended
problems for GPT-3.5, +15pp combined with self-consistency. This matches
our v1 milestone's 25%-on-long-bucket failure profile (the same
short→long dropoff Xu et al. quantified at ~16pp for general LLMs).

**Slot 2 (multi-part, `SYSTEM_PROMPT_MULTIPART`)** also adds structural
content. The challenge here is **format compliance under output
structure**, not reasoning depth. Population analysis: 415/751 (55%) of
free-form questions are multi-part — the **largest** single bucket, and
178 of those 415 are also long (>500 chars), so this prompt is doubly
load-bearing. Three failure modes the prompt explicitly guards against:
(a) wrong answer count, (b) wrong order, (c) splitting answers across
multiple `\boxed{}` (the judger's `extract_boxed_answer` keeps only the
last contiguous group of boxes — text between them = lost answers).
The prompt uses a Plan-and-Solve-style preamble ("identify each
sub-question, solve in order") which doubles as condition-enumeration
for the long-multipart subset, then enforces the single-`\boxed{}`
output contract with an explicit anti-pattern guard.

Population sizes (free-form only, n=751): multi-part=415 (55%),
single-long=23 (3%), single-short=302 (40%). **Slot 2 is the highest-
volume free-form route by a wide margin** — slot 3 only covers 3% of
free-form items because most long questions are *also* multi-part and
get routed to slot 2 first.

**Slot 1 (generic short, `SYSTEM_PROMPT_MATH`) — minimalized in v2.4.**
Replaced the verbose starter prompt ("You are an expert mathematician.
Solve the problem step-by-step...") with the Qwen3-Thinking-2507 model
card's official math prompt verbatim: *"Please reason step by step, and
put your final answer within `\boxed{}`."* — plus the multi-answer
safety hint as defense-in-depth. This same constant is also reused by
slot 4 (MCQ derivation, hidden options) and as the system role in
`build_mcq_stage2_messages`. The "expert mathematician" persona was
removed per the no-verbose-persona finding from the design philosophy
section above.

**Slot 4 (MCQ derivation, hidden options) — 369 items, intentionally
shares slot 1 prompt.** The model receives an MCQ question without the
options and must derive a clean answer that `verify_against_options`
can match. Population analysis confirms this is the right call:
slot 4 questions are structurally indistinguishable from slot 1 from
the model's perspective (same math-domain derivation problems written
as statements). Specializing slot 4 would force a divergence that's
hard to maintain for marginal benefit. If verify-rate turns out to be
poor in measurement, we can revisit and add a "produce a clean
closed-form answer" hint specifically for slot 4.

**Slot 5 (MCQ visible-options) and slot 6 (MCQ stage-2 reconciliation)
— minimalized in v2.6.** Both prompts had been left unchanged through
v2.5. In v2.6 they were aligned with the same v2.4 minimal-prompt
philosophy applied to slot 1: persona language ("You are an expert
mathematician") removed from slot 5, output contract sharpened with the
explicit `\boxed{C}` example for consistency, and typo-awareness clause
added to slot 5 (since the model sees options at this stage, charitable
interpretation against the option set helps recover transcription
errors). Slot 6's wording was also tightened ("options shown above"
makes the reference explicit, single-sentence reconcile instruction).

**Future-work idea: model-as-router for MCQ stage-1.** Documented in
[IDEA.md §5](IDEA.md). Proposes letting the model itself flag
options-required questions via a sentinel string in the slot-4 prompt
(`\boxed{___INSUFFICIENT_CONTEXT___}`), with detection in the
generation loop routing the item directly to stage-2. Deferred until
after the v2.6 baseline run, since the v2.5 heuristic strengthening
likely closed most of the heuristic gap; whether this idea pays off
depends on the per-batch `stage-2=Z` count we'll see from the run.

**Heuristic strengthening (v2.5, slot 5 routing).** Dataset inspection
surfaced two pattern classes the original `_OPTIONS_REFERENCED` regex
missed, both addressed in v2.5:

- *"what is correct"* — caught id 154 ("The following logical calculus,
  what is correct?") and similar phrasings where options are
  semantically required but the canonical "which of the following"
  trigger isn't used. **1 net new flag.**
- *"determine the (corresponding )?output"* — caught the **39
  algorithmic-output MCQs** (ids 73, 87, 91, 420, 600, 612, 800, 891,
  ...) of the form "We now define an algorithm: ... Given input x_list
  [10 values], determine the corresponding output sequence y_list."
  All options for these are **lists of 10 integers**;
  `verify_against_options` is brittle on list matching even if the model
  computes correctly (one off-by-one element → no match). Showing
  options lets the model "compute and pick the closest" instead of
  needing exact list reproduction. **39 net new flags.**

Population shift: heuristic-flagged MCQs went from **6/375 (1.6%) →
46/375 (12.3%)**. False-positive guards verified intact: phrases like
*"following the procedure"*, *"select an integer"*, *"what is the
correct value of x"* (with intervening words), and *"determine the
area"* still do **not** trigger the heuristic.

**Compute budget update for v2.2:** We bumped `MAX_TOKENS` from 8192 to
**32768** (Qwen's official model card recommends up to 81920 for
competition math). The 8192 cap on the laptop was almost certainly
truncating long-problem reasoning mid-`<think>`, and is a likely
contributing factor to the 25% long-bucket result independent of the
prompt itself. `max_model_len` was bumped to **49152** to accommodate
the bigger generation budget plus prompt headroom for stage-2 reconciliation
(whose prompt includes the full stage-1 response).

**Sources backing these design choices:**
- [Qwen3-4B-Thinking-2507 model card (HuggingFace)](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507) — official sampling params, output contract, and 81920-token recommendation for competition math
- [Xu 2024 — Can LLMs Solve Longer Math Word Problems Better? (E-GSM)](https://arxiv.org/html/2405.14804v1) — CoRe restate+enumerate intervention, +5.34pp standalone, +15pp with SC
- [DeepSeek-R1 paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) — empirical evidence that few-shot degrades Thinking-model accuracy
- [Helicone — How to Prompt Thinking Models](https://www.helicone.ai/blog/prompt-thinking-models) — practical anti-pattern catalog
- [Together AI — Prompting DeepSeek R1](https://docs.together.ai/docs/prompting-deepseek-r1) — corroborates no-exemplar / minimal-prompt findings

**Inference: self-consistency.** Sample K responses per question via
vLLM's `n=K` parameter. Start at K=3, scale to K=8 once compute permits.
Voting is on the extracted answer string, not the full response.

**Post-inference: validation and bounded retry.** For each question:
1. Extract the answer from each of K samples.
2. If the answer fails validation (rules below), retry once with a
   stronger format-enforcement prompt.
3. Majority-vote among valid extracted answers.

Validation rules:
- **MCQ**: extracted letter must be in the set of valid option letters.
- **Free-form (single)**: must have at least one `\boxed{...}`.
- **Free-form (multi-part)**: number of comma-separated items inside
  `\boxed{...}` must equal the question's `[ANS]` count.

Retry is bounded to one extra round. If retry also fails, fall back to
whatever the original samples produced (best-effort).

**Realization in v2.1 — stage-2 IS the targeted-reflection retry.** The
baseline notebook implements the MCQ branch of this pattern via the
two-stage flow: if `verify_against_options(stage1_response, options)`
returns None (no option matched), `build_mcq_stage2_messages` constructs
a multi-turn chat that shows the model its own stage-1 reasoning plus
the options, and asks it to reconsider. This is the same retry pattern
this section describes, but **only fired when there's a concrete failure
signal**, which is critical: the milestone showed unconditional
reflection regressed (11/20 vs 13/20 baseline) because reflecting on
already-correct answers introduces noise. The free-form retry version
of this pattern (trigger when no `\boxed{}` is present) is not yet
implemented — see Phase 2c checklist.

**Prompt text — instructions baked into the system prompts.**
- *Charitable interpretation*: "The mathematical formulas may contain
  transcription errors (missing backslashes, unclear notation);
  interpret the question charitably."
- *MCQ option-as-constraint*: "If your computed answer does not match
  any option, consider whether the question contains a typo and choose
  the option that would make the question mathematically sensible."
- *Restate-before-solve* (long questions): "Before solving, briefly
  restate the problem in your own words and note any potential
  ambiguities. Then solve."

**What's allowed vs. forbidden at inference time** (competition rules):

| | Allowed at inference | Forbidden at inference |
|---|---|---|
| Pre-generated few-shot exemplars (text baked in offline) | ✅ | — |
| External LLM call to help solve a test question | — | ❌ |
| Tool-augmented generation (code interpreter, calculator) | — | ❌ |
| Curating training data via external LLM (Phase 3) | ✅ | — |
| Using external LLMs to *diagnose* failure modes offline | ✅ | — |

The teammate's "ask GPT/Claude to solve failed questions" workflow is
allowed as **diagnostic research** (it informed the typo-awareness
strategy) but the strategy itself runs entirely inside Qwen3 at
inference time, with no external calls.

### Phase 2 checklist

**2a. DSMLP setup and self-consistency baseline**

- [x] Launch DSMLP pod (`launch-sp26-cuda128.sh`) and verify model loads.
- [x] Resolve the CUDA-version mismatch (`vllm<0.20` for the CUDA 12.8 container).
- [ ] Run self-consistency at K=3 on the existing 20-question subset
      (cache hits for previously-run prompt conditions; only SC needs
      new generation).
- [ ] Run self-consistency at K=8 once K=3 results are interpreted.

**2b. Build the v2 pipeline**

- [x] Implement `repair_latex(text)` preprocessor (context-aware regex on
      `{ } _ ^ ( )` to avoid false positives). Spot-checked against
      English-word collisions ("the tan of x", "in this case").
- [x] Implement `select_prompt(item)` conditional routing (returns
      3-tuple `(system, user, options_visible)`).
- [x] MCQ stage-2 reconciliation (`build_mcq_stage2_messages` +
      `verify_against_options`) — the targeted-reflection retry.
- [x] Heuristic-routed visible-options branch
      (`mcq_needs_options(question, options)`) for "which of the
      following ..." MCQs — single-stage with options inline.
- [x] Crash-safe mini-batch generation (BATCH_SIZE=5) wired through
      stages 1+2 with append-after-each-batch caching.
- [x] Per-routing-path accuracy in summary cell.
- [x] **Author the multi-part free-form prompt** (`SYSTEM_PROMPT_MULTIPART`,
      shipped in v2.3) — Plan-and-Solve-style "identify each sub-question,
      solve in order" preamble + explicit single-`\boxed{}` output contract
      with anti-pattern guard against multi-box splitting. No exemplars.
      Highest-volume free-form route (415 items / 55% of free-form).
- [x] **Author the long-context free-form prompt** (`SYSTEM_PROMPT_LONG`,
      shipped in v2.2) — CoRe-style restate + enumerate conditions
      before solving, no exemplars. Wired into `select_prompt`. Long
      bucket scored 25% in v1 baseline; expected lift +5pp standalone,
      +15pp with self-consistency per Xu 2024.
- [x] **Bump compute budget**: `MAX_TOKENS` 8192 → 32768, `max_model_len`
      16384 → 49152 (DSMLP can handle it; 8192 was likely truncating
      long-problem reasoning mid-`<think>`).
- [ ] **Implement free-form retry** mirroring the MCQ stage-2 pattern:
      trigger when stage-1 produces no `\boxed{}` (or a malformed one);
      reuse the multi-turn chat shape with a format-enforcement nudge.
- [ ] **Migrate v2.1 scaffolding** (preprocess + select_prompt + stage-2)
      into the few-shot, reflection, and SC notebooks. Without this,
      future A/B comparisons confound v1-vs-v2 with the prompt difference
      under test.

**2c. Scale up evaluation**

- [ ] Expand the evaluation subset from N=20 to **N ≥ 100** (stratified
      by type + length, with the same `random.Random(SEED)` recipe so
      it's deterministic and reproducible).
- [ ] Re-run baseline + the best v1 variant + the v2 combined pipeline
      on the expanded subset. Most cache hits should carry over for
      the baseline; v2 needs fresh generation.
- [ ] Compute per-question baseline solve-rate on the expanded subset.
      This becomes the new difficulty proxy for future stratification
      (replaces the question-length heuristic).
- [ ] **Decide the winning base-model configuration** (preprocessor
      on/off × routing scheme × K). Lock it in for Phase 3.

**Definition of done for Phase 2**: the v2 pipeline shows a
statistically meaningful improvement over baseline on N ≥ 100 (delta is
several standard errors above zero), and the winning configuration is
documented and pinned.

---

## Phase 3 — Supervised fine-tuning (SFT)

**Goal**: improve the base model's math reasoning by training on
high-quality reasoning data.

This is where we **switch to `transformers + peft + trl`**. The eval
pipeline stays in vLLM; we just swap which model weights vLLM loads.

### 3a. Dataset curation

- [ ] Collect public math reasoning datasets:
      [GSM8K](https://huggingface.co/datasets/gsm8k),
      [MATH](https://huggingface.co/datasets/hendrycks/competition_math),
      possibly subsets of
      [OpenMathInstruct](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2)
      or
      [NuminaMath](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT)
      depending on disk / compute budget.
- [ ] Format examples to match the competition's expected output
      (`\boxed{}` final answer, possibly with the per-step reasoning
      preserved from the source dataset).
- [ ] Hold out a small validation slice (e.g., 5%) that we never train on,
      separate from the competition's public eval subset.

### 3b. LoRA / QLoRA training

- [ ] Set up `transformers + peft + trl` on DSMLP (`pip install --user trl
      peft datasets accelerate`).
- [ ] Train with `trl.SFTTrainer` using LoRA (start with QLoRA if VRAM is
      tight — 4-bit base + LoRA adapters).
- [ ] Reasonable starting hyperparameters: LoRA rank 16, alpha 32,
      target attention + MLP modules, learning rate 2e-4, ~1 epoch over
      a 50k-example dataset, batch size matched to VRAM (likely 1 with
      gradient accumulation).
- [ ] Save the adapter to `models/sft_lora/`.

### 3c. Evaluate SFT model

- [ ] Load the SFT adapter into vLLM (`LLM(enable_lora=True, ...)` +
      `LoRARequest`).
- [ ] Run the **same evaluation subset and the same prompts as the Phase
      2 winner**. Cache invalidation isn't automatic for model changes,
      so use a fresh cache directory or rename the cache key to include
      a model tag.
- [ ] Compare SFT + Phase-2-winning-prompt vs. base + Phase-2-winning-prompt.

### 3d. Re-validate prompts on SFT model

Because fine-tuning changes prompt response patterns:

- [ ] Re-test the top 2–3 prompts from Phase 2 on the SFT model.
- [ ] Try one or two new prompts designed with the SFT-trained behavior
      in mind (e.g., shorter exemplars if the SFT model already CoTs
      verbosely).
- [ ] Lock in the **Phase 3 winning prompt** = SFT model + best prompt.

---

## Phase 4 — Reinforcement learning

**Goal**: push the model further by using the judger as a verifiable
reward signal, training the model to produce more correct answers.

See [IDEA.md](IDEA.md) for the full algorithm walk-through (MDP formulation,
RFT, DPO, GRPO). The plan recommends **RFT → GRPO** as the ladder.

### 4a. Rollout pipeline

The rollout loop (generate K responses per question, score with judger) is
required by both RFT and GRPO. We build it once and use it for both:

- [ ] Script: for each problem in the training set, sample G=8 responses
      via vLLM (using the trained SFT adapter as the starting policy),
      score each with `judger.auto_judge`, write
      `(question_id, response, reward, logprobs)` to a rollouts JSONL.
- [ ] `logprobs` are needed for GRPO (importance ratio); RFT just needs
      the responses and rewards.

### 4b. Rejection-sampling fine-tuning (RFT)

- [ ] Filter the rollouts to only those with `reward == 1`.
- [ ] Run `trl.SFTTrainer` on these filtered (question, correct_response)
      pairs, starting from the Phase 3 SFT checkpoint.
- [ ] Evaluate the RFT model on the same eval subset.

### 4c. GRPO

- [ ] Run `trl.GRPOTrainer` using:
      - Policy model: the Phase 3 SFT checkpoint (or the RFT checkpoint)
      - Reward function: `judger.auto_judge` (1.0 if correct, 0.0 else)
      - G=8 samples per prompt
      - PPO clip 0.2, KL coefficient ~0.04 (TRL defaults are reasonable)
- [ ] Evaluate the GRPO model on the same eval subset.

### 4d. Final policy selection

- [ ] Compare base, SFT, RFT, GRPO checkpoints on a large (full public
      set or N ≥ 200) evaluation subset.
- [ ] Pick the strongest checkpoint and lock it in for submission.

---

## Phase 5 — Final submission preparation

**Goal**: generate predictions on the private test set, package the
submission, hit the leaderboard.

- [ ] Load the **final selected checkpoint** (adapter + base) into vLLM.
- [ ] Apply the **final selected prompt** (whichever variant won the
      Phase 4 evaluation).
- [ ] Generate predictions on the **private test set** with
      `SAVE_EVAL=False` (the private set has no gold answers).
- [ ] Output format: `{id, is_mcq, response}` per line.
- [ ] Submit to Kaggle.
- [ ] Iterate sampling parameters (temperature, top-p, K for SC) if time
      permits — submissions are unlimited.

---

## Cross-cutting concerns

### Compute budget

- **Laptop** is sufficient for Phase 1 only.
- **DSMLP** (24 GB MIG slice) is the workhorse for Phases 2–4. Pod
  launches are ephemeral; `~/.local/` and `~/.cache/huggingface/` are
  persistent.
- If we hit DSMLP throughput limits during RL training (rollouts are slow),
  the fallback is to **rent cloud GPUs** (Lambda Labs / RunPod) for
  larger-batch GRPO. Daniel's $300 GCP free credit is the backup budget
  for this.

### Numerical determinism

vLLM and transformers use slightly different attention kernels, so the
exact same prompt + seed + sampling params can produce different outputs
across libraries. This matters for two transition points:

1. **Base model → SFT model**: We're already changing the model; minor
   numerical drift from changing libraries is ignored.
2. **SFT'd in transformers → evaluated in vLLM**: Use vLLM consistently
   for evaluation across all conditions. Don't mix-and-match.

### Prompt versioning

Treat each "prompt + model" combination as a distinct experimental
condition. Use the existing prompt-hash cache key, plus a manual model
tag in the cache filename, e.g.:

```
results/cache/{prompt_hash}_seed{seed}_model_{base|sft|rft|grpo}.jsonl
```

This way, switching between models doesn't accidentally hit stale cache
entries.

### Reproducibility

Keep everything we did for the milestone (frozen eval subset, fixed seed,
cache files) in every later phase. The cache is what lets us iterate
prompts cheaply.

---

## Team responsibilities

(Per current `mileStoneTemplate.tex`, may shift):

- **Leonardo**: prompt engineering extensions (dynamic / retrieval-based
  few-shot, conditional routing) for Phase 2.
- **Noah and Tony**: dataset curation (Phase 3a) and SFT pipeline
  (Phase 3b/c) on DSMLP.
- **Daniel**: RL pipeline (Phase 4) — rollouts, RFT, GRPO. Coordinates
  with Tony on the SFT → RL hand-off.

---

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| SFT degrades performance via catastrophic forgetting on non-math tasks | Use small LoRA rank; train for 1 epoch max; eval before/after on the same subset |
| GRPO is unstable (KL collapse, reward hacking) | Watch KL to reference model; lower learning rate; if collapse: raise the KL coefficient |
| `judger.auto_judge` has extraction bugs that GRPO exploits | Sanity-check the judger on 20 sample model outputs before training; add a length penalty if rollouts hit `MAX_TOKENS` |
| Phase 2's expanded subset doesn't show clear improvement | Bigger N is the answer for noise, but also try conditional prompt routing — small targeted interventions are more likely to net out positive than universal ones |
| DSMLP queue / unavailability | Cloud GPU fallback (GCP $300 credit) |
| Running out of time | Stay disciplined about the phase ordering. Each phase has a clear "definition of done" — don't get stuck polishing one phase forever. |
