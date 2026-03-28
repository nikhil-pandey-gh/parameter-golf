# Shared Tail Refiner

## Hypothesis

The repo’s strongest non-TTT stack already extracts most of its gains from a compact 11-pass transformer with careful quantization, EMA, partial RoPE, XSA, and strong eval. My hypothesis is that the next worthwhile architectural move is **not** a full recurrent rewrite, but a **compute-aware late refinement pass**: keep almost the same effective depth, reuse the deepest block one extra time with a learned step embedding, and spend the saved parameters on small high-leverage capacity like a larger BigramHash table.

This targets a middle ground between two repo observations:

- full-depth recurrence looked promising conceptually but was previously too step-hungry for the 10-minute budget,
- late-layer architectural tweaks keep paying off when they improve quality without much extra code or artifact cost.

## Why this is promising for this repository

The best pure training/export record before TTT was `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`, which already combined the strongest reusable ingredients in the repo: 11 effective layers, 3x MLP, XSA on the tail, partial RoPE, LN scale, EMA, VE, and GPTQ-lite.

At the same time, `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` explicitly notes that naive depth recurrence looked promising but needed too many steps for the wallclock budget. This candidate revisits that idea in a narrower form: **reuse only the deepest block as a refinement cell**, so the recurrence lands exactly where the repo already concentrates extra modeling power.

The design also keeps the changes local to an existing strong codepath rather than introducing a brand-new attention alternative or tokenizer stack.

## Influencing records and prior candidates

There were **no prior experiments under `candidates/`** when this candidate was created.

The most relevant record influences were:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
  - chosen as the base because it is the strongest repo stack that does not depend on TTT.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - contributed the `LeakyReLU(0.5)^2` activation, which was reported as a high-value, low-complexity gain.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
  - reinforced keeping partial RoPE and LN scaling in the backbone.
- `2026-03-18_FP16Embed_WD3600`
  - provided the negative-result motivation: revisit recurrence only if it can be made more compute-aware.
- `2026-03-20_10L_Int5MLP_MuonWD04_SWA50`
  - showed that increasing BigramHash capacity can still buy useful quality under tight byte budgets.

## External research that informed this candidate

The candidate is primarily motivated by the parameter-sharing / recurrent-depth literature, but adapted into the repo’s existing transformer stack:

- **ALBERT** (`arXiv:1909.11942`) argues that parameter-reduction techniques can improve memory efficiency and scaling without necessarily giving up downstream quality. That supports reusing depth parameters instead of always paying for unique late blocks.
- **Universal Transformer** (`arXiv:1807.03819`) shows that repeatedly applying a shared self-attentive block can improve language modeling and reasoning, especially when the recurrent step has its own notion of depth/time.
- **RWKV** (`arXiv:2305.13048`) and **Mamba** (`arXiv:2312.00752`) are useful modern reminders that recurrent inductive bias can remain competitive when implemented efficiently. I did **not** rewrite this repo around those architectures; instead I borrowed the narrower lesson that recurrence is worth revisiting when it can be layered onto an efficient training path.

## What changed vs. the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Shared tail refinement**
   - Defaults to `NUM_LAYERS=10` plus `RECURRENT_TAIL_STEPS=1`.
   - The deepest block is run once in the normal decoder path and then reused for one extra refinement step.
   - Each extra step gets its own learned step embedding, following the recurrent-depth intuition from Universal Transformer.
   - If value embeddings are enabled, the recurrent step also gets its own learned VE scale.

2. **LeakyReLU(0.5)^2 MLP activation**
   - Replaces `relu^2` with the activation style that the current best record reported as a strong standalone gain.

3. **Slightly larger BigramHash table**
   - Default `BIGRAM_VOCAB_SIZE` is increased from `2048` to `3072` to spend a fraction of the parameter savings on low-cost lexical capacity.

4. **Candidate-directory-friendly defaults**
   - `DATA_PATH` and `TOKENIZER_PATH` now default to paths resolved from the repository root, so the script can be launched directly from this candidate folder.

Everything else intentionally stays close to the base stack: EMA, GPTQ-lite export path, partial RoPE, LN scaling, XSA tail layers, VE, warmdown, and the optimizer split.

## How to run or evaluate it

From this directory:

```bash
RUN_ID=shared_tail_refiner \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs for ablation:

```bash
NUM_LAYERS=10 \
RECURRENT_TAIL_STEPS=1 \
BIGRAM_VOCAB_SIZE=3072 \
VE_LAYERS=8,9 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
python -m compileall train_gpt.py
```

The script keeps the same evaluation/export flow as the base record: validation BPB, mixed int6 export, roundtrip eval, and sliding-window eval.

## Main expected risks and tradeoffs

- **Shared depth may still underperform unique depth.** Reusing the last block saves bytes, but it also removes one set of unique parameters from the backbone.
- **The recurrent tail could still cost too many steps.** This is much lighter than looping the whole stack, but it still adds one more full late-block pass.
- **LeakyReLU² plus recurrence may interact nonlinearly.** Both are individually plausible here, but the combination has not been repo-validated yet.
- **Artifact budgeting still matters.** The candidate spends some of the saved depth budget on a larger BigramHash table, so export size should still be checked carefully on a real training run.

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603281745_shared-tail-refiner/train_gpt.py
```

Outcomes:

- `python -m compileall train_gpt.py train_gpt_mlx.py data` ✅
- `python -m compileall candidates/202603281745_shared-tail-refiner/train_gpt.py` ✅

Minimal CPU smoke test:

- **Not feasible in this environment.** The local Python runtime here does not have the required `torch` and `sentencepiece` packages installed, and the training script is CUDA-oriented, so I could not run a trustworthy forward-pass smoke check without inventing new infrastructure.
