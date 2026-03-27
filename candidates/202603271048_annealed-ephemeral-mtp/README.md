# 202603271048_annealed-ephemeral-mtp

## Hypothesis

A **training-only multi-token prediction (MTP)** auxiliary loss should improve sample efficiency for this repository's tiny 16MB-constrained models, provided we do **not** pay artifact bytes for the auxiliary heads and we **anneal** the auxiliary objective away during warmdown so the final checkpoints stay aligned with the primary next-token metric.

In short: use MTP to shape better internal representations early, then let the main next-token loss dominate the finish.

## Why this is promising for this repository

Repo review suggests that most recent wins come from stacking a mature 11-layer recipe and finding small improvements in either:

- training efficiency,
- quantization/export quality, or
- evaluation quality.

The strongest published stack in this repo already combines LeakyReLU^2, Partial RoPE, LN scaling, XSA on late layers, VE layers, EMA, GPTQ-lite/int6 export, and legal score-first TTT in evaluation (`records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`).

The repo review also showed that more invasive ideas like recurrent depth reuse or slower activations have tended to lose on the 600s budget, while smaller training/eval wins keep compounding. That makes MTP attractive here because it is:

- **orthogonal** to the existing record stack,
- **training-time only**, so it can be excluded from the final artifact,
- already partially scaffolded in the latest record code, reducing implementation risk.

## Prior records that influenced this candidate

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Closest architectural predecessor without the legal TTT emphasis:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

These two records established the current best reusable recipe:

- 11-layer banked stack with Parallel Muon,
- LeakyReLU^2 MLPs,
- Partial RoPE + LN scaling,
- XSA on the deepest layers,
- VE reinjection,
- strong post-training int6 export,
- EMA-centered averaging.

There were **no pre-existing `candidates/` folders** in the repository at the time this candidate was created.

## External research that informed it

Primary source:

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"**, arXiv:2404.19737
  - <https://arxiv.org/abs/2404.19737>

Why it matters here:

- The paper argues that predicting multiple future tokens improves sample efficiency by using the same shared trunk to learn richer predictive representations.
- That is especially relevant in this challenge because training time is capped at 10 minutes and the strongest recent repo gains have come from getting more quality out of the same wallclock budget.
- The paper also frames MTP as an auxiliary task on top of a shared trunk, which matches this repo's constraint that the final artifact budget matters more than transient training-only parameters.

I also reviewed parameter-sharing literature during the research pass, especially:

- ALBERT: <https://arxiv.org/abs/1909.11942>
- Universal Transformer: <https://arxiv.org/abs/1807.03819>

Those ideas are still interesting for future experiments, but they looked more invasive than necessary for the next candidate in this codebase.

## What changed versus the chosen base implementation

Base implementation copied from:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes five targeted changes:

1. **Turns MTP on by default**
   - `MTP_NUM_HEADS=2`
   - `MTP_LOSS_WEIGHT=0.15`

2. **Adds horizon-weighted MTP loss**
   - New `MTP_LOSS_DECAY` (default `0.5`) weights nearer future-token heads more strongly than farther ones.

3. **Anneals the auxiliary loss during warmdown**
   - New `MTP_ANNEAL=1` makes the effective MTP loss scale track the same warmdown multiplier as the learning rate.
   - That keeps MTP active when representation learning matters most and fades it out near the final export point.

4. **Actually wires MTP heads into optimization**
   - The copied base code already had `MTP_*` hyperparameters and loss logic, but the auxiliary heads were not added to any optimizer parameter group.
   - This candidate explicitly includes `mtp_heads` in the AdamW/replicated-gradient path so the MTP objective can train real parameters.

5. **Keeps MTP strictly training-only**
   - MTP heads remain excluded from export.
   - EMA/LAWA snapshot tracking now also excludes `mtp_heads`, so averaging state and exported artifacts stay focused on the inference-time model.
   - A small FlashAttention import fallback was added so the script can at least be imported in environments that do not have `flash_attn_interface` installed.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202603271048_annealed-ephemeral-mtp
RUN_ID=annealed_ephemeral_mtp \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults in this candidate already enable the MTP experiment. If you want to stack legal TTT evaluation later, reuse the `TTT_*` flags from the 2026-03-23 record after first measuring the training-only delta from MTP.

## Validation

Commands run in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603271048_annealed-ephemeral-mtp/train_gpt.py
```

Observed outcomes:

- Root lightweight compile check completed successfully.
- Candidate `train_gpt.py` compile check completed successfully.
- A minimal CPU smoke test was **not feasible in this runner** because the available Python environment does not currently have the runtime dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing), so importing the module for a local forward pass was impossible without first rebuilding the environment.

## Main expected risks / tradeoffs

- **Step-time overhead:** extra vocab projections for two future-token heads may cost enough wallclock to erase the sample-efficiency win.
- **Objective interference:** if the MTP loss stays too strong late in training, it may hurt the final next-token objective even with annealing.
- **Averaging choice:** excluding MTP heads from EMA/LAWA is principled for a training-only auxiliary objective, but it is still an empirical choice.
- **Interaction with TTT:** if this candidate is later stacked with legal TTT, the gains may not add linearly.

## Suggested next experiments if this is promising

- Sweep `MTP_NUM_HEADS` over `{1,2,4}`.
- Sweep `MTP_LOSS_WEIGHT` over `{0.05, 0.10, 0.15, 0.20}`.
- Compare `MTP_LOSS_DECAY` values `{1.0, 0.7, 0.5}`.
- Test whether the MTP gain survives when legal TTT is turned back on.
