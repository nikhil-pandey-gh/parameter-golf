# Leaky Lookahead MTP

## Hypothesis

The current best repository result (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) gets part of its gain from making the model more future-aware at evaluation time with legal score-first TTT, but that adds roughly 410 seconds of extra eval work. This candidate asks whether some of that future-token awareness can be baked into training instead by adding a small, export-free multi-token lookahead objective on top of the strongest clean pre-TTT stack.

The concrete bet is:

- keep the strong `2026-03-22` 11-layer GPTQ-lite + EMA + partial-RoPE base,
- adopt the already-proven `LeakyReLU(0.5)^2` MLP from the `2026-03-23` record,
- replace the dormant full-vocab MTP heads with lighter low-rank tied-output lookahead heads,
- fade the auxiliary loss during warmdown so the final optimization budget recenters on the true next-token objective.

## Why this is promising for this repository

Repository evidence says three things clearly:

- the best clean training stack is the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` record,
- the best overall result (`2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`) benefits from both `LeakyReLU^2` and future-aware evaluation,
- a dormant MTP path has existed in several strong record scripts, but all of those runs kept `MTP_NUM_HEADS=0`, so there is still unexplored room for a cheaper, better-shaped version of lookahead supervision.

That makes this candidate attractive: it reuses the strongest compression-aware training recipe in the repo, borrows a one-line activation improvement that already won, and adds a new training-only future-token signal that costs **zero artifact bytes** because the lookahead heads are stripped before export.

## Prior records and code that influenced this candidate

### Base implementation

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

This candidate starts from that script because it is the cleanest high-performing non-TTT foundation in the repository: 11 layers, XSA on the deepest layers, partial RoPE, BigramHash, VE128, EMA, GPTQ-lite int6 export, and warmdown tuned for the 10-minute budget.

### Additional repository influences

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributes the `LeakyReLU(0.5)^2` MLP idea,
  - motivates the broader “make the model more future-aware” direction.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - useful because its logs show the repo’s MTP path was present but disabled, so this candidate is not just blindly reusing an already-proven setting.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - another prior stack carrying the dormant MTP implementation, again with `MTP_NUM_HEADS=0` in the run configuration.

## External research that informed it

### 1. Multi-token prediction improves sample efficiency

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"**, arXiv:2404.19737

This paper argues that supervising multiple future tokens can improve sample efficiency and strengthen induction-like behavior. That is a natural fit for Parameter Golf’s hard 10-minute training cap: if the model can learn a stronger predictive backbone per seen token, that is often more valuable than adding more infrastructure.

### 2. Latent lookahead suggests future-token supervision can improve foresight without changing final decoding

- Lorenzo Noci et al., **"Thinking into the Future: Latent Lookahead Training for Transformers"**, arXiv:2603.20219

Their core point is that supervising multiple latent future steps can make the model “think ahead” before committing to a token. This candidate does not implement full recursive latent rollouts, but it adopts the same spirit in a much cheaper form: light lookahead heads trained only during optimization, then removed at export.

### 3. Decoder states already contain latent multi-token structure

- Raghavv Goel et al., **"Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing"**, arXiv:2603.17942

This work is important because it suggests models already expose latent multi-token predictive structure in embedding space. That motivated the choice to use **tied-output low-rank lookahead heads** instead of expensive full-vocab untied heads: the head only needs to nudge hidden states toward future-token-aligned directions, not learn an entirely separate output system.

## What changed versus the chosen base implementation

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes five targeted changes:

1. **Candidate-local path defaults**
   - Data and tokenizer defaults now resolve relative to the repository root from the candidate script location, so the script can be launched from `candidates/202603250726_leaky-lookahead-mtp/` directly.

2. **LeakyReLU(0.5)^2 MLP**
   - The MLP activation switches from `relu^2` to `leaky_relu(0.5)^2`, following the strongest repository evidence from the `2026-03-23` record.

3. **Lighter lookahead heads instead of full MTP output matrices**
   - The old dormant MTP path used one full `model_dim -> vocab_size` matrix per future horizon.
   - This candidate replaces that with a `LookaheadHead` consisting of:
     - RMSNorm,
     - a low-rank `down -> SiLU -> up` residual adapter,
     - projection through the shared token embedding when embeddings are tied.
   - Default settings: `MTP_NUM_HEADS=2`, `MTP_RANK=64`, `MTP_TIED_OUTPUT=1`.

4. **Warmdown-aware MTP weighting**
   - `MTP_LOSS_WEIGHT` defaults to `0.15`.
   - The auxiliary weight fades linearly once the LR multiplier falls below `MTP_WARMDOWN_THRESHOLD=0.25`, so the endgame focuses more heavily on the actual next-token objective.

5. **FlashAttention import fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA. This is mainly to make lightweight development/smoke validation easier; challenge training should still prefer FlashAttention 3 when available.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603250726_leaky-lookahead-mtp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the strong `2026-03-22` defaults for the main architecture/training stack, but changes the candidate-specific defaults to:

- `MTP_NUM_HEADS=2`
- `MTP_RANK=64`
- `MTP_LOSS_WEIGHT=0.15`
- `MTP_WARMDOWN_THRESHOLD=0.25`
- `MTP_TIED_OUTPUT=1`
- `MLP_NEGATIVE_SLOPE=0.5`

The script still performs the usual export, int6 roundtrip evaluation, and sliding-window evaluation at the end. The lookahead heads are excluded from the exported artifact.

## Main expected risks or tradeoffs

- **Auxiliary-loss mismatch:** even with warmdown fading, the lookahead loss may still pull capacity away from the true next-token objective.
- **Training-time overhead:** two extra horizons add compute, though the low-rank tied-output version is cheaper than the older dormant full-vocab MTP path.
- **Underpowered head risk:** the low-rank tied-output head may be too weak to realize the gains promised by full multi-token prediction papers.
- **Activation interaction risk:** `LeakyReLU^2` helped in the current best record, but its interaction with this exact EMA/GPTQ-lite stack is still untested.

## Validation

Commands run in this workflow:

1. Syntax check

```bash
python -m compileall candidates/202603250726_leaky-lookahead-mtp/train_gpt.py
```

Outcome: **passed**.

2. Minimal CPU smoke attempt

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch

path = Path('candidates/202603250726_leaky-lookahead-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcome: **not feasible in this container** because the workflow Python environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`).

Because of that missing runtime dependency, I could validate syntax but not execute an actual forward pass here. The script changes were therefore kept tightly scoped to the existing repository patterns, and the FlashAttention fallback was included specifically to reduce non-essential runtime coupling once PyTorch is available.
