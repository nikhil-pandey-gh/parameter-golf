# Candidate: MTP auxiliary heads on the 2026-03-23 SOTA stack

## Hypothesis

The current best Parameter Golf recipe already has a strong shared trunk: 11 layers, U-Net skips, XSA on deep layers, partial RoPE, EMA, GPTQ-lite export, and legal score-first TTT. My hypothesis is that this trunk is now good enough that a small **multi-token prediction (MTP)** auxiliary loss can improve representation quality and sample efficiency without increasing the exported artifact size, because the auxiliary heads are training-only and are excluded from the final checkpoint.

## Why this is promising for this repository

Recent records suggest the biggest gains now come from **small additions to an already strong 11-layer backbone**, not from restarting with a very different model family. The record progression improved by stacking inexpensive quality wins such as EMA, better post-training quantization, partial RoPE, LeakyReLU(0.5)^2, and legal TTT, while older coarse changes like just lengthening context or using simpler mixed precision gave smaller gains.

This candidate keeps the full winning stack intact and adds one training-only improvement that should be cheap in artifact bytes:

- `MTP_NUM_HEADS=2`
- `MTP_LOSS_WEIGHT=0.15`

The repo already carried dormant MTP support in several strong scripts, but every inspected log still reported `mtp_num_heads:0`, so this remains untested in practice.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Chosen base implementation. Keeps parameter banking, Parallel Muon, LeakyReLU(0.5)^2, partial RoPE, XSA, EMA/SWA, GPTQ-lite-style export, and legal TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Confirms that small quantization/averaging refinements still matter at this stage of the leaderboard.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Shows that zero-byte architectural nudges like partial RoPE and layer scaling can outperform simpler earlier stacks.
- Earlier sequence-length-only and simpler precision runs
  - Helpful as negative evidence: straightforward long-context or mixed-precision-only changes helped, but not enough to match the deeper, better-regularized 11-layer line.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** (arXiv:2404.19737)  
  https://arxiv.org/abs/2404.19737
  - Trains a shared trunk with multiple future-token heads and reports higher sample efficiency plus stronger induction-like behavior.
- DeepSeek-AI, **"DeepSeek-V3 Technical Report"** (arXiv:2412.19437)  
  https://arxiv.org/abs/2412.19437  
  https://arxiv.org/html/2412.19437v2
  - Uses a multi-token prediction training objective in a modern frontier recipe, which makes MTP feel like a live systems trick rather than an isolated academic result.

I also considered more invasive ideas from the literature such as MLA-lite K/V compression, ALBERT-style cross-layer sharing, and LSQ/PACT-style QAT refinements. Those remain interesting, but MTP was the best fit for a first candidate because it reuses existing code paths and adds almost no exported-byte risk.

## What changed vs. the chosen base implementation

Relative to `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes three targeted changes:

1. **Turns MTP on by default**
   - `MTP_NUM_HEADS`: `0 -> 2`
   - `MTP_LOSS_WEIGHT`: `0.2 -> 0.15`
2. **Fixes an optimizer wiring bug**
   - The base script logged `mtp_num_heads` and built `mtp_heads`, but did not actually place those weights into any optimizer group.
   - This candidate adds the auxiliary heads to the head optimizer, so the MTP path is trainable instead of inert.
3. **Makes default data/tokenizer paths robust when run from this candidate directory**
   - Defaults now resolve relative to the repository root instead of assuming the current working directory is the repo root.

## How to run

From this candidate directory:

```bash
RUN_ID=mtp_aux_heads \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already keep the 2026-03-23 stack and enable MTP. To ablate the change and recover the base behavior:

```bash
RUN_ID=mtp_ablation_off \
SEED=1337 \
MTP_NUM_HEADS=0 \
MTP_LOSS_WEIGHT=0.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script preserves the base export and evaluation flow:

- EMA weights are applied after training
- MTP heads are excluded from the exported checkpoint
- mixed int6 + lzma roundtrip evaluation is preserved
- stride-64 sliding-window evaluation is preserved
- legal score-first TTT is still available when `TTT_ENABLED=1`

## Main expected risks and tradeoffs

- **Wallclock tradeoff:** auxiliary heads add extra logits and cross-entropy work during training, so step time may increase enough to reduce the number of steps reached in 600 seconds.
- **Optimization coupling:** too much auxiliary loss can help trunk learning early but hurt next-token specialization late; `0.15` is meant to be a conservative compromise.
- **No byte savings by itself:** unlike quantization or layer sharing, MTP is a quality-per-step bet, not an artifact-size bet.
- **Training-only benefit:** if MTP helps only marginally, the extra compute may not pay for itself under the challenge’s strict wallclock limit.

## Validation

Repository baseline check:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
```

Outcome:

- Passed in this workflow runner.

Candidate syntax check:

```bash
python -m compileall candidates/202603311223_mtp-aux-heads/train_gpt.py
```

Outcome:

- Passed in this workflow runner.

Attempted smoke import / constructor check:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("candidates/202603311223_mtp-aux-heads/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("loaded", mod.Hyperparameters.mtp_num_heads, mod.Hyperparameters.mtp_loss_weight)
PY
```

Outcome:

- Not feasible in this runner as-is.
- The container is missing core declared runtime dependencies (`numpy`, `sentencepiece`, `torch`) and also lacks `flash_attn_interface`, so the script cannot be imported far enough to run a meaningful CPU smoke test without installing substantial extra GPU-oriented dependencies.
- Because of that environment limit, validation here is limited to Python syntax compilation plus static review of the MTP optimizer wiring and export path.
