# Training-Only MTP Auxiliary Heads

## Hypothesis

The strongest current stack already contains dormant multi-token prediction (MTP) support, but every reviewed record kept it off. Turning on a small training-only MTP objective should improve sample efficiency inside the fixed 600-second training budget, while export-time stripping of the auxiliary heads keeps the final artifact on the same 16 MB footing as the base model.

## Why this is promising for this repository

- The leaderboard trend is dominated by compounding training-efficiency wins on a shared 11-layer, 2048-token, compression-aware backbone rather than by wholesale architecture replacements.
- The repository has repeatedly rejected ideas that spend too much wallclock on extra recurrent depth, but MTP is a lighter trade: more supervision from the same forward pass, with no need to retain extra heads in the shipped artifact.
- The copied base already knows how to exclude `mtp_heads.*` from export, so the candidate can test the idea without adding new evaluation or serialization infrastructure.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`: strongest overall stack; this candidate copies that script because it already has LeakyReLU^2, parameter banking, Parallel Muon, and train-only MTP export stripping.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`: strongest clean pre-TTT stack and a good reminder that small training/export improvements still move the needle on the 11L backbone.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/` and `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`: establish partial RoPE, LN scaling, XSA, EMA, and the 11-layer 3x-MLP stack as the stable winning line.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`: useful negative result showing naive layer recurrence can lose badly under a hard wallclock cap; this candidate avoids extra retained depth.

At review time there was no existing `candidates/` directory, so there were no prior candidate iterations to compare against.

## External research that informed it

- Zechun Liu et al., **“MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases”** (arXiv:2402.14905): immediate block-wise weight sharing is an interesting small-model result, but I did **not** choose it for this iteration because this repo already has a negative recurrence result under a hard wallclock cap and a fair test here would require a wider width/retuning refactor, not just toggling sharing on.
- Fabian Gloeckle et al., **“Better & Faster Large Language Models via Multi-Token Prediction”** (arXiv:2404.19737): argues that predicting multiple future tokens with independent auxiliary heads improves sample efficiency and can improve algorithmic/induction behavior.
- DeepSeek-AI, **“DeepSeek-V3 Technical Report”** (arXiv:2412.19437): explicitly calls out a multi-token prediction training objective as part of a high-performing modern LM recipe.

These papers are not direct proofs for a 16 MB challenge model, but they make MTP one of the better-supported low-infrastructure bets that has not yet been tried in this repo's public record stack.

## What changed versus the chosen base implementation

Base code: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate changes:

1. Set `MTP_NUM_HEADS=2` by default.
2. Set `MTP_LOSS_WEIGHT=0.15` by default.
3. Wire `mtp_heads.*` into the AdamW small-parameter optimizer so the auxiliary objective is actually trained.
4. Keep the base script's export behavior that removes `mtp_heads.*` before serialization and quantized roundtrip eval, so the auxiliary heads remain training-only.

Everything else stays on the 2026-03-23 stack: LeakyReLU(0.5)^2 MLPs, 11 layers, BigramHash, XSA on late layers, partial RoPE, value embeddings, EMA/SWA machinery, Parallel Muon, and the existing quantized export path.

## How to run or evaluate

From this directory:

```bash
RUN_ID=mtp_aux_heads \
SEED=1337 \
MTP_NUM_HEADS=2 \
MTP_LOSS_WEIGHT=0.15 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The candidate resolves its default dataset and tokenizer paths from the script location, so it can be launched directly from this candidate folder without overriding `DATA_PATH` or `TOKENIZER_PATH`.
- `TTT_ENABLED=0` keeps the experiment focused on the new training objective first.
- After a clean pre-TTT comparison, you can re-run with the same stack plus `TTT_ENABLED=1` if you want to measure interaction with the current best evaluation recipe.

## Validation

Commands run in this repo:

```bash
python -m compileall candidates/202604040851_mtp-aux-heads/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202604040851_mtp-aux-heads/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_mtp_aux', path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
PY
```

Outcomes:

- `compileall`: passed.
- Repo-root path resolution: implemented in the script, but start-up validation against real files was not possible here because this runner does not have `data/datasets/fineweb10B_sp1024/` or `data/tokenizers/fineweb_1024_bpe.model` downloaded.
- Runtime smoke / 600-second budget validation: not completed in this runner because the environment is missing repo dependencies (`numpy` was the first missing package), and the script also targets the CUDA/FlashAttention training environment used by the records.

## Main risks and tradeoffs

- MTP increases training-time compute; if the extra auxiliary heads cut too many optimizer steps, the gain in sample efficiency may not compensate.
- The strongest public evidence for MTP comes from much larger models than this challenge regime, so the benefit may shrink at 16 MB scale.
- `MTP_NUM_HEADS=2` and `MTP_LOSS_WEIGHT=0.15` are principled starting values, not a tuned optimum; the first follow-up sweep should compare `1/2/4` heads and weights around `0.05-0.20`.
