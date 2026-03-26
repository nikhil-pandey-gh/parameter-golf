# 202603261220_mtp-tailoff

## Hypothesis

Enable the dormant multi-token prediction (MTP) path in the strongest current trainer, but only as a **training-only auxiliary loss** with a **warmdown tail-off**. The extra future-token target should improve sample efficiency early in the 600-second training budget, while fading the auxiliary loss near the end should preserve late-stage next-token calibration, quantization, and legal TTT behavior.

## Why this is promising for this repository

- The strongest public stack in this repo is the 2026-03-23 `LeakyReLU_LegalTTT_ParallelMuon` record, which already combines LeakyReLU^2, parameter banking, Parallel Muon, GPTQ-lite int6 export, and legal score-first TTT.
- That trainer, plus several earlier 11-layer records, already contains **dormant MTP support** but every saved run still logged `mtp_num_heads:0`, so this idea appears present-but-untried in practice.
- MTP is a particularly good fit for Parameter Golf because the auxiliary heads are stripped from export, so they can improve training without materially increasing artifact bytes.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest reviewed result (`val_bpb: 1.1194`) and direct base implementation.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT GPTQ-lite / EMA branch; confirms the value of tight export-aware optimization.
- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - earliest strong 11L branch carrying MTP plumbing, but with `MTP_NUM_HEADS=0`.
- `records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/`
  - another record family member carrying dormant MTP support, again left disabled.

## External research that informed it

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-Token Prediction"** ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)).
  - Key takeaway used here: predicting multiple future tokens with auxiliary heads on a shared trunk can improve sample efficiency and representation quality without adding inference-time cost.
- Recent March 2026 arXiv search results for multi-token prediction surfaced follow-up directions such as **"Self-Distillation for Multi-Token Prediction"** and **"Thinking into the Future: Latent Lookahead Training for Transformers"**.
  - I did not implement those broader methods directly, but they reinforced that future-looking auxiliary supervision is still an active and promising direction.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Candidate changes:

1. **Enable MTP by default**
   - `MTP_NUM_HEADS` now defaults to `1` instead of `0`.

2. **Use a lighter auxiliary weight**
   - `MTP_LOSS_WEIGHT` now defaults to `0.15`.

3. **Warmdown tail-off for the MTP loss**
   - Added `MTP_TAIL_OFF_SCALE` (default `0.25`).
   - The auxiliary loss keeps full strength while the LR scale is above `0.25`, then linearly decays to zero as the run enters the final warmdown.

4. **Keep export behavior unchanged**
   - MTP heads are still excluded from the exported checkpoint and from post-training evaluation, so the extra supervision is paid only during training.

5. **Make the candidate runnable from its own directory**
   - The default dataset and tokenizer paths are resolved relative to the repository root instead of the current working directory.
   - `MTP_LOSS_WEIGHT=0` now forces `MTP_NUM_HEADS=0`, which restores the no-MTP baseline path cleanly.

## How to run / evaluate

From this directory:

```bash
MTP_NUM_HEADS=1 \
MTP_LOSS_WEIGHT=0.15 \
MTP_TAIL_OFF_SCALE=0.25 \
TTT_ENABLED=1 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- The base script still inherits the 2026-03-23 stack: 11 layers, LeakyReLU^2, Parallel Muon, GPTQ-lite int6 + lzma, and legal score-first TTT when `TTT_ENABLED=1`.
- The candidate intentionally leaves the root `train_gpt.py` untouched.
- Default `DATA_PATH` / `TOKENIZER_PATH` values now resolve back to the repository root, so running from this candidate directory works without extra path overrides.

## Expected risks / tradeoffs

- **Throughput risk:** even one auxiliary future-token head increases training compute, so step count may fall enough to erase sample-efficiency gains.
- **Late-training interference:** if the auxiliary objective stays active for too long, it could hurt the final next-token loss or quantized export quality. The tail-off schedule is meant to reduce that risk.
- **TTT interaction uncertainty:** better pre-TTT representations may help TTT, but the extra auxiliary objective could also change how quickly the model adapts chunk-by-chunk.
- **No prior repo ablation:** the code path exists in earlier records, but I found no saved run where it was actually enabled, so the main uncertainty is empirical rather than implementation-related.

## Validation

Commands run in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603261220_mtp-tailoff/train_gpt.py
python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

Outcomes:

- `compileall` succeeded for the root scripts, `data/`, and this candidate trainer.
- A runtime smoke check was **not feasible in this environment** because the local Python environment does not have `torch` installed (`ModuleNotFoundError`), and the inherited 2026-03-23 trainer also expects CUDA/FlashAttention at execution time.
