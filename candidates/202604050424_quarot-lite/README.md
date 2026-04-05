# QuaRot-lite export path on the 11L LeakyReLU^2 stack

## Hypothesis

The current repo is already close to the 16 MB cap, so another small architecture change is less attractive than a cleaner **post-training quantization** path. This candidate adds a **fixed Walsh-Hadamard rotation** to the attention and MLP inputs **only for export/eval**, with the goal of reducing int6 quantization error without changing the trained function in full precision.

## Why this is promising here

- The best recent records are already dominated by tiny post-training gains: GPTQ-lite clip search, EMA, later warmdown, and legal evaluation tricks all win in the low-thousandths BPB range.
- The 4-hour non-record run still had a large pre-quant -> post-quant gap, which is strong evidence that storage quantization is still a real bottleneck in this repository.
- This adaptation leaves the **10-minute training path unchanged** and only changes the export/eval path, so it fits the challenge much better than heavier architectural ideas.

## Prior repo work that influenced this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`:
  strongest pre-TTT stack in the repo, plus the parameter-banked training path that this candidate forks.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`:
  showed that tiny quantization/export improvements still move BPB.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`:
  confirms the modern 11-layer / partial-RoPE / LN-scale stack is the right base.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`:
  useful evidence that quantization remains a limiting factor even after much longer training.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`:
  negative recurrence result pushed this candidate away from depth-reuse ideas and toward export-side improvements.

## External research

- **QuaRot** (Croci et al., 2024): <https://arxiv.org/abs/2404.00456>  
  Key motivation: fixed orthogonal rotations can remove outliers and the paper explicitly reports near-lossless **6/8-bit** quantization without calibration.
- **SpinQuant** (Liu et al., 2025): <https://arxiv.org/abs/2405.16406>  
  Motivates the broader direction: rotation choice matters a lot for PTQ, even if this candidate keeps the first implementation fixed and minimal.
- **AWQ** (Tang et al., 2024): <https://arxiv.org/abs/2306.00978>  
  Strong alternative export-side idea, but it needs activation-aware calibration logic. QuaRot-lite was chosen first because it is simpler to adapt to this repo's current code.
- **ALBERT** (Lan et al., 2020): <https://arxiv.org/abs/1909.11942> and **SCORE** (Godin et al., 2026): <https://arxiv.org/abs/2603.10544>  
  Relevant parameter-sharing / recurrent-depth references that were considered, but repo evidence currently points away from recurrence under the fixed wallclock budget.

## What changed vs the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added a normalized fast Walsh-Hadamard transform helper for power-of-two chunk sizes.
2. Added a **QuaRot-lite export path** that rotates the columns of:
   - `attn.c_q.weight`
   - `attn.c_k.weight`
   - `attn.c_v.weight`
   - `attn.proj.weight`
   - `mlp.fc.weight`
   - `mlp.proj.weight`
3. Added matching runtime input rotations in attention and MLP **only in the quantized eval model**.
4. Left the full-precision training model unchanged by constructing it with `quarot_enabled=False`.
5. Disabled `torch.compile` on the rotated eval path to keep the first version conservative and easier to reason about.

This is intentionally **QuaRot-lite**, not a full QuaRot or SpinQuant port: no learned rotations, no sign search, no activation calibration, and no training-time adaptation yet.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
QUAROT_ENABLED=1 QUAROT_CHUNK=512 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This keeps the 3/23 training stack, but swaps the export/eval path to the fixed-Hadamard variant.

If the training-only metric looks promising, the existing legal TTT flags from the 3/23 stack are the first follow-up to try stacking on top.

## Expected risks and tradeoffs

- The gains may be modest at **int6** compared with the bigger W4A4 gains reported in the rotation papers.
- This version uses a **fixed** rotation, so it may underperform learned or searched rotations.
- Evaluation is a bit slower because the rotated roundtrip path skips `torch.compile`.
- Rotation can improve quantization error while still hurting **lzma compressibility**, so artifact bytes must be watched together with BPB.
- The current implementation is tuned for the repo's dominant `model_dim=512`, `mlp_mult=3.0` setting. Other shapes still run, but the rotation path only activates when the input dimension is divisible by a **power-of-two** `QUAROT_CHUNK`.

## Validation

Commands run for this candidate:

```bash
python -m compileall train_gpt.py
python -c "import importlib.util; print('torch_present', bool(importlib.util.find_spec('torch'))); print('flash_attn_interface_present', bool(importlib.util.find_spec('flash_attn_interface')))"
```

Outcomes:

- `compileall` succeeded.
- The local container reports `torch_present False` and `flash_attn_interface_present False`, so a CPU-only smoke run was **not** feasible in this environment.
