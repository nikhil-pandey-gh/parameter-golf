# QERA-lite Late-Layer Residual Rescue

## Hypothesis

The strongest non-TTT 11-layer line in this repo already trains well, but it still gives back a noticeable chunk of quality during fp32 -> int6 export. This candidate keeps the proven 11L/XSA4/partial-RoPE/LN-scale/EMA/GPTQ-lite recipe, then adds a tiny **low-rank residual correction** on top of late-layer int6 weights so the exported model can recover part of that quantization loss without changing training-time compute.

## Why this is promising for this repo

- Recent record progress here has come mostly from **better quantization/export behavior**, not from replacing the core architecture.
- The `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` run reports `post_ema val_bpb:1.1385` and `final_int6_roundtrip_exact val_bpb:1.14656250`, so the mature 11-layer stack still leaves roughly **0.008 BPB** on the table before sliding-window eval.
- Unlike naive recurrence, this idea does **not** reduce optimization steps or change effective depth. It attacks the current bottleneck directly: the exported artifact.
- The base submission had about **450 KB** of artifact headroom, enough to try small fp16 residual factors on only the most sensitive late blocks.

## Prior repo experiments that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - carries the 11-layer U-Net stack, XSA on the last 4 layers, partial RoPE, LN scale, VE128, EMA, and GPTQ-lite percentile search.
- **Activation change borrowed from:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - swaps ReLU^2 for **LeakyReLU(0.5)^2**, which was the cleanest single-line gain in the current best line.
- **General trend from earlier records:** int6/int5 mixed export, EMA/SWA, weight decay, and embedding-sensitive precision choices repeatedly improved results.
- **Prior candidates:** there was **no `candidates/` directory** before this run.

## External research that informed it

- **LoftQ** argues that low-rank terms can absorb quantization error that plain low-bit weights cannot: <https://arxiv.org/abs/2310.08659>
- **LQ-LoRA** explicitly uses a **quantized matrix + high-precision low-rank residual** decomposition, including for compression-oriented settings: <https://arxiv.org/abs/2311.12023>
- **QERA** formalizes quantization-error reconstruction and shows that low-rank error terms help both fine-tuning and post-training quantization: <https://arxiv.org/abs/2410.06040>
- **Scaling Law for Quantization-Aware Training** highlights that weight quantization error remains a real bottleneck, especially around FC2/MLP-style layers: <https://arxiv.org/abs/2505.14302>

This candidate implements the simplest repo-friendly version of that line of work: **weight-space SVD residuals only on the last few int6 layers**, with no new training loop or calibration infrastructure.

## What changed versus the chosen base

1. **LeakyReLU(0.5)^2 MLP**
   - `torch.relu(x).square()` -> `F.leaky_relu(x, negative_slope=0.5).square()`
2. **QERA-lite export knobs**
   - added `QERA_RANK` (default `2`)
   - added `QERA_TOP_LAYERS` (default `3`)
3. **Late-layer low-rank reconstruction**
   - after GPTQ-lite-style per-row int6 quantization, the script computes the residual
     `W_fp32 - W_int6`
   - for matrices in the last `QERA_TOP_LAYERS` transformer blocks, it stores a rank-`QERA_RANK` fp16 SVD approximation of that residual
   - on load, dequantization reconstructs
     `W ~= W_int6 + U @ V`

## How to run / evaluate

Defaults in `train_gpt.py` are already set for this candidate. A standard 8-GPU run is:

```bash
SEED=1337 \
QERA_RANK=2 \
QERA_TOP_LAYERS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The export path will log:

- the usual post-EMA and int6 roundtrip metrics,
- `qera_rank` / `qera_top_layers`,
- and `qera_residual_mats`, the number of matrices that received residual rescue terms.

## Main risks / tradeoffs

- **Artifact bytes:** the residual factors are small, but they are still extra fp16 payload. If they do not compress as well as expected, `QERA_TOP_LAYERS=2` or `QERA_RANK=1` may be necessary.
- **Export time:** SVD is only done at export, not during training, but it is still extra CPU work.
- **Approximation quality:** this is a deliberately simple **weight-space** residual method, not a full activation-aware QERA implementation. The gain may be smaller than the paper line suggests.
- **Most likely sensitive tensors:** if the improvement is concentrated in MLP down/FC2-style matrices, a later ablation may want to rescue only those tensors instead of every late-layer int6 matrix.

## Validation

Executed in this workflow:

1. `python -m compileall candidates/202604042244_qera-lite-residual/train_gpt.py` - **passed**
2. A lightweight import-based smoke check was attempted, but this runner does not have the repo runtime stack installed (`numpy`, `torch`, and `sentencepiece` from `requirements.txt` were all missing), so a meaningful CPU-side start-up check was **not feasible** here.

Because the script also hard-requires CUDA for actual execution, the practical next validation step is a short GPU run on the normal training environment.
