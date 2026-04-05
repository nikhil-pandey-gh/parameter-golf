# LeakyReLU2 + FC2 Sidecar Export

## Hypothesis

The strongest next move is still on the quantization/export frontier, not a larger architectural rewrite. Recent QAT analysis argues that **FC2 outliers are a primary low-bit bottleneck** and that **mixed precision targeted at that bottleneck** closes more of the quantization gap than uniform treatment does. This candidate tests a lightweight version of that idea on the repo's strongest training-only 11-layer stack: keep the model mostly int6, but preserve a tiny fp16 "sidecar" for the most quantization-sensitive rows in the deepest MLP down-projection (`mlp.proj`) matrices.

## Why this is promising here

- The repository's gains have steadily come from **compression-aware improvements**: fp16 tied embeddings, mixed int6/int8 export, GPTQ-lite clip search, and then smaller follow-on wins from EMA / partial RoPE / LeakyReLU2.
- The non-record 4-hour baseline still had a meaningful pre-quant -> post-quant gap, which suggests export quality remains a live bottleneck even after longer training.
- This idea spends bytes only where recent research says the error is concentrated, instead of paying for broader fp16 passthrough or slower whole-model QAT.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Strongest clean training-only stack in the repo: 11L, XSA4, partial RoPE, LN scale, VE, EMA, GPTQ-lite.
- **Activation change:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 was already the cleanest recently demonstrated pre-TTT gain.
- **Quantization sensitivity lessons:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` and `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - Selective precision escape hatches can pay off when the protected tensors are unusually sensitive.

## External research

- **Scaling Law for Quantization-Aware Training** (Chen et al., 2025): <https://arxiv.org/abs/2505.14302>
  - Key takeaway used here: FC2 activation/outlier behavior is a major quantization bottleneck, and mixed precision aimed at that bottleneck is unusually effective.
- **SpinQuant: LLM Quantization with Learned Rotations** (Liu et al., 2025): <https://arxiv.org/abs/2405.16406>
  - Broader motivation: outlier management materially improves post-quantized quality. This candidate takes the lighter-weight path that fits this repo's constraints instead of implementing learned rotations.

## What changed vs. the chosen base

1. **LeakyReLU(0.5)^2 MLP**
   - Replaces relu^2 with the current repo's strongest known activation tweak.
2. **FC2 fp16 sidecar on export**
   - For the deepest `FC2_SIDECAR_LAST_N=3` `mlp.proj.weight` tensors, the exporter measures row-wise int6 reconstruction error and preserves the top `FC2_SIDECAR_ROWS=48` rows in fp16.
   - The full matrix is still stored in the normal int6 path; the fp16 sidecar only overwrites the worst rows during dequantization.
3. **Practical fallback path**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - `ALLOW_CPU=1` permits a local smoke-only CPU path instead of hard-failing on missing CUDA.
4. **Portable export path**
   - The candidate always writes a zlib-compressed artifact so export/load behavior does not depend on optional local packages.
5. **Late QAT disabled by default**
   - This candidate focuses the extra artifact budget on targeted export protection rather than broad late-stage fake quantization.

## How to run

From this directory:

```bash
RUN_ID=leaky_fc2_sidecar \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_MULT=3.0 \
MLP_NEGATIVE_SLOPE=0.5 \
BIGRAM_VOCAB_SIZE=2048 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
VE_ENABLED=1 \
VE_DIM=128 \
VE_LAYERS=9,10 \
EMA_ENABLED=1 \
SWA_ENABLED=1 \
SWA_EVERY=50 \
FC2_SIDECAR_ROWS=48 \
FC2_SIDECAR_LAST_N=3 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 \
MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

- The script still writes:
  - `final_model.pt`
  - `final_model.int6.ptz`
  - roundtrip metrics
  - sliding-window metrics when `EVAL_STRIDE > 0`
- The new export log also reports the number of FC2 sidecar tensors/rows and their raw byte cost.

## Main risks and tradeoffs

- The sidecar rows are selected by **weight reconstruction error**, which may not line up perfectly with final BPB.
- Protecting too many rows can erode artifact headroom; `FC2_SIDECAR_ROWS` is likely the main tuning knob.
- LeakyReLU2 helped the parameter-banked / TTT record, but its interaction with this simpler training stack still needs real GPU validation.
- The CPU fallback is for startup/smoke only, not representative challenge performance.

## Validation in this environment

- `python -m compileall candidates/202604050517_leaky-fc2-sidecar/train_gpt.py` — passed.
- `python -m py_compile candidates/202604050517_leaky-fc2-sidecar/train_gpt.py` — passed.
- Minimal runtime smoke was **not feasible here** because the local Python environment does not currently have the repo runtime dependencies (`torch`, `numpy`, `sentencepiece`) installed, and the workspace also does not contain the required tokenizer / FineWeb shard artifacts for a real run.
