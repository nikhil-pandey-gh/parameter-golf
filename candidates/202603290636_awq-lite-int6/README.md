# AWQ-lite Int6 Export on the 11L EMA + GPTQ-lite Stack

## Hypothesis

The strongest *static* models in this repository are already quite good architecturally, but they still lose measurable quality at export time when they are squeezed into the challenge artifact budget. The hypothesis here is that a small **activation-aware, training-free quantization pass** can reduce that remaining post-training quantization gap more efficiently than adding extra train-time overhead.

In practice, this candidate keeps the strong `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` recipe, then adds an **AWQ-lite** step that:

- collects per-input-channel RMS activation statistics from a short pass over training batches,
- searches a small set of activation-aware column scales per large MLP/attention matrix,
- combines that with the repo's existing GPTQ-lite style clip-percentile search, and
- stores the extra AWQ scale vectors in the compressed artifact so dequantization reconstructs the effective scaled weights exactly.

## Why this is promising for this repository

Repository history strongly suggests that **quantization is still one of the main bottlenecks** for tiny models under this 16 MB artifact limit.

Relevant evidence from prior records:

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` showed that keeping the tied embedding in higher precision almost removed the post-quant degradation.
- `records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow/README.md`, `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/README.md`, and `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md` all improved by making the model more export-aware.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` showed that better post-training clipping alone still buys real BPB improvement at zero training cost.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md` also documents that full QAT, late QAT, and depth recurrence were not obviously worth their training-speed cost in the 10-minute budget.

That makes a **training-free, export-time** idea a good fit here.

## Prior runs that influenced this candidate

The direct implementation base is:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Other influential runs:

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - influenced the decision to **disable late QAT by default** here, because that record documents a `torch.compile` constant-folding problem around runtime QAT toggles.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - influenced the decision to keep this candidate focused on the **static model/export path**, rather than mixing in TTT or a near-cap artifact.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - reinforced that an export-side improvement is likely more attractive than adding slower train-time recurrence.

## External research that informed it

This candidate is mainly inspired by **activation-aware post-training quantization** and related equivalent-transform quantization work:

- **AWQ**: Activation-aware Weight Quantization for LLM Compression and Acceleration (`arXiv:2306.00978`)  
  https://arxiv.org/abs/2306.00978
- **SmoothQuant**: offline migration of quantization difficulty between activations and weights (`arXiv:2211.10438`)  
  https://arxiv.org/abs/2211.10438
- **GPTQ**: strong one-shot post-training quantization baseline (`arXiv:2210.17323`)  
  https://arxiv.org/abs/2210.17323
- **QuaRot**: rotation-based outlier removal for easier low-bit quantization (`arXiv:2404.00456`)  
  https://arxiv.org/abs/2404.00456
- **SpinQuant**: learned rotations to improve PTQ quality (`arXiv:2405.16406`)  
  https://arxiv.org/abs/2405.16406

The repo does not currently have the infrastructure for full rotation learning or a full AWQ implementation, so this candidate adapts the most repo-friendly part of that literature: a **small activation-statistics pass plus equivalent per-channel scaling during quantization**.

## What changed versus the chosen base implementation

Starting from the 2026-03-22 11-layer EMA + GPTQ-lite stack, this candidate adds five targeted changes:

1. **AWQ-lite calibration pass on training tokens**
   - A short post-training pass collects per-input-channel RMS activations for large `CastedLinear` MLP and attention matrices.
   - Calibration uses training data, not validation data.

2. **Activation-aware int6 quantization search**
   - For each eligible matrix, the exporter searches `AWQ_ALPHA_CANDIDATES` and the existing GPTQ-lite clip percentiles.
   - The search objective is a simple activation-weighted reconstruction loss, so salient input channels get more protection.

3. **Artifact support for AWQ scale vectors**
   - When AWQ scaling wins for a matrix, the chosen per-input-channel scale vector is stored alongside the int6 payload.
   - Dequantization uses those scales to undo the temporary column scaling and reconstruct the AWQ-aware approximation back in the original weight space.

4. **More robust local validation path**
   - Added a `flash_attn_interface` fallback using `torch.nn.functional.scaled_dot_product_attention` so forward passes can still work without FlashAttention 3.
   - Added `SMOKE_TEST=1` mode to run a tiny synthetic forward + quantize/dequantize check without dataset access.

5. **QAT made opt-in instead of silently dead by default**
   - `LATE_QAT_THRESHOLD` now defaults to `0.0` for this candidate.
   - If someone explicitly enables QAT again, the script avoids compiling the training model so the runtime toggle is not frozen away.

## How to run or evaluate it

From the candidate directory:

```bash
RUN_ID=awq_lite_int6 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=4 \
EMA_ENABLED=1 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
WARMDOWN_ITERS=3500 MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 MUON_MOMENTUM_WARMUP_STEPS=1500 \
MUON_WD=0.04 ADAM_WD=0.04 GRAD_CLIP_NORM=0.3 \
AWQ_ENABLED=1 AWQ_CALIBRATION_BATCHES=8 AWQ_CALIBRATION_TOKENS=65536 AWQ_CALIBRATION_SEQ_LEN=256 \
AWQ_ALPHA_CANDIDATES=0.0,0.25,0.5,0.75,1.0 \
EVAL_STRIDE=64 MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

A tiny local smoke test is available with:

```bash
SMOKE_TEST=1 python train_gpt.py
```

That path does **not** need the dataset, but it still needs the repo's Python dependencies installed.

## Validation run in this workflow

Commands run here:

```bash
python -m compileall candidates/202603290636_awq-lite-int6/train_gpt.py
SMOKE_TEST=1 python candidates/202603290636_awq-lite-int6/train_gpt.py
```

Outcomes:

- `python -m compileall ...` ✅ passed.
- `SMOKE_TEST=1 python ...` ⚠️ could not run in this GitHub Actions runner because the repository runtime dependencies were not installed. The environment was missing at least `numpy`, `sentencepiece`, and `torch` from the existing `requirements.txt`.

## Main expected risks and tradeoffs

- **Artifact headroom:** storing per-matrix AWQ scale vectors costs extra bytes, so this idea is more comfortable on the 2026-03-22 base than on the near-cap 2026-03-23 TTT stack.
- **Export-time overhead:** the AWQ-lite search makes the export path slower because it adds calibration plus per-matrix alpha/clip search.
- **Approximation quality:** this is intentionally lighter-weight than full AWQ, so some layers may not benefit and some matrices may fall back to ordinary GPTQ-lite behavior.
- **No full GPU validation in this environment:** this workflow only verified syntax, not end-to-end training or final BPB.

## Suggested next experiments

If this candidate looks promising on GPU, the next obvious follow-ups are:

1. stack **LeakyReLU(0.5)^2** on top of this static export path,
2. try **fewer AWQ alpha candidates** to reduce export time,
3. try **AWQ only on the most quantization-sensitive late layers**, and
4. combine this export pass with a **working late QAT path** only if the throughput cost stays acceptable.
