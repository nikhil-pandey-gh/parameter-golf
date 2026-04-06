# Progressive Int6 Quantization Regularization

## Hypothesis

The strongest remaining lever in this repo is still **post-training quantization damage**. This candidate starts from the best clean pre-TTT 11-layer stack and replaces the fragile late-QAT toggle with a **progressive int6 proximity regularizer**: once warmdown meaningfully starts, training adds a small loss that pulls the exported matrix weights toward their row-wise int6 reconstructions. The goal is to land in a part of parameter space that is already export-friendly before the final GPTQ-lite search and compressed packing step.

## Why this is promising here

- Multiple records show that the largest gains came from making compression less destructive rather than only making the dense model better.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600` showed that reducing quantization sensitivity on the tied embedding materially shrank the roundtrip gap.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that even zero-training-cost export improvements like GPTQ-lite clip search still buy measurable BPB.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` documented that the repo's earlier late-QAT path could be compiled away when toggled too late, which makes an **out-of-graph regularizer** especially attractive.

## Influential prior experiments

1. `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
   - Chosen base implementation.
   - Supplies the 11-layer 512d / MLP3x / BigramHash / XSA4 / EMA / GPTQ-lite / warmdown3500 stack.
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
   - Contributed the `LeakyReLU(0.5)^2` MLP activation change, which was already ablated as a meaningful pre-TTT win.
3. `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090`
   - Useful negative evidence: naive layer recurrence was bad there, so this candidate stays focused on compression-aware training instead of recurrent depth reuse.

No prior `candidates/` directory existed in this checkout, so there was no earlier candidate lineage to extend directly.

## External research that informed this candidate

1. **Esser et al., "Learned Step Size Quantization" (arXiv:1902.08153)**  
   Low-bit models improve when quantization is treated as part of training instead of a purely post-hoc export step. This candidate does not implement full LSQ, but it borrows the same core idea: make the optimizer see the low-bit target during training rather than only at the end.
2. **Agustsson et al., "Soft-to-Hard Vector Quantization" (arXiv:1704.00648)**  
   Compression improves when training gradually moves parameters toward discrete codes. This candidate uses a simplified version of that idea: a ramped MSE penalty between float weights and their row-wise int6 reconstructions during warmdown.
3. **Lan et al., "ALBERT" (arXiv:1909.11942)**  
   Parameter sharing remained an attractive fallback idea during research, but local repo evidence for naive recurrence was weak, so I used ALBERT mostly as a contrast case while choosing a lower-risk compression-first candidate.

## What changed vs. the base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **Progressive int6 quantization regularizer**
   - New `QREG_*` hyperparameters.
   - Once LR scale drops below `QREG_START_SCALE`, training adds a ramped loss that penalizes the distance between the Muon-managed matrix weights (transformer blocks plus matching projection weights) and their row-wise int6 reconstructions.
   - This is outside the compiled forward graph, so it does not rely on toggling class-level QAT flags after `torch.compile`.
2. **LeakyReLU(0.5)^2 MLP**
   - Imports the later activation win from the 2026-03-23 record.
3. **Late QAT disabled by default**
   - `LATE_QAT_THRESHOLD` now defaults to `0.0` so the new regularizer is the primary compression-aware late-phase mechanism.
4. **FlashAttention import fallback**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA. This is mainly for portability and lightweight smoke importing; intended leaderboard runs should still use the FlashAttention path.

## Run / evaluation

From the repository root:

```bash
cd /path/to/parameter-golf
RUN_ID=progressive_qreg_seed1337 \
SEED=1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
MLP_LEAK=0.5 \
QREG_ENABLED=1 \
QREG_START_SCALE=0.35 \
QREG_MAX_WEIGHT=0.02 \
QREG_CLIP_RANGE=31 \
LATE_QAT_THRESHOLD=0.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The copied base defaults still assume the strong 11-layer stack:

- 11 layers, 512 model dim, 8 heads / 4 KV heads
- 3x MLP
- Bigram hash + SmearGate + XSA on the last 4 layers
- partial RoPE (16 dims), LN scale, VE on layers 9 and 10
- EMA + GPTQ-lite export + zstd when available (otherwise zlib fallback) + warmdown 3500
- sliding-window eval at stride 64

## Validation run here

1. `python -m compileall candidates/202604061949_progressive-int6-qreg/train_gpt.py`
   - **Passed**
2. Attempted a minimal CPU import / tiny-model smoke script
   - **Not feasible in this runner as-is** because the workflow environment did not have the repo's Python dependencies installed (`numpy`, `torch`, and `sentencepiece` were all missing), and the full training entrypoint is CUDA-only by design.

## Main expected risks / tradeoffs

- **Extra training overhead:** the regularizer adds one more full pass over matrix weights per step after warmdown begins, so the final BPB depends on whether lower quantization damage outweighs any lost steps.
- **Proxy mismatch:** the regularizer uses a cheap row-wise int6 proxy, while export still uses GPTQ-lite percentile search plus zstd. The training proxy is directionally aligned with export, but not identical.
- **Base dependence:** LeakyReLU(0.5)^2 was positive in the later record stack, but the exact gain on this pre-TTT branch may differ.
- **Validation gap in this workflow:** only syntax-level validation completed locally; meaningful quality validation still needs an actual repo-style CUDA run.
