# Bank-aware Late QAT Ramp

## Hypothesis

The current best banked model stack is still mostly trained in float and only compressed afterward. In the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` code path, the dominant banked weights bypass `CastedLinear`, so the existing late-QAT toggle does not pressure the tensors that dominate the final int6 artifact. This candidate adds a **bank-aware late-QAT ramp** that captures GPTQ-lite per-row scales once, recompiles once when late QAT begins, and then fake-quantizes the banked core weights during the final training phase.

## Why this is promising here

- Repo evidence says **post-training compression is still a core bottleneck**. The 4-hour non-record run reached much better pre-quant loss than its final artifact quality, and several records explicitly improved by reducing the quantization gap.
- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` documents that its earlier late-QAT path was effectively inactive after `torch.compile` constant-folded the flag.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` showed that **better int6 scale selection alone** is worth about `-0.0006 BPB`.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` moved the model to parameter banks for speed, which makes the old `CastedLinear`-only fake quant even less representative of the final artifact.

The main bet is that training against a closer approximation of the deployed int6 model should improve the non-TTT model and stack cleanly with legal TTT.

## Influential prior records

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Base implementation copied here: LeakyReLU(0.5)^2, parameter banking, parallel Muon, VE, XSA, legal TTT support.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - GPTQ-lite per-row percentile search motivated reusing export-matched scales for the late-QAT phase.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - Shows why a recompile-aware late-QAT design is needed.
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
  - Earlier evidence that QAT can reduce the quantization gap when it actually touches the relevant weights.

## External research

- **Learned Step Size Quantization (LSQ)**, Esser et al., arXiv:1902.08153
  - Main inspiration for training directly against low-bit deployment noise instead of relying only on post-training compression.
- **MobileLLM**, Liu et al., arXiv:2402.14905
  - Reinforces that compact LMs benefit disproportionately from architecture and deployment-aware design decisions.
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints**, Ainslie et al., arXiv:2305.13245
  - Relevant to the repository's compact GQA backbone, which this candidate keeps unchanged.
- **RoFormer / RoPE**, Su et al., arXiv:2104.09864
  - Relevant to the partial-RoPE stack carried forward here.

## What changed versus the chosen base

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added `BANK_QAT=1` support, enabled by default.
2. Added non-persistent buffers for bank QAT scales and a ramp coefficient.
3. When the LR scale crosses `LATE_QAT_THRESHOLD`, the script:
   - computes GPTQ-lite per-row int6 scales for the bank tensors,
   - flips bank-QAT on,
   - recompiles the model once so the new fake-quant path is live under `torch.compile`.
4. During the late phase, the banked `qo`, `kv`, `mlp_up`, and `mlp_down` weights are fake-quantized with a straight-through estimator using the stored per-row scales and a ramped `alpha`.
5. All changes are local to this candidate directory; the repository root and prior records are unchanged.

## How to run

From this candidate directory:

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 BANK_QAT=1 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to isolate the pre-TTT effect, set `TTT_ENABLED=0`.

## Validation

Ran:

```bash
python -m compileall candidates/202603280914_bank-qat-ramp/train_gpt.py
```

Attempted:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("candidates/202603280914_bank-qat-ramp/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("module-import-ok")
PY
```

Outcome:

- Passed syntax compilation.
- The import probe failed immediately with `ModuleNotFoundError: No module named 'numpy'`; the container also lacks other runtime deps such as `torch` and `sentencepiece`.
- A real CPU smoke test was therefore **not feasible** in this environment, and the script also expects CUDA/NCCL-style distributed execution plus FlashAttention 3 at runtime.

## Main risks and tradeoffs

- The one-time late-phase recompile may cost a small amount of wall-clock time.
- The stored per-row scales are captured once at QAT activation, so they can drift slightly as training continues.
- Bank fake-quant in the late phase may reduce the number of optimizer steps completed inside the 600-second budget.
- This candidate assumes the strongest gains still come from shrinking the int6 deployment gap rather than adding another evaluation-only trick.
