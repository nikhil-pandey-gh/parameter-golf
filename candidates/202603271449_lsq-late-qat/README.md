# LSQ Late QAT on the 2026-03-22 GPTQ-lite Base

## Hypothesis

The strongest non-TTT record in this repository already has a strong 11-layer architecture, but it still pays a measurable gap between post-EMA full-precision quality and the final exported int6 artifact.

This candidate tests whether **export-aware, per-row LSQ late QAT** can close part of that gap more directly than fixed post-training clip search. Instead of using a fresh GPTQ-lite percentile search only at export time, the model learns the same per-row int6 step sizes it will later ship with, but only during the late low-LR part of training.

## Why this is promising here

Repository evidence says the remaining easy wins are mostly in quantization and evaluation, not in adding yet another architecture trick:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the strongest non-TTT base and already includes 11 layers, 3x MLP, XSA4, partial RoPE, LN scale, EMA, GPTQ-lite, and warmdown tuning.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` improves further, but much of the gain comes from TTT and a more complex systems stack.
- The 2026-03-21 partial-RoPE record notes that a late-QAT branch in that lineage was effectively dead due to `torch.compile` constant folding. This candidate explicitly avoids that failure mode by recompiling once when LSQ turns on.

So the bet is: **keep the mature 2026-03-22 model stack, but make the late quantizer trainable and export-consistent**.

## Prior records that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- **Quantization/eval ceiling:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- **Partial RoPE + LN scale evidence:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`

## External research

- **LSQ**: Esser et al., *Learned Step Size Quantization*, arXiv:1902.08153  
  <https://arxiv.org/abs/1902.08153>
- **LSQ+**: Bhalgat et al., *LSQ+: Improving low-bit quantization through learnable offsets and better initialization*, arXiv:2004.09576  
  <https://arxiv.org/abs/2004.09576>
- **AWQ**: Tang et al., *Activation-aware Weight Quantization for LLM Compression and Acceleration*, arXiv:2306.00978  
  <https://arxiv.org/abs/2306.00978>

LSQ/LSQ+ motivate learning quantizer step sizes directly instead of fixing them from row max statistics. AWQ is adjacent evidence that the export path for weight-only quantization is still a strong optimization target for language models.

## What changed versus the base implementation

Starting from the 2026-03-22 non-TTT record, this candidate changes only the export/quantization path and a few usability details:

1. Every `CastedLinear` now has a **training-only per-row LSQ scale parameter**.
2. LSQ is **enabled late** using `LSQ_START_SCALE` and the model is **recompiled once** at activation time so the quantized forward path is not optimized away.
3. The exported int6 artifact uses the **learned LSQ row scales** for attention/MLP-classified weights instead of running a fresh GPTQ-lite percentile search for those tensors.
4. Default `DATA_PATH` and `TOKENIZER_PATH` now resolve **relative to the repository root**, so the script can be run from this candidate directory directly.
5. `flash_attn_interface` is now optional at import time; the script falls back to PyTorch SDPA when FlashAttention 3 is unavailable.
6. A small `SMOKE_TEST=1` path was added to exercise model construction, LSQ setup, quantization, and round-trip loading without the full training loop.

## How to run

From the candidate directory:

```bash
cd candidates/202603271449_lsq-late-qat

RUN_ID=lsq_late_qat \
LSQ_ENABLED=1 \
LSQ_START_SCALE=0.15 \
LSQ_LR=0.002 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script defaults to the same data/tokenizer locations used by the repository, but now resolves them relative to the repo root, so running from inside this directory works without extra path overrides.

To ablate back toward the original export behavior:

```bash
cd candidates/202603271449_lsq-late-qat

LSQ_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To run the lightweight smoke mode once dependencies are installed:

```bash
cd candidates/202603271449_lsq-late-qat
SMOKE_TEST=1 python train_gpt.py
```

## Main risks and tradeoffs

- **Hyperparameter sensitivity:** LSQ scale learning can destabilize late training if `LSQ_LR` is too high.
- **Modest upside:** the current leaderboard gap may now be dominated more by eval/TTT than by export quantization.
- **EMA/export coupling:** the learned scales are training-only, so correctness depends on export using those same learned scales consistently.
- **One-time recompilation cost:** enabling LSQ late forces a single recompile, which adds some complexity and a small runtime interruption.

## Validation run in this environment

Commands executed here:

```bash
python -m compileall candidates/202603271449_lsq-late-qat/train_gpt.py
```

Outcome:

- `compileall` passed successfully.

Attempted smoke test:

```bash
cd candidates/202603271449_lsq-late-qat
SMOKE_TEST=1 python train_gpt.py
```

Outcome:

- Could not complete in this runner because the repository Python dependencies are not installed in the provided environment.
- The failure occurred at import time with `ModuleNotFoundError: No module named 'numpy'`.
- Because bash tool sessions do not have internet/package-install access in this workflow, I could not install the missing dependencies here.

So the candidate is **syntax-validated** and includes a dedicated smoke path, but a live smoke run still needs an environment with the repository dependencies available.
