# Candidate: LeakyReLU^2 + targeted FC2 int8 late QAT

## Hypothesis

On top of the `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` frontier stack, a **real** late-stage QAT path that targets the most quantization-sensitive FFN down-projection (`FC2`) layers should reduce the post-quantization gap more than it hurts training speed.

This candidate makes that bet in a deliberately conservative way:

- switch the MLP activation from `ReLU^2` to `LeakyReLU(0.5)^2`, borrowing the cleanest idea from the newer `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` record,
- keep the final `FC2_INT8_LAST_K=2` MLP down-projection layers at int8 during export,
- and apply late fake-quantization only to those same layers so training sees the export precision where it matters most.

## Why this is promising for this repository

Repository evidence says the current frontier is already dominated by compression-aware design, not just raw pre-quant loss:

- `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` explicitly documents that its intended late QAT never activated because `torch.compile` constant-folded the class flag.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` improved further with better post-training quantization, suggesting the remaining gap is still export-related.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` shows `LeakyReLU(0.5)^2` is a credible low-cost MLP improvement, but that activation has not yet been tried on the 2026-03-22 GPTQ-lite frontier stack.

So this candidate targets the exact seam that still looks under-explored here: **a genuinely active, export-aware late QAT path on the GPTQ-lite frontier, plus the strongest simple activation improvement from the next record.**

## Prior records and candidates that influenced this candidate

There were no prior `candidates/` in the repo when this candidate was created, so the design is informed by `records/` only.

Primary influences:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the base implementation because it is the cleanest non-TTT GPTQ-lite frontier script.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - contributed the `LeakyReLU(0.5)^2` MLP activation.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - provided the cautionary note that the old late-QAT flag was effectively dead under `torch.compile`.

## External research that informed it

- **arXiv:2505.14302** — *A unified scaling law for QAT that models quantization error as a function of model size, training data volume, and quantization group size*.
  - The paper identifies the transformer **FC2 layer / outlier-driven quantization error** as a major bottleneck and shows mixed precision can reduce that bottleneck.
- **arXiv:2306.01076** — *Quantization-aware tensor-compressed training for transformers*.
  - Reinforces the general idea that transformer compression works better when training is aware of the final low-precision target rather than relying only on post-training quantization.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **LeakyReLU^2 MLP**
   - `torch.relu(self.fc(x)).square()` -> `F.leaky_relu(self.fc(x), negative_slope=0.5).square()`

2. **Targeted late QAT that survives `torch.compile`**
   - replaces the old global `_qat_enabled` switch with per-module `qat_mix` / `qat_qmax` buffers,
   - targets only the final `FC2_INT8_LAST_K` `blocks.*.mlp.proj` layers,
   - reuses the same clipped int8 quantizer as export for those targeted FC2 layers,
   - synchronizes the late-QAT enable decision across ranks before mutating the targeted modules,
   - keeps the late-QAT trigger wallclock-based via `LATE_QAT_THRESHOLD`.

3. **Mixed-precision export aligned with that QAT target**
   - the same final `blocks.*.mlp.proj.weight` layers are exported as int8,
   - remaining large attention/MLP matrices still use GPTQ-lite int6.

4. **Run-from-candidate-directory defaults**
   - default `DATA_PATH` and `TOKENIZER_PATH` now resolve from the repository root, so running from this candidate directory works without editing paths.

## How to run or evaluate it

From the candidate directory:

```bash
cd candidates/202603280555_leaky-mixed-fc2-qat
SEED=1337 \
FC2_INT8_LAST_K=2 \
LATE_QAT_THRESHOLD=0.15 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- By default the script resolves `DATA_PATH` to `<repo>/data/datasets/fineweb10B_sp1024` and `TOKENIZER_PATH` to `<repo>/data/tokenizers/fineweb_1024_bpe.model`, so it is intended to be launched from this candidate directory directly.
- The script still expects the same CUDA + FlashAttention environment as the existing frontier records.

## Validation

Commands run during candidate creation:

```bash
python -m compileall candidates/202603280555_leaky-mixed-fc2-qat/train_gpt.py
```

Outcome:

- **Passed**.

Attempted smoke validation:

```bash
PYTHONPATH="/tmp/gh-aw/agent:$PYTHONPATH" python - <<'PY'
# import the candidate module, stub flash_attn_interface, instantiate a tiny GPT,
# run a forward/backward pass, and round-trip mixed_quantize_int6(...)
PY
```

Outcome:

- **Could not run in this workflow environment** because both `python` and `python3` lacked the repository's `torch` dependency (`ModuleNotFoundError: No module named 'torch'`).
- That means this candidate received a syntax-level validation only in this run; a real smoke test should be the next step in a PyTorch-enabled environment.

## Main expected risks and tradeoffs

- **Throughput risk**: even targeted fake-quantization adds some per-step overhead to the last two FC2 layers.
- **Artifact-size risk**: moving two late FC2 layers from int6-ish export to int8 may reduce compression headroom even if it improves fidelity.
- **Selectivity risk**: this candidate deliberately targets only late FC2 layers; if the quantization bottleneck is broader, the gain may be too small.
- **Interaction risk**: `LeakyReLU(0.5)^2` helped in the newer record, but it has not yet been combined with this exact GPTQ-lite + EMA stack.

## Suggested next experiments if this helps

1. Sweep `FC2_INT8_LAST_K` over `1, 2, 3, 4`.
2. If the artifact budget stays safe, extend targeted late QAT from late FC2 only to late attention output projections.
3. If throughput remains acceptable, test a broader targeted set such as late `mlp.fc` + `mlp.proj` instead of only `mlp.proj`.
