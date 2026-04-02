# AWQ-lite + LeakyReLU^2 on the 11L GPTQ-lite stack

## Hypothesis

The strongest remaining bottleneck in this repository is still the **post-training quantization gap**. The 2026-03-22 non-TTT record already showed that better clip search in the int6 exporter helps, while the 2026-03-23 record showed that **LeakyReLU(0.5)^2** improves the trained model itself. This candidate combines those two observations:

1. keep the strong 11-layer EMA + XSA + Partial-RoPE + VE stack from the 2026-03-22 GPTQ-lite record,
2. carry over LeakyReLU(0.5)^2 from the 2026-03-23 top record,
3. replace plain weight-MSE GPTQ-lite clip selection with an **activation-aware percentile search** that uses a few post-EMA calibration passes to weight quantization error by observed channel importance.

The expectation is that this should reduce the int6 roundtrip loss more directly than more training-side complexity, while staying cheap enough for the 10-minute track.

## Why this is promising for this repository

- The unlimited-compute baseline still had a large pre-quant -> post-quant regression, which is evidence that better training alone does not fully solve the 16MB artifact problem.
- The best non-TTT record here already won by improving the exporter (`GPTQ-lite`) rather than by changing the core 11-layer architecture.
- The latest top record showed that LeakyReLU(0.5)^2 is a real gain on this challenge family, but that activation change has not yet been paired with an activation-aware int6 export path.
- This candidate is a small, local code change inside `train_gpt.py`, so it fits the repository's preference for self-contained experiments instead of new infrastructure.

## Prior records that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest pre-TTT 11-layer stack in this repo, and already exporter-focused.
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - showed LeakyReLU(0.5)^2 is a meaningful improvement, not just noise.
- **Quantization bottleneck evidence:** `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/`
  - much better pre-quant model quality, but post-quant compression still gave back a lot of the gain.
- **Late-QAT caution:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - explicitly documents that one late-QAT path was dead-code-eliminated under `torch.compile`, so this candidate focuses on PTQ by default instead of leaning on that path.

There were **no prior `candidates/` directories** in this repository when this candidate was created.

## External research that informed it

- **GPTQ** — Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* ([arXiv:2210.17323](https://arxiv.org/abs/2210.17323))
  - motivates data-aware weight quantization as a strong post-training lever.
- **AWQ** — Lin et al., *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978))
  - argues that activation statistics, not just weight magnitude, identify which channels matter most under low-bit quantization.
- **SmoothQuant** — Xiao et al., *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* ([arXiv:2211.10438](https://arxiv.org/abs/2211.10438))
  - reinforces the same theme: activation outliers dominate quantization difficulty and can be handled with offline statistics.
- **PTQ combinations** — Sharify et al., *Post Training Quantization of Large Language Models with Microscaling Formats* ([arXiv:2405.07135](https://arxiv.org/abs/2405.07135))
  - explicitly studies combining AWQ, SmoothQuant, and GPTQ-style ideas instead of treating them as mutually exclusive.

This candidate does **not** implement full AWQ/SmoothQuant rescaling or learned rotations. It implements the smallest repo-friendly version of that idea: use observed activation energy to choose the int6 clipping percentile per matrix.

## What changed vs the chosen base implementation

Relative to `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`:

1. **LeakyReLU(0.5)^2 MLP**
   - `torch.relu(x).square()` -> `F.leaky_relu(x, negative_slope=0.5).square()`
2. **Activation-aware calibration pass after EMA**
   - registers `forward_pre_hook`s on each `CastedLinear`
   - collects per-input-channel mean-square statistics over a few post-training forward passes
3. **AWQ-lite GPTQ percentile search**
   - existing code picked the best clip percentile using raw weight reconstruction MSE
   - this candidate weights reconstruction error by the observed activation energy of each input channel
4. **Run-from-directory defaults**
   - `DATA_PATH` and `TOKENIZER_PATH` now default to the repository root `data/` tree, so the script works when launched from this candidate directory
5. **Late QAT disabled by default**
   - `LATE_QAT_THRESHOLD` defaults to `0.0` in this PTQ-focused variant so the candidate isolates the exporter change

## How to run or evaluate

From this candidate directory:

```bash
cd candidates/202604022219_awq-lite-leaky-gptq
RUN_ID=awq_lite_leaky_gptq \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
AWQ_ENABLED=1
AWQ_CALIBRATION_STEPS=4
MLP_NEGATIVE_SLOPE=0.5
LATE_QAT_THRESHOLD=0.0
```

If your dataset/tokenizer live somewhere else, override `DATA_PATH` and `TOKENIZER_PATH` explicitly as usual.

## Main expected risks and tradeoffs

- This is **AWQ-lite**, not full AWQ: it uses diagonal channel importance from a few calibration batches, not channel rescaling or Hessian-aware reconstruction.
- Calibration adds a handful of extra forward passes after training.
- The activation-aware objective may favor the most common channels too aggressively and hurt rare-token behavior.
- LeakyReLU(0.5)^2 and activation-aware PTQ may interact nonlinearly; the combination is plausible but not yet validated on 8xH100.
- The obvious follow-up, if this helps, is to port the same exporter idea onto the 2026-03-23 banked + TTT stack.

## Validation

Commands run in this repository:

```bash
python -m compileall candidates/202604022219_awq-lite-leaky-gptq/train_gpt.py
```

Outcome: **passed**

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202604022219_awq-lite-leaky-gptq/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train_gpt', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print('import ok')
PY
```

Outcome: **failed in this environment with `ModuleNotFoundError: No module named 'numpy'`**

A quick dependency probe also showed that `torch`, `sentencepiece`, and `flash_attn_interface` are not installed in this runner.

A real CPU start-up smoke test was therefore **not feasible here without installing missing runtime dependencies**, and the full script also expects the challenge's CUDA/FlashAttention environment anyway.
