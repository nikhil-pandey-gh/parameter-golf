# Candidate: Modern 11L Mixed-Int5 MLP Stack

## Hypothesis

The strongest TTT-free 11-layer stack in this repo is probably leaving artifact budget on the table by quantizing its MLP path too conservatively.

This candidate tests whether the mature `11L + XSA4 + EMA + partial RoPE + VE + GPTQ-lite` recipe can tolerate **int5 MLP weights while keeping attention/value-sensitive paths at int6**, especially once paired with the already-proven **LeakyReLU(0.5)^2** activation from the current best record.

## Why this is promising for this repository

Three repo trends point in the same direction:

- The modern 11-layer family around the 2026-03-22 record is the strongest **TTT-free** base in `records/`.
- The 2026-03-20 `10L_Int5MLP_MuonWD04_SWA50` run showed that **int5 MLP + int6 attention** can buy meaningful artifact savings without collapsing quality.
- The 2026-03-23 leaderboard leader showed that **LeakyReLU(0.5)^2** is a cheap, real improvement on top of an already strong stack.

The core bet is that the 11-layer family is now stable enough that more aggressive MLP compression is worth retrying there, instead of only on the older 10-layer line.

## Prior records and candidates that informed this

There were no prior folders under `candidates/` when this candidate was created.

The main repo influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - best stable non-TTT base for this candidate
  - provides the 11-layer XSA/EMA/partial-RoPE/VE/GPTQ-lite stack
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - establishes that **int5 MLP / int6 attention** is viable in this repo
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - provides the LeakyReLU(0.5)^2 activation win
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful as a negative result: naive layer recurrence was net harmful under a fixed wallclock budget

## External research that informed it

This candidate is motivated by the common observation in recent quantization papers that **outlier-heavy channels dominate low-bit error**, so mixed-precision or outlier-aware handling should be targeted rather than uniform.

- **SmoothQuant** (Xiao et al., 2022): <https://arxiv.org/abs/2211.10438>
  - argues that quantization difficulty is unevenly distributed and can be shifted or rebalanced across components
- **RPTQ** (Yuan et al., 2023): <https://arxiv.org/abs/2304.01089>
  - highlights per-channel range differences as a main obstacle for low-bit quantization
- **QuaRot** (Croci et al., 2024): <https://arxiv.org/abs/2404.00456>
  - reinforces the idea that reducing outlier sensitivity is what makes more aggressive low-bit quantization workable

This candidate does **not** implement those methods directly. Instead, it adapts their shared lesson to the existing codebase with a minimal change: keep the more fragile attention path at int6, but push the MLP path to int5 using the repo's existing row-wise GPTQ-lite export machinery.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in `candidates/202603291511_modern-int5-mlp/train_gpt.py`:

- **LeakyReLU(0.5)^2 MLP** by default instead of ReLU^2.
- **Mixed GPTQ-lite export**:
  - MLP matrices default to `[-16, 15]` (int5-style)
  - attention-style matrices default to `[-31, 31]` (int6-style)
  - both paths still use the same row-wise percentile search idea from GPTQ-lite
- **Late QAT is disabled by default** (`LATE_QAT_THRESHOLD=0`) because the earlier record family documented a `torch.compile` constant-folding issue that made the class-attribute toggle ineffective.
- **FlashAttention fallback** to PyTorch SDPA when `flash_attn_interface` is unavailable.
- **`SMOKE_TEST_ONLY=1` path** for a tiny CPU-only forward + quantize/dequantize roundtrip validation.
- **Repo-root-relative default data/tokenizer paths**, so the script works when launched from inside the candidate directory as requested.

## How to run or evaluate it

Full train/eval run from the candidate directory:

```bash
cd candidates/202603291511_modern-int5-mlp
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

From the repository root, equivalent:

```bash
SEED=1337 torchrun --standalone --nproc_per_node=8 candidates/202603291511_modern-int5-mlp/train_gpt.py
```

Useful knobs:

```bash
MLP_NEGATIVE_SLOPE=0.5 \
MLP_QUANT_MIN=-16 MLP_QUANT_MAX=15 \
ATTN_QUANT_MIN=-31 ATTN_QUANT_MAX=31 \
GPTQ_CANDIDATE_PCTS=0.9990,0.9995,0.9999,0.99999,1.0 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 candidates/202603291511_modern-int5-mlp/train_gpt.py
```

Cheap CPU smoke test:

```bash
SMOKE_TEST_ONLY=1 python candidates/202603291511_modern-int5-mlp/train_gpt.py
```

## Validation run for this candidate

Validation was done inside a temporary virtualenv because the runner's system Python is externally managed.

Commands run:

```bash
source /tmp/gh-aw/agent/pg-venv/bin/activate
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603291511_modern-int5-mlp/train_gpt.py
SMOKE_TEST_ONLY=1 python candidates/202603291511_modern-int5-mlp/train_gpt.py
```

Outcomes:

- baseline repo syntax check: **passed**
- candidate syntax check: **passed**
- candidate CPU smoke test: **passed**
- CUDA main-path smoke test: **not feasible in this runner** (`torch.cuda.is_available() == False`)
- `flash_attn_interface` availability check: **absent in this runner**, so only the CPU-side SDPA fallback path was exercised here
- smoke output:

```text
smoke_test:ok loss:6.9493 roundtrip_loss:6.9447 logits_shape:(2, 32, 1024) roundtrip_logits_shape:(2, 32, 1024) flash_attn_available:False
```

## Main expected risks or tradeoffs

- **MLP int5 may be too aggressive** on the 11-layer stack, even if it worked on the older 10-layer family.
- **LeakyReLU^2 changes activation statistics**, so the better low-bit behavior may depend on retuning learning rates or warmdown.
- **The export path is better-tested than the train path** for this exact combination; the CPU smoke test only proves that the script starts, builds the model, and survives mixed-precision roundtrip.
- **The full CUDA `main()` path was not exercised here** because this runner does not expose a CUDA device. The script now has a FlashAttention fallback, but that specific GPU fallback path still needs a real CUDA smoke run.
- **This candidate intentionally avoids full legal TTT**. That keeps the implementation compact, but also means it is competing with the leaderboard's best non-TTT stack, not necessarily the overall best score.
