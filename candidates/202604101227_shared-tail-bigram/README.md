# Shared Tail MLP + Bigger Bigram + LeakyReLU^2

## Hypothesis

The current best non-TTT stack already looks saturated on quantization and evaluation tweaks, so the next useful lever is **parameter sharing that does not add training-time compute**. This candidate shares only the **late MLP weights** across the final four layers while keeping attention, norms, skip weights, and control tensors layer-specific. The saved artifact budget is reinvested into a **larger bigram hash table**, and the MLP uses **LeakyReLU(0.5)^2** from the latest record.

## Why this is promising here

- Repo evidence says **naive layer recurrence is a bad fit** for the 10-minute wallclock because extra passes cost too many steps.
- Repo evidence also says the current frontier is already strong on **XSA, EMA, partial RoPE, GPTQ-lite, late QAT, and sliding eval**, so another tiny quantization tweak is unlikely to move much.
- Selective late sharing keeps compute roughly unchanged while targeting the part of the model where the repository already spends most of its parameter budget: the 11-layer, 3x-MLP deep stack.

## Prior records that influenced this candidate

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT base stack; this candidate starts from that script.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - supplies the LeakyReLU(0.5)^2 MLP activation.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - shows that larger bigram tables can still help once the artifact budget allows them.
- Negative counterexample: `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - documents that naive layer recurrence hurt badly in a fixed wallclock setting, which is why this candidate uses sharing without extra depth passes.

## External research that informed it

- **ALBERT** (arXiv:1909.11942): cross-layer parameter sharing can preserve quality while reducing parameter count.
- **Universal Transformer** (arXiv:1807.03819): iterative reuse can improve parameter efficiency, but it changes the compute/optimization tradeoff.
- **Loop, Think, & Generalize** (arXiv:2604.07822): recurrent-depth transformers can generalize better when reuse is trained carefully.
- **Thinking Deeper, Not Longer** (arXiv:2603.21676): stable shared-depth systems benefit from identity-biased stabilization rather than naive looping.
- **SCORE** (arXiv:2603.10544): controlled recurrent depth can improve convergence, reinforcing the idea that reuse should be structured.
- **Parameter-Efficient Quality Estimation via Frozen Recursive Models** (arXiv:2603.14593): recursion alone is not a universal win; representation quality and selective sharing matter.
- **Architectural Trade-offs in Small Language Models Under Compute Constraints** (arXiv:2512.20877): small-model improvements are highly budget-sensitive, so changes that preserve throughput are preferable.

## What changed versus the chosen base

Starting point: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate changes:

1. **Late shared MLP bank**
   - final 4 layers use a bank of 2 shared MLP modules, alternating by layer.
   - attention blocks remain unique.
   - norms, residual mixing, attn/mlp scales, skip weights, XSA flags, and VE scales remain unique.
2. **LeakyReLU(0.5)^2 MLP**
   - replaces ReLU^2 with a leaky version in all MLPs.
3. **Bigger bigram hash**
   - `BIGRAM_VOCAB_SIZE` default increases from `2048` to `4096`.
4. **New knobs**
   - `SHARED_MLP_LAST_N` (default `4`)
   - `SHARED_MLP_BANK_SIZE` (default `2`)
   - `MLP_NEGATIVE_SLOPE` (default `0.5`)

## How to run / evaluate

From the candidate directory:

```bash
RUN_ID=shared_tail_bigram \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key candidate-specific overrides:

```bash
BIGRAM_VOCAB_SIZE=4096 \
SHARED_MLP_LAST_N=4 \
SHARED_MLP_BANK_SIZE=2 \
MLP_NEGATIVE_SLOPE=0.5
```

## Validation

Commands run:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604101227_shared-tail-bigram/train_gpt.py
python - <<'PY'
import importlib.util
from pathlib import Path

path = Path("candidates/202604101227_shared-tail-bigram/train_gpt.py")
spec = importlib.util.spec_from_file_location("candidate_train_gpt", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
print("import_ok")
PY
```

Outcomes:

- `compileall`: passed
- CPU import smoke: **not feasible in this runner**
  - this environment is missing repository runtime dependencies from `requirements.txt`
  - the import failed immediately on `ModuleNotFoundError: No module named 'numpy'`
  - because the script also depends on `torch` and `flash_attn_interface`, a real CPU forward-pass smoke test was not possible here without installing the full training stack

## Main risks / tradeoffs

- Sharing late MLPs may underfit if those layers need more specialization than the shared bank allows.
- The larger bigram table adds parameters back, so the net artifact win depends on the shared-tail savings compressing as expected.
- LeakyReLU^2 is known-good in the latest record, but the interaction with shared late MLPs is still uncertain.
