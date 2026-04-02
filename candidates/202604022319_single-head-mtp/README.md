# Candidate: Single-Head MTP on the 1.1194 LeakyReLU² + Legal TTT stack

## Hypothesis

Enable a **single auxiliary multi-token prediction (MTP) head** during training on top of the current best 11-layer stack. The auxiliary head should improve sample efficiency inside the fixed 600-second training budget, while the final exported artifact stays flat because the export path already strips `mtp_heads` before quantization.

## Why this is promising here

- The repo's strongest line is already close to the 16 MB cap, so **artifact-neutral** improvements are unusually valuable.
- Recent records already contain a complete MTP implementation, but every inspected run kept `mtp_num_heads:0`, so this is a clean unexplored switch rather than a speculative rewrite.
- The current best record spends the extra evaluation budget on legal TTT, which means a better pre-TTT checkpoint should compound with the existing post-training adaptation path.

## Prior repo evidence that shaped this choice

- **Base implementation:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` is the current strongest record at **1.1194 mean post-TTT bpb** and already packages the mature 11-layer stack with Partial RoPE, LN scaling, VE, GPTQ-lite int6, legal TTT, and Parallel Muon.
- **Best non-TTT checkpoint family:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` shows the strongest pure-model line before TTT.
- **Winning trends across earlier records:** the repo repeatedly benefits from sample-efficiency and evaluation-efficiency improvements that do **not** increase artifact size much: sliding-window eval, EMA/SWA, Partial RoPE, XSA, and tighter quantization.
- **Novelty check:** prior logs and READMEs consistently show `mtp_num_heads:0` or `MTP_NUM_HEADS=0`, so this candidate is the first intentional MTP activation on the mature stack.

## External research that informed it

- **Fabian Gloeckle et al., "Better & Faster Large Language Models via Multi-token Prediction" (arXiv:2404.19737, 2024).** The paper reports better sample efficiency from predicting multiple future tokens with auxiliary heads on a shared trunk, with no inference-time requirement to keep those heads.
- **Team OLMo, "2 OLMo 2 Furious" (arXiv:2501.00656, 2024/2025).** Useful as a broader reminder that recent efficient-language-model recipes keep winning by improving **per-token efficiency**, not only by scaling parameters.
- **George Grigorev, "IMU-1: Sample-Efficient Pre-training of Small Language Models" (arXiv:2602.02522, 2026).** This helped prioritize low-overhead sample-efficiency interventions over artifact-growing changes on a budget-constrained stack, even though this candidate deliberately isolates MTP rather than activating every IMU-style attention tweak at once.

## What changed vs the chosen base

Starting from `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`:

1. **Enabled one MTP head by default** via `MTP_NUM_HEADS=1`.
2. Kept `MTP_LOSS_WEIGHT=0.2`, matching the latent implementation already present in the repo.
3. **Wired `mtp_heads` into the AdamW-optimized small-matrix parameter set**, so the auxiliary head actually learns instead of remaining a zero-initialized loss-only branch.
4. Set the script defaults to the actual 03-23 record-style run where relevant (`BIGRAM_VOCAB_SIZE=1536`, `TTT_ENABLED=1`, `TTT_FREEZE_BLOCKS=0`) so the candidate is runnable as-is.
5. Made dataset/tokenizer defaults **repo-root-relative to the script path**, so the candidate can be launched from inside its own directory without breaking `DATA_PATH` or `TOKENIZER_PATH`.
6. Left the export path intact: `mtp_heads` are still excluded from the exported state dict, so the artifact budget is governed by the same final model as the base stack.

## How to run / evaluate

From the repository root:

```bash
torchrun --standalone --nproc_per_node=8 candidates/202604022319_single-head-mtp/train_gpt.py
```

From the candidate directory itself:

```bash
cd candidates/202604022319_single-head-mtp
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional explicit invocation matching the intended defaults:

```bash
cd candidates/202604022319_single-head-mtp
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.2 \
BIGRAM_VOCAB_SIZE=1536 \
TTT_ENABLED=1 TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

This repository checkout does **not** include the prepared FineWeb shards or SentencePiece model expected by the training script. In a real run, either populate the default repo-root-relative paths or override `DATA_PATH` / `TOKENIZER_PATH`.

## Main risks / tradeoffs

- **Step-time overhead:** even one auxiliary vocabulary head adds compute; if the per-step slowdown is larger than expected, the 600-second budget may lose enough optimizer steps to erase the sample-efficiency gain.
- **Scale mismatch:** the MTP paper is strongest at much larger scales, so a ~22M-parameter model may realize only a small fraction of the benefit.
- **Interaction risk with TTT:** better pretraining should help TTT, but the adaptation dynamics could also change enough that the historical `-0.0025` TTT gain no longer transfers one-for-one.
- **Evaluation cost stays high:** this candidate inherits legal TTT, so total end-to-end evaluation remains much longer than the pure-model 03-22 line.

## Validation

Commands run on this checkout:

- `python -m compileall candidates/202604022319_single-head-mtp/train_gpt.py` -> **passed**
- Candidate code review found that `mtp_heads` were initially missing from all optimizers; this was **fixed** before finalizing the candidate
- `python - <<'PY' ... Path('candidates/202604022319_single-head-mtp/train_gpt.py').resolve().parents[2] ... PY` -> **repo-root path logic resolved correctly** to `/home/runner/work/parameter-golf/parameter-golf`
- Same static path check -> **expected training assets are absent in this checkout**: `data/datasets/fineweb10B_sp1024` and `data/tokenizers/fineweb_1024_bpe.model` do not exist
- `python - <<'PY' import importlib.util; ... find_spec('torch'|'numpy'|'sentencepiece') ... PY` -> **runtime dependencies are missing on this runner** (`torch=False`, `numpy=False`, `sentencepiece=False`)

CPU import/constructor smoke was therefore **not feasible here** without first installing the repo's heavyweight Python runtime and separately providing the missing dataset/tokenizer assets; the actual forward path also depends on `flash_attn_interface`, which is a GPU-oriented dependency rather than a cheap CPU smoke path.
