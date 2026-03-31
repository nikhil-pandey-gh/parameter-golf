# MTP auxiliary loss on the 11L GPTQ-lite stack

## Hypothesis

A **training-only multi-token prediction (MTP) auxiliary head** can improve sample efficiency under the repository's fixed 10-minute wallclock without increasing exported model bytes, because the auxiliary head is used only during training and is explicitly dropped before quantized export. I pair that with two repo-proven low-cost tweaks — **LeakyReLU(0.5)^2** and a larger **BigramHash(3072)** table — so the candidate starts from a stronger training stack instead of probing MTP in isolation on an outdated baseline.

## Why this is promising for this repository

The historical record stack shows three important trends:

- Cheap evaluation and schedule changes matter, but the strongest recent training runs already converged on a stable 11-layer recipe with XSA, partial RoPE, LN scale, EMA, GPTQ-lite export, and value embeddings.
- The current top record found a meaningful win from the one-line **LeakyReLU(0.5)^2** activation change and from increasing the bigram hash budget.
- The record codebase already contains **MTP plumbing**, but the public record runs keep `MTP_NUM_HEADS=0`, so this remains a real unexplored gap rather than a repeated leaderboard idea.

That makes MTP a good fit here: it adds **training signal instead of artifact bytes**, which is exactly the sort of leverage this challenge rewards.

## Prior repository influences

This candidate is primarily based on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
  - strong 11-layer training stack with EMA, GPTQ-lite clip search, partial RoPE, LN scale, XSA on the deepest layers, SmearGate, BigramHash, value embeddings, and export-time removal of `mtp_heads`.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - evidence that **LeakyReLU(0.5)^2** was a high-value, low-complexity MLP upgrade and that a larger BigramHash budget can help.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md`
  - confirmation that partial RoPE + LN scale remained strong on the late 11-layer stack.

There were **no prior `candidates/` directories** in this repository, so this is the first candidate iteration in that location.

## External research that informed this candidate

- Fabian Gloeckle et al., **"Better & Faster Large Language Models via Multi-token Prediction"** (2024)
  - <https://arxiv.org/abs/2404.19737>
  - Key relevance: MTP improves sample efficiency by asking the model to predict several future tokens from a shared trunk. Crucially for this repo, the paper explicitly frames MTP as an **auxiliary training task**.
- Guoliang Zhao et al., **"Self-Distillation for Multi-Token Prediction"** (2026)
  - <https://arxiv.org/abs/2603.23911>
  - Key relevance: reinforces that MTP head quality is the limiting factor, suggesting follow-up work should focus on improving the auxiliary heads rather than abandoning MTP entirely.
- Lorenzo Noci et al., **"Thinking into the Future: Latent Lookahead Training for Transformers"** (2026)
  - <https://arxiv.org/abs/2603.20219>
  - Not implemented here, but useful as supporting evidence that **extra predictive foresight during training** is a live research direction with potential benefits for autoregressive models.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four targeted changes:

1. **Enable one MTP auxiliary head by default**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.1`
   - The auxiliary head is still excluded from `export_sd`, so it does not count toward the exported int6 artifact.

2. **Switch the MLP activation to LeakyReLU(0.5)^2**
   - This ports the strongest cheap activation change from the current top record into the simpler 11-layer GPTQ-lite stack.

3. **Increase the default BigramHash budget from 2048 to 3072 buckets**
   - This follows the top record's direction while staying comfortably within the 16MB budget envelope of the late int6/zstd stacks.

4. **Add a FlashAttention fallback path**
   - If `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA. This does not change the intended fast path on challenge hardware, but it makes local import/smoke workflows less brittle and still allows CUDA execution on setups without the FA3 Python binding.

## How to run

From the repository root:

```bash
cd candidates/202603310536_mtp-aux-loss
RUN_ID=mtp_aux_loss SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Because this candidate resolves `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, it can also be launched directly from inside the candidate directory without extra path overrides. Override environment variables as usual if you want to sweep:

```bash
cd candidates/202603310536_mtp-aux-loss
MTP_NUM_HEADS=2 MTP_LOSS_WEIGHT=0.05 BIGRAM_VOCAB_SIZE=4096 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

Evaluation behavior is inherited from the base stack:

- EMA weights are applied before export.
- The final artifact is exported as mixed int6/int8 and compressed with `zstd` if available, otherwise `zlib`.
- Sliding-window evaluation uses `EVAL_STRIDE=64` by default.

A typical leaderboard-style run is therefore just the training command above; the script performs the post-export roundtrip evaluation automatically at the end.

## Validation in this environment

I ran the following lightweight checks in this workflow environment:

```bash
python -m compileall candidates/202603310536_mtp-aux-loss/train_gpt.py
python - <<'PY'
import importlib.util
print(f"torch_installed={int(importlib.util.find_spec('torch') is not None)}")
PY
```

Outcomes:

- `compileall` **passed**.
- A local CPU forward/backward smoke test was **not feasible in this runner** because the workflow image does not have PyTorch installed (`torch_installed=0`).
- I still added a FlashAttention fallback path so that a model import/forward smoke check is possible in environments that do have PyTorch, even if `flash_attn_interface` itself is missing. The end-to-end training script remains **CUDA-only**, matching the base record stack.

## Main risks and tradeoffs

- **Throughput risk:** MTP adds extra output-head work every step. If the step-time hit is larger than the sample-efficiency gain, the candidate could regress.
- **Objective interference:** the auxiliary future-token loss may help early training but slightly misalign the final next-token head if the weight is too high.
- **Byte budget risk:** the bigger BigramHash table and LeakyReLU port should still fit comfortably under the late-stack artifact budget, but this needs confirmation on a real run.
- **Unverified sweep point:** this candidate intentionally picks a conservative default (`1` head, `0.1` weight) rather than claiming those are already optimal.

## Suggested next experiments

If this candidate starts cleanly and looks promising on hardware, the next sweeps I would try are:

- `MTP_LOSS_WEIGHT` in `{0.05, 0.1, 0.2}`
- `MTP_NUM_HEADS` in `{1, 2}`
- late-training disable or annealing of the auxiliary loss
- a follow-up self-distilled MTP variant inspired by <https://arxiv.org/abs/2603.23911>
