# Candidate: SigSoftmax head on the 11L GPTQ-lite stack

## Hypothesis

The strongest non-TTT stack in this repo already spends almost all of its byte budget on a well-tuned 11-layer model body, but every record still uses a standard softmax output head over a tied embedding matrix. Record history says that tied embedding/output path is unusually sensitive, so a zero-parameter head change that increases output-distribution expressivity is a strong next bet.

This candidate swaps the base stack's standard softmax loss for **SigSoftmax**, which keeps the same tied embedding matrix and nearly the same compute shape while addressing the classic softmax bottleneck. The goal is to improve validation BPB without paying extra artifact bytes or introducing a new training/export subsystem.

## Why this is promising for this repository

- **The output path is already known to be fragile here.** `2026-03-18_FP16Embed_WD3600` showed that simply keeping `tok_emb.weight` in fp16 sharply reduced the post-quantization gap, which is strong evidence that the tied embedding/head is a bottleneck worth treating carefully.
- **Current wins mostly attack other axes.** The big record jumps came from sliding-window eval, quantization/export, deeper 11-layer stacks, XSA, Partial RoPE, LN scale, EMA/SWA, and legal TTT. None of the records directly attacked the output-head expressivity itself.
- **SigSoftmax is a low-risk fit for the challenge constraints.** It adds no learned parameters, keeps the strong 2026-03-22 architecture intact, and should be close to free in artifact size compared with adding more modules or more low-bit machinery.

## Prior records that influenced this candidate

- **Direct base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Reused as the starting point because it is the strongest clean non-TTT stack with the repo's now-standard 11L/XSA4/Partial-RoPE/LN-scale/GPTQ-lite recipe.
- **Output-path evidence:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/`
  - Motivated this candidate because it showed that the tied embedding/output path is exceptionally quantization-sensitive.
- **Architectural carry-over behind the base:** `2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271`, `2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`, and `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
  - These runs established the winning family: 11 layers, MLP3x, XSA on deep layers, Partial RoPE, LN scale, EMA/Tight-SWA, and later leaky-ReLU/TTT on top.

## External research that informed it

- **Softmax bottleneck:** Zhilin Yang et al., *Breaking the Softmax Bottleneck*  
  https://arxiv.org/abs/1711.03953
- **SigSoftmax:** Sekitoshi Kanai et al., *sigsoftmax: Reanalysis of the Softmax Bottleneck*  
  https://arxiv.org/abs/1805.10829
- **Alternative considered but not chosen:** Ben Peters et al., *Sparse Sequence-to-Sequence Models* (`alpha`-entmax)  
  https://arxiv.org/abs/1905.05702

SigSoftmax was chosen over entmax because it is simpler to drop into the existing cross-entropy path and does not require introducing a different normalization routine or new hyperparameters just to get a first signal.

## What changed versus the chosen base implementation

1. Added `OUTPUT_HEAD` with `sigsoftmax|softmax`, defaulting to `sigsoftmax`.
2. Added a single `_apply_output_head()` helper that:
   - applies the existing logit softcap,
   - converts those scores into SigSoftmax-compatible logits via `z + logsigmoid(z)` when enabled.
3. Routed the following through that helper:
   - the main training loss,
   - the optional MTP auxiliary loss,
   - `forward_logits()` used by sliding-window evaluation.
4. Left the rest of the 2026-03-22 stack unchanged:
   - 11 layers / 512 dim / 8H / 4KV,
   - MLP3x,
   - SmearGate + BigramHash,
   - XSA on the last 4 layers,
   - Partial RoPE + LN scale + VE,
   - EMA + tight SWA,
   - GPTQ-lite mixed int6/int8 export,
   - sliding-window eval.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202604051046_sigsoftmax-head
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate resolves its default `DATA_PATH` and `TOKENIZER_PATH` back to the repository root, so this command works from inside the candidate directory without extra path overrides.

To ablate back to the base behavior without changing code:

```bash
cd candidates/202604051046_sigsoftmax-head
SEED=1337 OUTPUT_HEAD=softmax torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script keeps the base stack's output lines, including the final int6 roundtrip metrics and the stride-64 sliding-window metrics.

## Validation

- `python -m compileall candidates/202604051046_sigsoftmax-head/train_gpt.py` — passed
- `python - <<'PY' ... importlib.util.find_spec(...) ... PY` — reported `torch: missing` and `flash_attn_interface: missing`

A minimal CPU-only launch was **not** feasible in this environment. This candidate is derived from the record CUDA path, and the local environment is missing both `torch` and `flash_attn_interface`, so there was no honest way to run a meaningful softmax-vs-sigsoftmax ablation or startup smoke test here. The immediate next validation step is a paired GPU run against the 2026-03-22 base with `OUTPUT_HEAD=softmax` vs `OUTPUT_HEAD=sigsoftmax`.

## Main risks and tradeoffs

- **Softcap interaction:** SigSoftmax is being applied after the repo's existing logit softcap, and that interaction may damp the theoretical gain from a more expressive head.
- **Quantization may erase some of the benefit:** even though the head change adds no parameters, better pre-quant probabilities do not guarantee better post-export BPB.
- **This is intentionally narrow.** If the result is flat, the next experiment should probably combine this head change with one of the repo's other cheap, high-signal knobs such as LeakyReLU^2 or the dormant depth/value-path controls, rather than adding more quantization complexity first.
