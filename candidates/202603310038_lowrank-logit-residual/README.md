# Low-Rank Logit Residual

## Hypothesis

Recent record progression suggests this repo's strongest stack is already close to saturated on generic architecture tweaks, while the tied embedding / output matrix remains a disproportionate bottleneck for both quality and quantization. The hypothesis here is that keeping the strong tied-embedding baseline but adding a **tiny untied output-only correction head** can recover some output-space flexibility without paying the byte cost of fully de-tying embeddings.

Concretely, this candidate adds a low-rank residual path on top of the final logits:

- base logits from the tied embedding matrix,
- plus a small `model_dim -> rank -> vocab_size` correction,
- with the correction zero-initialized so training starts from the parent record's behavior.

## Why this is promising for this repository

Two repository trends point in the same direction:

1. `2026-03-18_FP16Embed_WD3600` showed that `tok_emb.weight` is unusually sensitive and that protecting it during export gives a much larger gain than many broader model changes.
2. Later winning stacks kept leaning on tied embeddings for artifact efficiency, which means the same matrix still has to serve both input representation and output prediction.

External work adds a concrete explanation for that tension. *Weight Tying Biases Token Embeddings Towards the Output Space* (Gupta et al., 2026, arXiv:2603.26663) argues that tied matrices get dominated by output-side gradients early in training, hurting their usefulness as input embeddings. This candidate keeps the byte-efficient tied matrix, but gives the model a small separate output correction path so it does not need to force the shared table to do all the work.

A related embedding-focused direction is *DEPT: Decoupled Embeddings for Pre-training Language Models* (2025), which reinforces the idea that input/output embedding roles are worth separating even when a full untied matrix would be too expensive for this challenge.

## Records and prior experiments that influenced it

This candidate is deliberately built on the strongest local stack rather than the root baseline alone.

Primary base:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`

Relevant influences:

- `2026-03-18_FP16Embed_WD3600`: established that the embedding/output matrix is the most fragile tensor in the artifact.
- `2026-03-19_WarmdownQuantization`: reinforced that this challenge is often bottlenecked by quantization quality, not just pre-quant loss.
- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`: showed the value of polishing the export/eval pipeline once the 11-layer stack is strong.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`: strongest reviewed end-to-end stack, including parameter banking, XSA, VE, and leaky-ReLU-squared MLPs.

## External research that informed it

- Gupta et al., *Weight Tying Biases Token Embeddings Towards the Output Space* (2026), arXiv:2603.26663.
  - Main takeaway used here: tied embeddings are pulled toward output-space needs, especially early in training.
- Iacob et al., *DEPT: Decoupled Embeddings for Pre-training Language Models* (2025).
  - Main takeaway used here: input/output embedding roles are worth partially separating, even if full de-tying is too expensive.

## What changed versus the chosen base implementation

Starting from `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`, this candidate makes two coupled changes:

- New `LowRankLogitResidual` module with default `LOGIT_RESID_RANK=16`.
- The residual head is applied only at the output logits, after the shared trunk and final norm.
- The correction path is zero-initialized on the vocab projection, so the run begins from the parent stack's tied-logit behavior.
- The residual head is excluded from the inherited late-QAT int6 path and is explicitly routed through the existing per-row int8 export path. This avoids applying the wrong fake-quant regime to the new head, though the head still relies on post-training quantization at export time.
- The candidate reduces the default `BIGRAM_VOCAB_SIZE` from the copied script's `2048` to `1536` to preserve artifact headroom for the new output path.
- No root files are touched, and all other inherited behavior (banked weights, XSA, VE, optional legal TTT, export roundtrip checks) remains intact.

## How to run

From this candidate directory:

```bash
SEED=1337 \
BIGRAM_VOCAB_SIZE=1536 \
LOGIT_RESID_RANK=16 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
SEED=1337 \
BIGRAM_VOCAB_SIZE=1536 \
LOGIT_RESID_RANK=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That removes the new head while keeping the candidate's tighter artifact budget.

True parent-stack recovery from the copied Mar 23 script:

```bash
SEED=1337 \
BIGRAM_VOCAB_SIZE=2048 \
LOGIT_RESID_RANK=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script preserves the inherited export + roundtrip flow:

- train on the default 600s wallclock cap,
- export the mixed-precision artifact,
- re-load it,
- run the standard roundtrip validation,
- run sliding-window evaluation,
- optionally run legal score-first TTT if `TTT_ENABLED=1` is explicitly set.

For clean attribution of the new idea, the intended first comparison is with `TTT_ENABLED=0`.

## Expected risks and tradeoffs

- The extra head adds a small amount of training-time compute, though far less than de-tying the full embedding.
- If the residual path learns too quickly, it may overfit output correction instead of improving the shared representation; a rank or LR sweep may be needed.
- Because the head is output-only, it specifically targets logit quality; if the true bottleneck is input-side embedding quality alone, the gain may be limited.
- The best rank is not obvious: too small may underfit, too large may waste bytes or destabilize training.

## Validation

Planned lightweight validation for this environment:

- `python -m compileall candidates/202603310038_lowrank-logit-residual/train_gpt.py`
- minimal CPU smoke test **if feasible**

Results will be updated below after running them.

- `compileall`: `python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603310038_lowrank-logit-residual/train_gpt.py` completed successfully in the repo root.
- CPU smoke test: not feasible in this runner because Python is present but `torch` is not installed (`importlib.util.find_spec("torch") -> None`), so executing the training script would fail before model startup.
