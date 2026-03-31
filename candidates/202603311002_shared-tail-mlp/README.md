# Shared Tail MLP Banks + Bigger Bigram Hash

## Hypothesis

The current best 11-layer recipes seem to be bottlenecked more by how efficiently they spend artifact bytes than by a lack of training-time tricks. This candidate tests whether the deepest MLPs are redundant enough to share: keep all attention banks unique, but reuse only the last four MLP banks in an `ABAB` pattern so the model still gets 11 layers of compute while exporting fewer unique MLP weights.

The saved artifact budget is reinvested into a larger default `BigramHash` table (`3072` buckets instead of the parent script's `2048` default), because prior records repeatedly found that extra token-pair capacity gives small but reliable BPB gains.

## Why this looks promising for this repository

Repository review showed a clear pattern:

- the strongest records win by improving **compression-aware capacity allocation** rather than by changing the whole training stack,
- **BigramHash / SmearGate / 3x MLP / 11 layers / XSA / Partial RoPE / EMA / GPTQ-lite** is the current winning cluster,
- no reviewed record or prior candidate used **cross-layer parameter sharing** in the actual submission model,
- a non-record 4-hour baseline still underperformed the modern 10-minute records once export quality lagged, which reinforces that artifact efficiency matters as much as raw optimization.

So this candidate tries a new lever that fits those lessons: save bytes where the model is largest, then spend them where previous records already showed incremental gains.

## Prior records that informed this candidate

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - best current 10-minute stack and direct code base for this candidate.
  - established that the modern frontier is the 11-layer banked model with XSA, Partial RoPE, EMA/SWA, GPTQ-lite export, and optional legal TTT.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - highlighted that better export alone can still buy measurable BPB.
  - reinforced the importance of preserving compression quality while changing the model.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - showed that zero-parameter structural changes on the tail stack can help.
  - also showed that some seemingly helpful training tricks can be fragile under `torch.compile`.

- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - showed that spending saved compression budget on a larger `BigramHash` is worthwhile.

- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  - established the usefulness of bigram-aware front-end features and a wider MLP.

There were no existing `candidates/` folders in the repository at review time, so this is the first candidate iteration in that tree.

## External research that informed it

- **ALBERT** (`arXiv:1909.11942`)
  - classic evidence that cross-layer parameter sharing can preserve quality while reducing parameter count.

- **Universal Transformer** (`arXiv:1807.03819`)
  - motivates reusing a shared transformation multiple times instead of paying for every depth step with unique weights.

- **Thinking Deeper, Not Longer: Depth-Recurrent Transformers** (`arXiv:2603.21676`)
  - argues that shared-weight depth can work if recurrence is stabilized with identity-biased scaling.

- **Ruyi2 Technical Report** (`arXiv:2602.22543`)
  - more recent evidence that family-based parameter sharing can be practical in large-scale LM training.

This candidate intentionally implements the lightest repo-compatible version of those ideas: share only the tail MLP banks, keep the attention path unique, and damp repeated shared uses with a smaller initial residual MLP scale.

## What changed versus the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Tail-only MLP bank sharing**
   - new defaults:
     - `SHARED_MLP_TAIL_LAYERS=4`
     - `SHARED_MLP_TAIL_BANKS=2`
     - `SHARED_MLP_REPEAT_SCALE=0.5`
   - for the default 11-layer model, the logical tail MLP layers map to physical bank indices:
     - `[0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 8]`
   - attention banks remain fully unique; only the largest MLP banks are shared.

2. **Identity-biased repeated passes**
   - when a shared MLP bank is reused later in the tail, that layer's `mlp_scale` is initialized with an extra `0.5x` damping factor.
   - this is a cheap recurrence-stability bias inspired by the depth-recurrent literature.

3. **Export path preserves the savings**
   - the quantization/export helpers now unbank shared MLP banks only once as `shared_mlp.*`, instead of expanding them back to one tensor per logical layer.
   - this is important; otherwise sharing during training would not reduce final artifact size.

4. **Bigger default lexical memory**
   - `BIGRAM_VOCAB_SIZE` default is raised to `3072`.
   - this is the main reinvestment of the saved MLP bytes.

5. **CPU-friendly attention fallback for smoke tests**
   - if `flash_attn_interface` is unavailable, attention falls back to PyTorch SDPA.
   - when `NUM_HEADS != NUM_KV_HEADS`, the fallback explicitly expands KV heads before SDPA so grouped-query defaults still work off-CUDA.
   - the CUDA/FlashAttention path remains the intended training path for real runs.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603311002_shared-tail-mlp
RUN_ID=shared_tail_mlp SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

That uses the candidate defaults, including:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- 3x MLP
- `BigramHash(3072)`
- XSA on the last 4 layers
- Partial RoPE (`ROPE_DIMS=16`)
- tail shared MLP banks with `4 -> 2` sharing

If you want to test the full evaluation-side stack used by the strongest recent records, enable legal TTT explicitly:

```bash
cd candidates/202603311002_shared-tail-mlp
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
RUN_ID=shared_tail_mlp_ttt SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you want to turn the sharing off for ablation:

```bash
cd candidates/202603311002_shared-tail-mlp
SHARED_MLP_TAIL_LAYERS=0 RUN_ID=shared_tail_mlp_ablate SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main risks and tradeoffs

- Sharing the deepest MLPs may remove too much capacity exactly where the 11-layer stack currently does its most useful refinement.
- Bigger `BigramHash` helps only if the saved bytes are not outweighed by the representational loss from sharing.
- `torch.compile` has already caused surprises in this repo (for example, dead-code-eliminated late QAT), so shared-bank logic must be treated carefully.
- This candidate changes the training-side model, not just the evaluator, so it is less "free" than sliding-window or TTT-only tweaks.
- Tail sharing could interact differently with TTT than with standard post-training evaluation.

## Validation

Commands run in this environment:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202603311002_shared-tail-mlp/train_gpt.py
```

Outcome:

- passed

Attempted CPU smoke test:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
import torch
...
PY
```

Outcome:

- could not run in this environment because the local Python runtime does not have `torch` installed
- the candidate now includes a non-FlashAttention SDPA fallback, including manual KV-head expansion for GQA, so a lightweight import/CPU smoke test is possible on a machine that has PyTorch but not CUDA/FlashAttention
