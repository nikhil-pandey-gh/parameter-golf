# Paired MLP Sharing

## Hypothesis

The strongest recent training-only stack in this repo already leans hard on deeper 11-layer U-Net structure, EMA, Partial RoPE, XSA, and quantization-aware export. A still-open direction is **cross-layer parameter sharing**: if mirrored encoder/decoder layers reuse the same MLP weights while keeping attention, norms, skip weights, and learned residual scales layer-specific, the model may keep most of the depth benefit while reducing duplicated feedforward parameters.

In this repository, that should help in two ways:

1. it lowers the number of unique MLP matrices that must survive post-training compression, and
2. it creates room to slightly strengthen token-local features (here, a larger BigramHash table) without blowing past the artifact budget.

This candidate also folds in the recent **LeakyReLU(0.5)^2** MLP activation improvement from the latest record, since it is orthogonal to the sharing change.

## Why this is promising for this repo

The repo's best clean training-only runs cluster around:

- 11-layer U-Net-style depth,
- 3x MLPs,
- Partial RoPE + LN scaling,
- XSA on late layers,
- EMA / SWA style averaging,
- compression-aware export.

What the repo has **not** tried yet is ALBERT-style / paired cross-layer sharing. The current architecture is a particularly natural place to test it because the stack is already split into encoder and decoder halves with mirrored skip connections. Sharing the MLPs across mirrored layers is a smaller and safer intervention than sharing full blocks or replacing attention altogether.

## Prior repo influences

At review time there was **no `candidates/` directory**, so this idea is new relative to both the baseline and prior experiments.

The most important record influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen as the direct code base because it is the strongest low-complexity training-only record in the repo.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - supplied the strong Partial RoPE + LN-scale architectural core.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - supplied the LeakyReLU(0.5)^2 activation change and the evidence that a larger bigram table helped in the strongest later stack.

## External research that informed this candidate

- **ALBERT** — arXiv:1909.11942
  - motivated cross-layer parameter sharing as a way to improve parameter efficiency without discarding depth.
- **Universal Transformer** — arXiv:1807.03819
  - reinforced the idea that recurrent/shared depth can be useful when the architecture keeps layer-wise iterative refinement behavior.
- **ShishuLM** — arXiv:2510.13860
  - especially relevant because it explicitly studies lightweight language models and paired weight sharing for lower memory / latency.
- **SCORE: Replacing Layer Stacking with Contractive Recurrent Depth** — arXiv:2603.10544
  - recent evidence that shared-depth alternatives can reduce parameter count while remaining competitive in transformer language models.

## What changed versus the chosen base

Base implementation copied from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

1. **Mirrored paired MLP sharing**
   - logical layer `i` and logical layer `num_layers - 1 - i` reuse the same `MLP` module.
   - attention, norms, residual mixing, `attn_scale`, `mlp_scale`, skip weights, VE scales, and XSA placement remain layer-specific.
   - for the default 11-layer stack this reduces the number of unique MLPs from 11 to 6.

2. **LeakyReLU(0.5)^2 MLP activation**
   - replaces `relu^2` with `leaky_relu(negative_slope=0.5)^2`.

3. **Slightly larger BigramHash table by default**
   - default `BIGRAM_VOCAB_SIZE` is raised from `2048` to `3072`.

4. **Compression-aware export behavior for shared MLPs**
   - shared MLP weights live under `shared_mlps.*` and intentionally stay on the higher-fidelity mixed-quant path than the original int6 MLP bucket.
   - the hypothesis is that fewer unique MLP matrices plus gentler export may narrow the post-quantization gap.

5. **Safer runtime fallback for validation / portability**
   - if `flash_attn_interface` is unavailable, attention falls back to `torch.nn.functional.scaled_dot_product_attention`.
   - a `SMOKE_TEST=1` mode was added so the script can do a tiny synthetic forward pass without dataset or tokenizer setup when PyTorch is available.

## How to run

From the repository root:

```bash
cd candidates/202603261851_paired-mlp-sharing
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The defaults are already set to the candidate configuration. Useful overrides:

```bash
SEED=1337 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_SEQ_LEN=2048 \
EVAL_SEQ_LEN=2048 \
SHARE_MLP_PAIRS=1 \
MLP_NEGATIVE_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional tiny smoke path when dependencies are installed:

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Expected risks and tradeoffs

- **Over-sharing risk**: mirrored layers may still want different feedforward subspaces even if they can modulate them with layer-specific scales.
- **Quantization tradeoff**: keeping shared MLPs on a gentler export path may improve fidelity, but the extra bytes must still leave enough margin under the 16MB cap.
- **Interaction risk**: the larger BigramHash table and shared MLPs may interfere rather than complement each other.
- **Training-speed uncertainty**: parameter sharing reduces artifact redundancy more than FLOPs, so the main expected gain is quality-per-byte rather than speed.

## Validation run in this workflow

Commands run:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
python -m compileall candidates/202603261851_paired-mlp-sharing/train_gpt.py
```

Outcome:

- both compile-only checks passed.
- a real CPU smoke run was **not feasible in this workflow environment** because the runner did not have a usable PyTorch runtime installed (`import torch` failed), so only syntax validation was executed here.
