# SmoothSquare Equalization on the 2026-03-22 GPTQ-lite Base

## Hypothesis

The strongest non-TTT stack in this repo is already very good in float space; the remaining gap is mostly in the post-training int6 export path. This candidate applies a **SmoothQuant/AWQ-style equivalent transformation** only to each squared-ReLU MLP pair before serialization: scale each hidden channel in `mlp.fc.weight`, then compensate in `mlp.proj.weight` with the inverse square so the full-precision transform is function-preserving while the MLP weights become easier to quantize.

Because the MLP activation is `relu(x)^2`, the function is degree-2 homogeneous. That means the balancing scale can be chosen with a **cubic-root rule**:

```text
up_row * s ~= down_col / s^2  =>  s = (down_stat / up_stat)^(1/3)
```

The expectation is a smaller int6 roundtrip / sliding-window gap at essentially zero training-time cost. In code, the untouched EMA checkpoint is still saved as `final_model.pt`; the equalized copy is used only for the int6 export path.

## Why this is promising for this repository

- Repo history says the frontier is now dominated by **quantization-aware tricks**, not broad architecture rewrites.
- The best pre-TTT run, `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`, already showed that a better export path (`GPTQ-lite`) was worth `-0.0006 BPB`.
- The unlimited-compute non-record baseline still had a meaningful pre-quant vs post-quant gap, which suggests quantization remains a bottleneck even after long training.
- MLP weights dominate a large fraction of the artifact, and this repo's `relu^2` MLPs make exact equalization unusually easy.

## Prior repository runs that influenced this candidate

1. **Primary base**: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233`
   - strongest clean training-only stack to adapt
   - already uses EMA + tight SWA + GPTQ-lite int6 export
2. `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248`
   - useful caution that quantization-side changes can silently fail under compile-time assumptions
3. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`
   - shows the training stack is already near the frontier, so a new **orthogonal quantization gain** is more attractive than redoing TTT here
4. `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3`
   - highlights that even long training still leaves a noticeable quantization gap

There were no prior `candidates/` directories in this repo at implementation time.

## External research that informed it

- **SmoothQuant** (Xiao et al., arXiv:2211.10438): equivalent transformations can shift quantization difficulty without changing the full-precision function.
- **AWQ** (Lin et al., arXiv:2306.00978): scaling salient channels before weight-only quantization can materially reduce quantization error.
- **OmniQuant / LET** (Shao et al., arXiv:2308.13137): learnable equivalent transformations are a strong low-bit lever, but this candidate chooses the simpler exact transformation path already compatible with this repo.
- **SpinQuant / QuaRot** (Liu et al., arXiv:2405.16406; Croci et al., arXiv:2404.00456): more evidence that quantization-preserving basis changes are powerful, though those methods are heavier than this repo likely needs for a first pass.

## What changed vs the chosen base implementation

Base file: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

This candidate keeps the 11-layer GPTQ-lite/EMA/XSA/Partial-RoPE/VE stack intact and only adds:

1. Three config knobs:
   - `MLP_EQUALIZE=1`
   - `MLP_EQUALIZE_MAX_SCALE=4.0`
   - `MLP_EQUALIZE_EPS=1e-6`
2. `equalize_relu2_mlp_state_dict(...)`, which:
   - finds each `blocks.{i}.mlp.fc.weight` / `blocks.{i}.mlp.proj.weight` pair
   - computes per-hidden-channel cubic-root scaling from row/column absolute maxima
   - folds the compensating transform into a quantization-only export copy before int6 serialization
3. Export logging so runs show the equalization scale range before `mixed_quantize_int6(...)`

No architecture, optimizer, tokenizer, or evaluation logic was otherwise changed.

## How to run

From this candidate directory:

```bash
RUN_ID=smoothsquare_eq \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default dataset and tokenizer paths relative to the repository root, so the command above can be launched directly from `candidates/202604062314_smoothsquare-eq/` without rewriting `DATA_PATH` or `TOKENIZER_PATH`.

To make the new behavior explicit:

```bash
RUN_ID=smoothsquare_eq \
SEED=1337 \
MLP_EQUALIZE=1 \
MLP_EQUALIZE_MAX_SCALE=4.0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The rest of the default hyperparameters intentionally match the 2026-03-22 record base.

## Validation

### Completed here

```bash
python -m compileall candidates/202604062314_smoothsquare-eq/train_gpt.py
```

Outcome: **passed**.

### Not feasible here

I attempted to add a tiny CPU functional smoke for the new equalization helper, but this runner image does not have **PyTorch** installed. A full training start is also not practical in this environment because the inherited record script expects the challenge CUDA stack (including FlashAttention) plus dataset shards. For that reason, validation here is limited to syntax-level checks.

## Main expected risks / tradeoffs

- The transform is exact in float arithmetic, but it may still fail to improve the **row-wise int6** quantizer if the dominant outliers live outside the MLP channels it rebalances.
- Equalizing MLP weights may help quantization error while doing little for compressed size if the remaining bottleneck shifts to embeddings or attention.
- This candidate intentionally keeps the rest of the 2026-03-22 stack fixed, so it does **not** yet test whether the same export trick composes with the faster banked stack or the current legal-TTT winner.
- The scale cap (`MLP_EQUALIZE_MAX_SCALE`) is conservative; if it is too tight the effect may be muted, and if loosened too far it could over-amplify marginal channels before quantization.
