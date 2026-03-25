# Shared-depth adapters

## Hypothesis

The strongest non-TTT training stack in this repository already uses most of the artifact budget on an 11-layer, 512-dim, 3x-MLP transformer. The next meaningful gain is likely not another tiny export tweak, but a better **bytes-to-depth trade**: keep the strongest 11L EMA/GPTQ-lite recipe, reuse the deepest blocks across multiple logical layers, and add tiny per-layer adapters so repeated passes are not identical.

This candidate therefore replaces the fully unique deep tail with a **shared-depth decoder suffix**:

- **13 effective layers** by default
- **9 unique block cores** total
- deepest **6 logical layers** realized by only **2 shared cores** repeated 3 times each
- unique per-logical-layer norms, residual scales, and **rank-8 low-rank adapters**

The intuition is that this repository's frontier already proved that extra depth matters (`9L -> 10L -> 11L` was consistently good), while recent records mostly competed on quantization/export polish. Shared depth is a direct way to buy more effective depth under the same 16 MB artifact cap.

## Why this seems promising here

Recent records suggest three things very clearly:

1. More useful depth helps when the artifact still fits.
2. The strongest stack has already converged on `XSA + EMA + Partial RoPE + LN scale + GPTQ-lite + bigram/smear`.
3. The remaining budget bottleneck is mostly model bytes, not code size.

That makes this repo a good fit for **parameter sharing across depth**, especially in the late decoder-style layers where repeated contextual refinement may matter more than unique parameters.

## Prior records that influenced this candidate

Primary base implementation:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

Other records that shaped the design:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/`
  - showed that `XSA + EMA` is a strong late-layer stack
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - showed that `Partial RoPE + LN scale` improves the 11-layer line with zero or tiny byte cost
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - confirmed that the frontier remains deep, artifact-tight, and late-layer-sensitive, even if this candidate intentionally avoids TTT complexity

## External research that informed it

The candidate is motivated by a cluster of weight-sharing / depth-reuse papers:

- **MobileLLM** (`https://arxiv.org/abs/2402.14905`)
  - highlights that sub-billion LMs are highly architecture-sensitive, and reports gains from deep-thin designs plus block-wise sharing
- **ALBERT** (`https://arxiv.org/abs/1909.11942`)
  - classic evidence that cross-layer parameter sharing can dramatically cut parameters with limited quality loss
- **Universal Transformer** (`https://arxiv.org/abs/1807.03819`)
  - motivates recurrent/shared depth as a strong inductive bias
- **Subformer** (`https://arxiv.org/abs/2101.00234`)
  - specifically studies sandwich-style sharing for generative transformers
- **Beyond Universal Transformer: block reusing with adaptor** (`https://arxiv.org/abs/2303.13072`)
  - supports the exact pattern used here: reuse blocks, then add tiny per-depth adapters
- **ResidualTransformer** (`https://arxiv.org/abs/2310.02489`)
  - argues for a shared full-rank core plus small layer-specific residual components
- **SOLAR 10.7B** (`https://arxiv.org/abs/2312.15166`)
  - recent evidence that depth up-scaling can be a simple, effective way to increase model capacity

## What changed versus the chosen base implementation

This candidate forks:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

and keeps the following intact:

- EMA export path
- GPTQ-lite-style per-row clip search for int6 export
- SmearGate + BigramHash
- partial RoPE
- late-layer XSA
- shared value embedding path
- sliding-window evaluation support
- Muon/Adam optimizer split

The main architectural changes are:

1. **Shared-depth schedule**
   - New defaults:
     - `NUM_LAYERS=13`
     - `SHARED_PREFIX_LAYERS=7`
     - `SHARED_SUFFIX_UNIQUE=2`
     - `SHARED_SUFFIX_REPEATS=3`
   - Default logical-to-core schedule:
     - `[0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 8, 7, 8]`

2. **Unique per-logical-layer wrappers**
   - each logical layer keeps its own:
     - RMSNorms
     - residual mixing
     - attention/MLP residual scales
     - optional XSA flag

3. **Per-layer low-rank adapters**
   - `LAYER_ADAPTER_RANK=8` by default
   - one adapter on the attention branch and one on the MLP branch per logical layer
   - intended to preserve some layer identity after block sharing

4. **Lightweight fallback attention path**
   - when `flash_attn_interface` is unavailable, the script can fall back to PyTorch SDPA for smoke/debug use

5. **Smoke-test path**
   - `SMOKE_TEST=1` instantiates the model, runs a tiny forward pass, and checks quantization round-trip coverage without touching dataset files

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603252251_shared-depth-adapters
```

Default training/eval launch:

```bash
NUM_LAYERS=13 \
SHARED_PREFIX_LAYERS=7 \
SHARED_SUFFIX_UNIQUE=2 \
SHARED_SUFFIX_REPEATS=3 \
LAYER_ADAPTER_RANK=8 \
VE_LAYERS=11,12 \
XSA_LAST_N=4 \
ROPE_DIMS=16 \
LN_SCALE=1 \
MUON_WD=0.04 \
ADAM_WD=0.04 \
MATRIX_LR=0.025 \
SCALAR_LR=0.025 \
TIED_EMBED_LR=0.035 \
WARMDOWN_ITERS=3500 \
ITERATIONS=9000 \
MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Optional local smoke check (no dataset access):

```bash
SMOKE_TEST=1 python train_gpt.py
```

## Validation run in this environment

Commands attempted:

```bash
python -m compileall candidates/202603252251_shared-depth-adapters/train_gpt.py
SMOKE_TEST=1 python candidates/202603252251_shared-depth-adapters/train_gpt.py
```

Outcomes:

- `python -m compileall ...` **succeeded**
- `SMOKE_TEST=1 ...` could **not** be completed in this runner because the environment does not have `torch` installed (`ModuleNotFoundError: No module named 'torch'`)

The smoke-test entrypoint remains in the candidate script for use on a normal Parameter Golf runtime with PyTorch available.

## Main expected risks and tradeoffs

- **Too much sharing may underfit** and erase the gains from extra effective depth.
- Reusing the same late cores could make **quantization error compound** across repeated passes.
- `torch.compile(fullgraph=True)` may be more brittle with the new logical-to-core indirection than the fully unique baseline.
- The default schedule may not be optimal; a `8 unique + 2 shared x3` or `7 unique + 3 shared x2` variant could work better.
- Rank-8 adapters may be too small or too large; this is a natural ablation axis.

## Suggested next experiments

- Compare the default `7 + (2 shared x 3 repeats)` schedule against `7 + (3 shared x 2 repeats)`.
- Sweep `LAYER_ADAPTER_RANK` across `0, 4, 8, 16`.
- Test whether the deepest repeated core should keep `XSA` on every repeated use, or only the final passes.
- If artifact headroom remains, try a slightly larger BigramHash or VE schedule on top of the shared-depth stack.
