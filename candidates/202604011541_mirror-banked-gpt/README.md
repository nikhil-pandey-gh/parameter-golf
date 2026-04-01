# Mirror-Banked GPT

## Hypothesis

The current frontier in this repository is mostly **artifact-bound, not obviously compute-bound**: the strongest 11-layer runs already sit near the 16 MB cap, and recent gains have come from using those bytes more effectively rather than radically changing FLOPs. This candidate tests whether we can keep the strong **11 logical layer** recipe from the current best lineage while storing only **6 physical weight banks** shared across mirrored depths.

The key bet is that this differs from the earlier negative "layer recurrence" result in an important way: it does **not** add extra logical depth or reduce step throughput. Instead, it keeps the same logical pass count and per-layer control parameters, but shares the largest matrices that dominate the compressed artifact.

## Why this is promising here

Three repo trends point in this direction:

- The best records have converged on an **11-layer, 512-dim, 3x-MLP, GQA, U-Net-ish skip** stack with aggressive int6/int8 export and evaluation tricks.
- The top runs are all very close to the **16,000,000 byte** ceiling, so freeing bytes inside the largest tensors is now more valuable than another tiny LR sweep.
- The non-record recurrence experiment on 1x5090 found that naive looping was negative because it **cut effective training steps**. This candidate avoids that failure mode by preserving the same 11 logical transformer applications and only sharing the stored bank tensors.

Because the banked `train_gpt.py` already separates large matrix banks from small per-layer controls, this is a natural place to try **ALBERT-style sharing** without losing layer-specific RMSNorm scaling, residual mixing, skip weights, XSA placement, or value-embedding placement.

## Influential prior records and candidates

There was **no pre-existing `candidates/` directory** when this candidate was created, so there were no older candidate iterations to inherit from.

This candidate is mostly influenced by:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest overall lineage in-repo
  - parameter-banked large weights
  - LeakyReLU(0.5)^2, XSA on late layers, partial RoPE, VE, legal TTT
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strong pre-TTT base showing the same 11-layer stack is already competitive before adaptation
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
  - evidence that spending freed bytes on a larger **BigramHash** table can help
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - useful negative result: simple layer recurrence hurt because it reduced step count under a fixed wallclock budget

## External research that informed this

- **ALBERT**: cross-layer parameter sharing can preserve logical depth while reducing stored parameters.
  - https://arxiv.org/abs/1909.11942
- **Universal Transformer**: repeated/iterative depth can work when depth reuse is treated as a first-class design choice instead of an afterthought.
  - https://arxiv.org/abs/1807.03819

These papers are not a claim that sharing will automatically win here; they are the main reason this seemed like the strongest *new* byte-saving direction not already represented in the repo history.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

This candidate makes three focused changes:

1. **Mirror-shared parameter banks**
   - Added `PHYSICAL_LAYERS` and `BANK_SHARE_SCHEME`.
   - Default config uses `NUM_LAYERS=11`, `PHYSICAL_LAYERS=6`, `BANK_SHARE_SCHEME=mirror`.
   - The default logical-to-physical mapping is:
     - `[0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]`
   - Logical blocks keep their own small per-layer controls, but the large `qo_bank`, `kv_bank`, `mlp_up_bank`, and `mlp_down_bank` tensors are reused across mirrored depths.

2. **Larger lexical shortcut by default**
   - Default `BIGRAM_VOCAB_SIZE` increased from `2048` to `4096`.
   - The point is to reinvest some of the artifact savings into a repo-proven feature rather than only banking the headroom.

3. **Direct bank-wise export quantization**
   - The base script unbanked 3D tensors into per-layer 2D weights before int6 packing.
   - That would duplicate shared banks at export time and erase the byte savings.
   - This candidate quantizes bank tensors directly, with per-row scales over the last dimension, so shared banks remain shared in the serialized artifact.

## How to run or evaluate

From the repository root:

```bash
cd candidates/202604011541_mirror-banked-gpt
RUN_ID=mirror_banked_gpt \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
TTT_ENABLED=1 \
TTT_FREEZE_BLOCKS=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Notes:

- `PHYSICAL_LAYERS=6`, `BANK_SHARE_SCHEME=mirror`, and `BIGRAM_VOCAB_SIZE=4096` are already the defaults in this candidate.
- If you want a faster pre-TTT sanity run, set `TTT_ENABLED=0`.
- Like the base lineage, this script expects CUDA + FlashAttention 3 for the real training/eval path.

## Main expected risks and tradeoffs

- **Underfitting from over-sharing**: encoder and decoder roles may want genuinely different weights even if they sit at mirrored depths.
- **Optimization coupling**: a shared bank now receives gradients from multiple logical positions, which may smooth useful specialization.
- **Export-path novelty**: direct bank-wise quantization is simple, but it is new relative to the base script and may interact differently with compression.
- **Bigram size may not be optimal**: `4096` is a grounded default, not a tuned optimum.

## Validation

I ran lightweight validation that fits this repository and this workflow environment.

### 1. Python compilation

Command:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604011541_mirror-banked-gpt/train_gpt.py
```

Outcome:

- `train_gpt.py` compiled
- `train_gpt_mlx.py` compiled
- `data/cached_challenge_fineweb.py` compiled
- `data/download_hf_docs_and_tokenize.py` compiled
- `candidates/202604011541_mirror-banked-gpt/train_gpt.py` compiled

### 2. CPU smoke test with a FlashAttention stub

Because the workflow container did not initially have the repo Python dependencies installed, I created a temporary venv under `/tmp/gh-aw/agent/pg-venv`, installed minimal runtime deps there, injected a tiny CPU `flash_attn_interface` stub, and exercised the candidate model on random inputs.

The smoke test covered:

- GPT construction with `NUM_LAYERS=11`, `PHYSICAL_LAYERS=6`, `BANK_SHARE_SCHEME=mirror`
- forward loss computation
- export-side `mixed_quantize_int6`
- roundtrip `dequantize_mixed_int6`
- strict `load_state_dict` into a fresh mirrored-bank model
- second forward pass after roundtrip

Observed output:

```python
{'loss': 4.916476249694824, 'loss_roundtrip': 4.916472911834717, 'layer_to_bank': [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]}
```

That is only a structural smoke test, not a training-quality claim, but it confirms the new bank-sharing path and the new direct-quantization export path both run end-to-end on CPU with a local attention stub.

### 3. GPU smoke status

I did **not** run a real CUDA training/evaluation smoke here because this workflow container is not provisioned with the dataset + CUDA + FlashAttention training environment needed by the repo’s actual runtime path. The CPU stub smoke above is the closest safe low-cost substitute available in this environment.
