# Shared deep K/V + larger BigramHash

## Hypothesis

The last four layers are already the repo's "refinement stack": they carry XSA, get the most value from shared value embeddings, and are the most plausible place to share attention substructure without destabilizing the whole model. If those deepest layers share a single K/V projection pair, the saved artifact budget can be reinvested into a larger BigramHash table while keeping cheap per-layer K/V scaling vectors untied.

## Why this is promising here

- The current winning line keeps stacking **small, targeted** changes on the 11-layer/XSA/partial-RoPE/EMA/int6 export recipe instead of replacing the whole model.
- Local evidence says **bigger BigramHash keeps helping** (`2048 -> 3072` was worth about `-0.0009 BPB` in the current best ablation), while naive **full layer recurrence regressed badly** in the 1x5090 sweep.
- External work on **cross-layer sharing** and **attention redundancy** suggests that deep attention weights are more reusable than shallow ones, especially if you avoid blunt whole-model tying:
  - ALBERT: <https://arxiv.org/abs/1909.11942>
  - Subformer: <https://arxiv.org/abs/2101.00234>
  - LISA: <https://arxiv.org/abs/2408.01890>
  - MQA / GQA: <https://arxiv.org/abs/1911.02150>, <https://arxiv.org/abs/2305.13245>

## Prior work that influenced this candidate

- **Chosen base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
  - cleanest strong pre-TTT stack with 11L, XSA4, partial RoPE, LN scale, VE, EMA/SWA, late QAT, and GPTQ-lite export.
- **Borrowed proven activation change:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - carries over `LeakyReLU(0.5)^2`, which the current best record reported as a meaningful gain.
- **Bigram/Smear/value trends:** the 2026-03-20 to 2026-03-23 records consistently rewarded BigramHash, SmearGate, XSA, and shared value features.
- **Prior candidates:** none existed in `candidates/` when this was created.

## What changed vs the chosen base

1. **Deep shared K/V projections**
   - `SHARED_KV_LAYERS=7,8,9,10` by default.
   - Layers 7-10 reuse one shared `c_k` and one shared `c_v` module.
   - Each layer keeps its own `kv_scales` control tensor so the shared K/V basis can still be modulated per depth.
2. **Larger BigramHash**
   - default `BIGRAM_VOCAB_SIZE` moves from `2048` to `4096`.
   - the freed deep-layer K/V parameters are intentionally spent on a larger token-pair memory.
3. **Wider VE coverage**
   - default `VE_LAYERS` moves from `9,10` to `7,8,9,10` so the shared refinement stack still gets token-identity reinforcement.
4. **LeakyReLU(0.5)^2**
   - replaces ReLU^2 in the MLP, following the current best record.
5. **Shared-weight-aware export**
   - mixed int6 export now tracks tensor aliases so shared K/V weights are not serialized multiple times if PyTorch exposes them more than once in the `state_dict`.

## How to run

From the repository root:

```bash
RUN_ID=shared_kv_bigram \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=4096 XSA_LAST_N=4 \
SHARED_KV_LAYERS=7,8,9,10 VE_LAYERS=7,8,9,10 VE_DIM=128 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 MUON_WD=0.04 ADAM_WD=0.04 \
WARMDOWN_ITERS=3500 ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 \
EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 candidates/202604091245_shared-kv-bigram/train_gpt.py
```

The script resolves its default dataset and tokenizer paths from the repo root inferred from `__file__`, so it can also be run from inside the candidate directory without rewriting `DATA_PATH` or `TOKENIZER_PATH`.

## Evaluation / expected outcome

- Expect this to behave like a **conservative cross-layer-sharing probe**, not a tokenizer or optimizer rewrite.
- The intended win is better **artifact allocation**: spend fewer bytes on redundant deep K/V projections and more on token-pair memory.
- If the sharing is too aggressive, the likely failure mode is worse long-range conditioning in the deepest layers; `SHARED_KV_LAYERS` is therefore exposed so the sharing span can be narrowed quickly.

## Validation

| Command | Outcome |
| --- | --- |
| `python -m compileall candidates/202604091245_shared-kv-bigram/train_gpt.py` | Passed |
| `python - <<'PY' ... importlib.util.find_spec('torch') / find_spec('flash_attn_interface') ... PY` | `torch` and `flash_attn_interface` were both absent in this session environment |

Because this session environment does not have PyTorch or FlashAttention installed and the training script hard-requires CUDA, a minimal CPU smoke run was **not feasible here** without adding new infrastructure.

## Main risks / tradeoffs

- **Too much sharing:** even deep-only K/V tying may over-regularize the model.
- **Budget reallocation may be wrong:** the larger BigramHash may not repay the lost attention flexibility.
- **Interaction risk:** LeakyReLU^2, wider VE coverage, and deep K/V sharing may not compose linearly even though each ingredient is individually motivated.
