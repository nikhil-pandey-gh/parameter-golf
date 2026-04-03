# Hybrid KV Reinvestment

## Hypothesis

The repo has already harvested most of the obvious gains from better optimization, late-layer attention tweaks, quantization, and score-first TTT, but every published record still uses `NUM_KV_HEADS=4` uniformly across depth. This candidate tests whether a **depth-aware KV allocation** can free meaningful parameter budget in early layers with little quality loss, then reinvest those saved bytes into features that prior records repeatedly found useful: richer lexical shortcuts and stronger token-identity injection.

For the standard 11-layer candidate run, the schedule is:

- layers 0-4: **1 KV head** (MQA-style)
- layers 5-6: **2 KV heads**
- layers 7-10: **4 KV heads**

The late layers keep full KV capacity for XSA, VE, and score-first TTT, while the early layers pay the compression cost.

## Why this is promising here

- Repo review showed a clear pattern: deeper 11-layer stacks, partial RoPE, LN scale, EMA/SWA, GPTQ-lite quantization, and legal TTT all helped, but **KV-head count never moved off 4** in the published records.
- The best overall run in this repo, `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon`, already demonstrated that the current frontier is about stacking small wins on a strong 11L banked/TTT base rather than rebuilding the training stack from scratch.
- The same record also showed that **bigger lexical shortcuts help** (`BigramHash 2048->3072` improved post-TTT BPB in its ablation), which makes KV compression especially attractive if the saved budget can be reallocated.

## Influencing records and prior candidates

There were **no prior `candidates/` directories** in the repository at implementation time.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`  
  Direct code base and overall training/eval stack: LeakyReLU(0.5)^2, parameter banking, parallel Muon, GPTQ-lite int6, and legal score-first TTT.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`  
  Reinforced the value of VE, GPTQ-lite clip search, and the late-stage 11L recipe.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`  
  Confirmed that partial RoPE + LN scale stack well on the late-layer-XSA architecture.

## External research

This candidate is mainly motivated by three lines of primary-source evidence:

1. **Multi-Query Attention** — Shazeer, 2019 (`arXiv:1911.02150`)  
   Sharing keys/values across heads can preserve quality while materially reducing KV state and bandwidth.
2. **Grouped-Query Attention** — Ainslie et al., 2023 (`arXiv:2305.13245`)  
   Intermediate KV sharing often recovers most of the quality gap between full MHA and pure MQA.
3. **Multi-head Latent Attention** — DeepSeek-V2, 2024 (`arXiv:2405.04434`)  
   Aggressive KV compression can coexist with strong language-model quality if the saved capacity is spent carefully elsewhere.

This implementation is intentionally modest compared with MLA: it keeps standard attention math and just applies a **depth-aware GQA/MQA schedule** that fits the existing codebase.

## What changed versus the chosen base

Base implementation: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Hybrid KV schedule**
   - Added `KV_HEAD_SCHEDULE`
   - The standard 11L / 8H / 4-KV candidate run resolves to `1,1,1,1,1,2,2,4,4,4,4`
   - Other `NUM_LAYERS` / `NUM_KV_HEADS` configurations keep the old uniform behavior unless `KV_HEAD_SCHEDULE` is set explicitly
   - Replaced the single uniform KV bank with grouped K/V banks keyed by KV-head count
   - Threaded the schedule through training, export, unbanking, re-banking, and eval reload
2. **Parameter reinvestment**
   - `BIGRAM_VOCAB_SIZE`: `3072`
   - `VE_DIM`: `192`
   - `VE_LAYERS`: `7,8,9,10`
3. **Candidate defaults aligned with the best banked/TTT recipe**
   - `TTT_ENABLED=1`
   - `TTT_FREEZE_BLOCKS=0`

The rest of the stack stays intentionally close to the strongest existing implementation.

## How to run or evaluate

From this candidate directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
# Disable legal TTT if you only want the post-quant sliding score.
TTT_ENABLED=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Fall back to uniform 4-KV GQA for comparison.
KV_HEAD_SCHEDULE=4 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Validation

Commands run during implementation:

```bash
python -m compileall candidates/202604031047_hybrid-kv-reinvestment/train_gpt.py
```

Outcome:

- `compileall`: **passed**
- Lightweight CPU smoke test: **not feasible in this runner as attempted**, because the available Python environment did not have `torch` installed, and the full training path also requires CUDA plus `flash_attn_interface`

## Main risks and tradeoffs

- **Quality risk**: early-layer MQA may throw away too much KV diversity, especially if this repo's small models rely on broad early mixing more than larger LMs do.
- **Schedule risk**: the exact split (`1,1,1,1,1,2,2,4,4,4,4`) is heuristic; the best allocation may be more conservative or even more aggressive.
- **Reinvestment risk**: bigger BigramHash / VE may not be the highest-return use of the saved budget; the gains could come instead from more MLP width, more VE depth, or a different TTT policy.
- **Complexity risk**: grouped KV banks make the export/reload path more delicate than the uniform-bank baseline, even though the underlying attention computation is unchanged.
