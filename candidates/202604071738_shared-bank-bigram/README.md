# Shared Bank Reuse + Bigger Bigram

## Hypothesis

The current repo frontier is close to the 16 MB artifact ceiling, so the next useful gain may come from **reallocating bytes** rather than adding more execution depth. This candidate keeps the strong March 23 banked 11-layer stack, but **shares only the heavy bank tensors across a palindromic layer map** while keeping each logical layer's norms, residual mixing, skip weights, XSA placement, and value-embedding scales independent.

The saved bytes are then spent on a larger hashed lexical side channel by raising the default `BIGRAM_VOCAB_SIZE` from 2048 to **4096**.

## Why this is promising here

- The best records already show that **BigramHash**, **EMA**, **GPTQ-lite / int6 export**, **Partial RoPE**, and **XSA** are reliable contributors.
- The repo also shows that naive **extra recurrence** can fail when it increases compute and reduces training steps.
- This candidate avoids that failure mode: it keeps the same **logical depth and execution count** as the 11-layer stack, but reduces stored heavy weights so the artifact budget can be spent more deliberately.

## Prior repo evidence

### Main influences

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md`  
  Best pure training/export stack before legal TTT; establishes EMA + GPTQ-lite + warmdown 3500 as a strong base.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`  
  Supplies the banked heavy-weight layout and the current best activation choice, `LeakyReLU(0.5)^2`.
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/README.md`  
  Reinforces that BigramHash is worth spending budget on.

### Important negative result

- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md` and `results.tsv`  
  The `layer recurrence x2` trial regressed badly because it increased compute and reduced step count. This candidate does **not** increase logical depth beyond 11 executed layers; it only shares stored bank slices.

### Prior candidates

There were **no existing `candidates/` experiments** in the repository when this candidate was created.

## External research

This candidate is mainly informed by:

- **ALBERT** — cross-layer parameter sharing as a way to shift scarce parameter budget away from duplicated transformer blocks: <https://arxiv.org/abs/1909.11942>
- **Universal Transformer** — recurrent/shared-depth transformer computation, useful here as a reminder that execution depth and stored weights do not have to be tied 1:1: <https://arxiv.org/abs/1807.03819>
- Supporting compact-model / compression context reviewed during research:  
  <https://arxiv.org/abs/1902.08153>  
  <https://arxiv.org/abs/1805.06085>  
  <https://arxiv.org/abs/1809.10853>  
  <https://arxiv.org/abs/2211.10438>  
  <https://github.com/mit-han-lab/llm-awq>

The concrete takeaway for this repo was: **share the expensive block matrices, not the entire per-layer control surface**, and use the recovered budget on features that already work locally.

## What changed vs the chosen base implementation

Base implementation:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. Added **repo-root-relative default paths** so the script can be run directly from this candidate directory.
2. Added `BANK_LAYER_MAP`, with a default **palindromic sharing map** for 11 layers:
   - `0,1,2,3,4,5,4,3,2,1,0`
3. Resized the heavy parameter banks (`qo_bank`, `kv_bank`, `mlp_up_bank`, `mlp_down_bank`) to the number of **shared bank layers** instead of the number of logical layers.
4. Kept logical-layer-local parameters **untied**:
   - RMSNorms
   - residual mixing
   - skip weights
   - XSA placement
   - VE layer scales
5. Updated export quantization to serialize each shared bank slice only once via `bank_blocks.*`, and to store the chosen `BANK_LAYER_MAP` in checkpoint metadata so the artifact is self-describing.
6. Increased the default hashed lexical budget:
   - `BIGRAM_VOCAB_SIZE=4096`

## How to run

From the repository root:

```bash
cd candidates/202604071738_shared-bank-bigram
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
BANK_LAYER_MAP=0,1,2,3,4,5,4,3,2,1,0 \
BIGRAM_VOCAB_SIZE=4096 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

To disable sharing and recover the original per-layer banking behavior:

```bash
BANK_LAYER_MAP=0,1,2,3,4,5,6,7,8,9,10 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Evaluation

The script keeps the same evaluation/export flow as the base stack:

- EMA-applied diagnostic validation
- int6+lzma roundtrip
- sliding-window evaluation
- optional legal TTT if `TTT_ENABLED=1`

## Risks and tradeoffs

- **Capacity risk:** sharing too aggressively can underfit, even if the artifact gets much smaller.
- **Budget-allocation risk:** a bigger BigramHash table may not repay the loss from fewer unique block matrices.
- **Best map uncertainty:** the default palindromic map is motivated by the repo's U-Net-like shape, but other maps may work better.
- **Quantization interaction:** a smaller set of shared banks changes error concentration under post-training quantization; the best clip policy may shift.

## Suggested next ablations

1. Compare the default map against no sharing:
   - `BANK_LAYER_MAP=0,1,2,3,4,5,6,7,8,9,10`
2. Try a flatter shared core:
   - `BANK_LAYER_MAP=0,1,2,3,4,5,5,4,3,2,1`
3. Sweep how the recovered bytes are spent:
   - `BIGRAM_VOCAB_SIZE`
   - `VE_DIM`
   - keeping more tensors out of int6 if artifact headroom remains

## Validation run here

Executed in this workflow:

```bash
python -m compileall candidates/202604071738_shared-bank-bigram/train_gpt.py
```

Outcome:

- Syntax compilation succeeded.
- A CPU startup smoke run was **not feasible** in this workspace because:
  1. the cached FineWeb shards are not present under `data/datasets/`, and
  2. this training script still requires CUDA / flash-attention style runtime dependencies for an actual launch.
