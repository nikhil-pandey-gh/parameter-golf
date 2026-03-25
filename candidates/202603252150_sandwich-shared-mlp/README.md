# Candidate: Sandwich-Shared MLP Budget Reallocation

## Hypothesis

The strongest current non-TTT stack in this repo already spends most of its artifact budget on 11 layers with 3x MLPs, GPTQ-lite int6 export, XSA, EMA, Partial RoPE, and value embeddings. My hypothesis is that the **middle MLPs are now over-parameterized relative to the byte budget**, so selectively sharing only those MLP weights can free meaningful artifact space **without adding any extra per-step compute**.

That recovered budget is then reinvested into two places that repo history says are unusually high leverage:

1. **fp16 tied-embedding export** instead of quantizing the tied embedding, because that tensor has repeatedly been the most quantization-sensitive component in this challenge.
2. **A slightly larger BigramHash table** (`3072` buckets instead of `2048`), because multiple records show that short-context lexical side channels still buy a little BPB even after the deeper 11-layer stack is in place.

I also carry over the recent **LeakyReLU(0.5)^2** activation change because the latest record showed it was a cheap, consistent gain on the same family of 3x-MLP models.

## Why this is promising for this repository

This candidate is designed around a pattern that the record history already makes clear:

- **Naive depth recurrence is bad here** because it raises per-step compute and costs too many optimizer steps under the 10-minute wallclock cap.
- **Compression-aware byte reallocation is good here** because many of the best gains came from spending bytes on the right tensors rather than simply adding more compute.
- **BigramHash and fp16 embedding precision both have strong local evidence** as high-yield uses of budget.

So instead of adding effective depth, this candidate tries to **remove redundant middle-layer MLP bytes** and spend them where the repo has already shown a strong return.

## Base implementation and record influences

### Chosen base

This candidate starts from:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

That record is the strongest clean pre-TTT stack in the repo and already includes the architecture family that has dominated recent wins: 11 layers, 3x MLP, XSA on the deepest layers, Partial RoPE, LN scaling, EMA, GPTQ-lite int6 export, and shared value embeddings.

### Prior records that directly influenced this candidate

- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
  - showed that keeping the tied embedding in fp16 can almost eliminate a painful post-quant gap.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`
  - showed that spending bytes on a larger BigramHash table still helps after the model has become quantization-aware.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
  - showed that `LeakyReLU(0.5)^2` is a cheap activation improvement on the current 11-layer family.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/README.md`
  - reported that simple layer recurrence was net negative because extra effective depth reduced step count too much.

### Prior candidates

There was **no existing `candidates/` directory** in the repository when this candidate was created, so there were no prior candidate implementations to review or avoid duplicating.

## External research that informed this candidate

- **ALBERT** (`arXiv:1909.11942`) argues that cross-layer parameter sharing can reduce parameters substantially while preserving model quality.
- **Subformer** (`arXiv:2101.00234`) is especially relevant here because it studies **parameter sharing in generative transformers** and argues that **sandwich-style sharing** is much better than naive across-the-board tying.
- **Parameter Reduction Improves Vision Transformers** (`arXiv:2512.01059`) is not an LM paper, but it is directly relevant to this exact mechanism: its `GroupedMLP` result shows that **sharing adjacent MLP blocks can preserve compute and even improve stability**.

Those papers collectively suggest a stronger version of the repo's own lesson: full recurrence is too blunt, but **selective sharing of the most redundant weights** can still be useful.

## What changed versus the base implementation

Compared with the `2026-03-22` record script, this candidate makes four focused changes:

1. **Sandwich-style MLP sharing**
   - New hyperparameter: `SHARED_MLP_GROUPS=2-4,5-6`
   - Layers `2,3,4` share one MLP bank.
   - Layers `5,6` share one MLP bank.
   - Attention, norms, residual mixing, layer scales, XSA flags, and value-embedding injection all remain **per-layer**.
   - This preserves the base model's attention specialization while tying only the largest token-wise submodules in the middle of the stack.

2. **fp16 tied embedding passthrough at export**
   - `tok_emb.weight` is stored as fp16 instead of going through int8/int6 export.
   - This follows the repo's prior evidence that the tied embedding is unusually quantization-sensitive.

3. **Bigger lexical side channel**
   - Default `BIGRAM_VOCAB_SIZE` increases from `2048` to `3072`.
   - This uses a small fraction of the bytes saved by MLP sharing.

4. **LeakyReLU(0.5)^2 MLP activation**
   - Replaces plain ReLU^2 with `LeakyReLU(0.5)^2` inside the MLP.
   - This is a lightweight carry-over from the latest winning family.

I also added two small engineering conveniences:

- a **FlashAttention fallback** to `scaled_dot_product_attention` when FlashAttention is unavailable, and
- a `CPU_SMOKE_TEST=1` mode so the script can do a tiny model-init/forward/quantization sanity check in a CPU environment that already has the Python deps installed.

## How to run or evaluate it

From the repository root:

```bash
cd candidates/202603252150_sandwich-shared-mlp

torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The candidate defaults already encode the main idea:

- `BIGRAM_VOCAB_SIZE=3072`
- `SHARED_MLP_GROUPS=2-4,5-6`

Useful ablations:

```bash
# Disable sharing
SHARED_MLP_GROUPS= BIGRAM_VOCAB_SIZE=3072 torchrun --standalone --nproc_per_node=8 train_gpt.py

# More conservative sharing
SHARED_MLP_GROUPS=2-3,5-6 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Larger lexical side channel
BIGRAM_VOCAB_SIZE=4096 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

If you already have the repo dependencies installed and want the tiny CPU sanity path:

```bash
cd candidates/202603252150_sandwich-shared-mlp
CPU_SMOKE_TEST=1 python train_gpt.py
```

## Validation performed

### Successful lightweight validation

```bash
python -m compileall candidates/202603252150_sandwich-shared-mlp/train_gpt.py
```

Outcome: **success**.

### Attempted CPU smoke test

```bash
cd candidates/202603252150_sandwich-shared-mlp
CPU_SMOKE_TEST=1 python train_gpt.py
```

Outcome: **not feasible on this runner** because the environment does not have the repo's required Python packages installed (`numpy`, `torch`, and `sentencepiece` were all missing). The smoke path is implemented in the script, but this workflow runner was missing the ML runtime needed to execute it.

## Main expected risks and tradeoffs

- **Shared middle MLPs may under-specialize** if the base 11-layer stack really needs distinct token-wise transforms at every interior layer.
- **fp16 embedding passthrough consumes real artifact bytes**, so the benefit depends on the sharing change paying for it.
- **LeakyReLU^2 and larger BigramHash may change the best optimizer sweet spot**, especially if the sharing configuration changes the quantization statistics of the MLP bank.
- The most likely next ablations are:
  - `SHARED_MLP_GROUPS=2-3,5-6` versus `2-4,5-6`
  - `BIGRAM_VOCAB_SIZE=3072` versus `4096`
  - fp16 embedding passthrough on/off
  - LeakyReLU^2 on/off
