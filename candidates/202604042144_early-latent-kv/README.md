# Early latent K/V on the 11L EMA+GPTQ-lite base

## Hypothesis

Start from `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` and keep the proven 11-layer recipe intact, but reclaim parameters in the first 7 attention blocks with an MLA-lite low-rank latent K/V path. Spend part of that budget on a larger `BIGRAM_VOCAB_SIZE=4096`, while also folding in the strong `LeakyReLU(0.5)^2` MLP change seen in `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`.

## Why this looks promising here

The chosen base already concentrates its most delicate attention behavior late in the stack: the last 4 layers use XSA, layers 9-10 use VE, and partial RoPE/LN-scale are already tuned. That makes the early 7 layers the safest place to compress K/V with a latent bottleneck while keeping queries full-width and preserving the full-width last-4 XSA path.

## Influences

- Local base record: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- Local activation record: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- External MLA references: DeepSeek-V2 (`https://arxiv.org/abs/2405.04434`) and DeepSeek-V3 (`https://arxiv.org/abs/2412.19437`)

## What changed vs the chosen base

- Added `KV_LATENT_DIM` (default `128`) and `KV_LATENT_LAYERS` (default `0,1,2,3,4,5,6`)
- In latent-K/V layers only, replaced full `c_k`/`c_v` with `kv_down -> k_up` and `kv_down -> v_up`
- Kept `q` full-width and left the attention output shape unchanged
- Automatically keeps the last `XSA_LAST_N` layers full-width, even if `KV_LATENT_LAYERS` overlaps them
- Increased `BIGRAM_VOCAB_SIZE` default from `2048` to `4096`
- Switched the MLP default activation from `relu^2` to `LeakyReLU(0.5)^2`
- Added a startup log line reporting active latent-K/V layers and latent dimension
- Made default dataset/tokenizer paths resolve from the repository root so the script is runnable from this candidate directory

## How to run / evaluate

From `candidates/202604042144_early-latent-kv/`:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
KV_LATENT_DIM=128 KV_LATENT_LAYERS=0,1,2,3,4,5,6 BIGRAM_VOCAB_SIZE=4096 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script still performs its normal training, EMA application, GPTQ-lite/int6 export, and round-trip evaluation flow.

## Main risks / tradeoffs

- The latent bottleneck may over-compress early contextual features and hurt pre-TTT quality
- Sharing one latent projection for both K and V could reduce representational flexibility
- A bigger bigram table spends saved budget on lexical priors; if the latent path underperforms, this may not fully pay back the loss
- LeakyReLU(0.5)^2 is strong in nearby records, but interaction effects with the latent K/V path are still unproven

## Validation

Runtime smoke testing was not feasible in this environment because `torch`, `flash_attn_interface`, and `sentencepiece` are unavailable locally, so validation was limited to syntax compilation.

Validated commands:

```bash
python -m compileall train_gpt.py
python -m py_compile train_gpt.py
```

Outcome: both passed.
