# Candidate: Paired Block Sharing

## Hypothesis

The strongest underexplored lever in this repo is **fixed-depth weight sharing**, not extra recurrent unrolling. If adjacent transformer blocks share the large attention and MLP banks while keeping **layer-specific norms, residual scales, skip weights, XSA flags, and VE scales** untied, the model should keep most of its depth-specific behavior while paying for fewer high-entropy matrices in the artifact. Those saved bytes can be reinvested into slightly richer token priors with almost no step-time penalty.

In short: **share the expensive banks, keep the cheap per-layer controls, and use the saved budget on higher-capacity token-side features**.

## Why this is promising here

Repository review showed a clear trend:

- the leaderboard keeps reusing the same 11-layer 512d stack with **3x MLP, BigramHash, SmearGate, partial RoPE, XSA, LN scale, VE, EMA, and better quantization**;
- **true depth recurrence already failed** under fixed wallclock, because extra logical depth cost too many steps;
- the best clean training-side stack is the 2026-03-22 11L EMA + GPTQ-lite recipe, while the 2026-03-23 top score adds LeakyReLU^2, parameter banking, and legal TTT on top.

That makes block sharing attractive: it targets the **artifact bottleneck** directly without repeating the failed “loop the same layer more times” idea.

## Prior records and candidates that influenced this

- **2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233**: strongest clean training-side stack; this candidate keeps its general architecture motifs.
- **2026-03-23_LeakyReLU_LegalTTT_ParallelMuon**: used as the code substrate because its banked weights make *true* shared export practical instead of aliasing duplicated tensors in a state dict.
- **2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248**: confirms partial RoPE + LN scale are robust zero-/low-cost wins worth preserving.
- **2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA** and **2026-03-20_10L_Int5MLP_MuonWD04_SWA50**: motivate keeping BigramHash/SmearGate and using some recovered budget on richer token priors.
- **2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090**: important negative result; explicit layer recurrence hurt badly under a 10-minute wallclock.
- There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **MobileLLM / MobileLLM-LS** (`arXiv:2402.14905`): argues sub-billion LMs are unusually architecture-sensitive, and reports gains from **immediate block-wise weight sharing** on top of deep/thin + GQA designs.
- **ALBERT** (`arXiv:1909.11942`): classic evidence that cross-layer sharing can cut parameters aggressively while preserving much of the model’s behavior.
- **GQA** (`arXiv:2305.13245`): supports keeping grouped-query attention as the attention baseline instead of changing head layout at the same time.
- I also reviewed **MLA / DeepSeek-V2** (`arXiv:2405.04434`) as a possible compression path, but it looked too KV-cache/inference-focused and too invasive for this repo’s lightweight training script.

## What changed vs. the chosen base implementation

This candidate copies the **2026-03-23** `train_gpt.py` stack into an isolated candidate directory, then makes the following changes:

1. **Adjacent paired bank sharing by default**
   - New knobs:
     - `SHARE_ATTENTION_BANKS=1`
     - `SHARE_MLP_BANKS=1`
     - `BANK_SHARE_MODE=adjacent`
     - `BANK_SHARE_GROUP_SIZE=2`
   - With the default 11 layers and pair size 2, the script now uses **6 unique attention bank groups** and **6 unique MLP bank groups** instead of 11 unique banks each.

2. **Layer-specific controls stay untied**
   - RMSNorms, residual mixing, attn/MLP scales, skip weights, XSA placement, VE layer scales, and other small control tensors remain per-layer.
   - This is the core design choice: share the heavy matrices, not the cheap layer identity.

3. **Direct quantization of shared 3D banks**
   - The original banked export path unbanked tensors into per-layer 2D names before quantization.
   - That would silently duplicate shared weights in the artifact.
   - This candidate instead quantizes the bank tensors **directly**, row-wise across each 2D slice inside the 3D banks, so sharing survives serialization.
   - The quantized artifact also stores the bank-sharing config so adjacent / mirror / partial-sharing ablations reload into the correct topology.

4. **Modest reinvestment of saved bytes into token priors**
   - `BIGRAM_VOCAB_SIZE`: `2048 -> 4096`
   - `VE_DIM`: `128 -> 192`
   - `VE_LAYERS`: `"9,10" -> "7,8,9,10"`

5. **Inherited strong defaults kept**
   - LeakyReLU^2 MLP
   - parameter banking + Parallel Muon
   - XSA on the last 4 layers
   - partial RoPE
   - LN scale
   - shared value embeddings
   - EMA / SWA / GPTQ-lite export path
   - legal TTT remains in the script, but **`TTT_ENABLED=0` by default** for this candidate since the architectural change is the main experiment

## How to run or evaluate

From the candidate directory:

```bash
cd candidates/202604090027_paired-block-sharing

SHARE_ATTENTION_BANKS=1 \
SHARE_MLP_BANKS=1 \
BANK_SHARE_MODE=adjacent \
BANK_SHARE_GROUP_SIZE=2 \
BIGRAM_VOCAB_SIZE=4096 \
VE_DIM=192 \
VE_LAYERS=7,8,9,10 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

- **MLP-only sharing**: `SHARE_ATTENTION_BANKS=0 SHARE_MLP_BANKS=1`
- **Attention-only sharing**: `SHARE_ATTENTION_BANKS=1 SHARE_MLP_BANKS=0`
- **No sharing**: `SHARE_ATTENTION_BANKS=0 SHARE_MLP_BANKS=0`
- **Mirror sharing**: `BANK_SHARE_MODE=mirror`

If you re-enable legal TTT on this banked script, keep **`TTT_FREEZE_BLOCKS=0`**. Partial block freezing is intentionally rejected because the large shared banks are top-level parameters rather than `blocks.{i}.*` tensors.

## Validation

Commands run in this repository:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data candidates/202604090027_paired-block-sharing/train_gpt.py
python - <<'PY'
import importlib.util
print(f'torch_present={importlib.util.find_spec("torch") is not None}')
print(f'flash_attn_interface_present={importlib.util.find_spec("flash_attn_interface") is not None}')
PY
```

Outcomes:

- `compileall` **passed**
- CPU startup smoke test was **not feasible in this runner**
  - `torch_present=False`
  - `flash_attn_interface_present=False`
  - the script also requires CUDA at runtime, so there was no safe local path to verify a real training startup beyond syntax

## Main expected risks and tradeoffs

- **Over-sharing could erase useful layer diversity**, especially for attention banks.
- **Shared attention may be riskier than shared MLP**, so the first ablation to try is likely MLP-only sharing.
- The reinvestment here is intentionally conservative; the recovered budget might be better spent on a different capacity target (for example a slightly wider token-side module or a different VE layout).
- Because this is implemented on the banked 2026-03-23 substrate for correctness of export, it is more engineering-heavy than starting from the cleaner 2026-03-22 script.
