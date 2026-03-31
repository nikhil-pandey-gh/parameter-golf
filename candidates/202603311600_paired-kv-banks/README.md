# Paired late-KV banks

## Hypothesis

The strongest pure-training stack in this repository already compresses attention aggressively across heads with 4 KV heads, adds shared value embeddings on late layers, and gets most of its gains from better export-aware training. My hypothesis is that the **late decoder KV projections are still over-parameterized** for this 16MB regime: if the last four layers share their `K`/`V` projections in paired banks, the model should keep most of the depth benefit while freeing enough artifact budget to reinvest in a larger BigramHash table.

Concretely, this candidate shares `c_k` and `c_v` in the final four layers as two interleaved banks (`[7,9]` and `[8,10]` by default), keeps layer-specific queries/output projections/norms/scales, and then **serializes those shared weights only once** during int6 export.

## Why this is promising for this repository

Repository history points in the same direction:

- `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` is the cleanest strong base: 11 layers, XSA4, partial RoPE, LN scaling, EMA, GPTQ-lite, SmearGate, BigramHash, and shared late value embeddings.
- `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` shows that carrying forward **LeakyReLU(0.5)^2** is worth roughly another ~0.002 BPB on a closely related stack, even before TTT.
- Earlier records repeatedly show that **cheap lexical memory** such as BigramHash is a strong use of bytes, and that **compression-aware export** is often the bottleneck.

So this candidate tries to trade a bit of redundant late-layer KV capacity for a bigger default bigram table and a smaller compressed artifact, rather than spending the budget on more training-only complexity.

## Prior records and candidates that influenced it

There were **no prior `candidates/` directories** in the repository when this candidate was created.

The main repository influences were:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`

In short: the code base is the 2026-03-22 training/export recipe, the carried-forward activation comes from 2026-03-23, and the byte reallocation target comes from the repo's repeated BigramHash wins.

## External research that informed it

This candidate is mainly inspired by three papers:

- **ALBERT** (Lan et al., 2019, arXiv:1909.11942): cross-layer parameter sharing can preserve most of the modeling benefit of depth while reducing parameter count and memory.
- **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** (Ainslie et al., 2023, arXiv:2305.13245): reducing KV redundancy is often a much safer trade than reducing query capacity.
- **DeepSeek-V2** (DeepSeek-AI, 2024, arXiv:2405.04434): Multi-head Latent Attention shows that the KV pathway can be compressed very aggressively while retaining strong language modeling quality.

This implementation is intentionally much simpler than MLA. It uses those papers as evidence that **KV representations are a good compression target** in compact decoder-only models.

## What changed versus the chosen base implementation

Base: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

Changes in this candidate:

- The script now resolves default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root, so it can be run directly from this candidate directory.
- The MLP activation is changed from `relu^2` to **LeakyReLU(0.5)^2**.
- The last four layers share `c_k` and `c_v` weights in **paired late-KV banks** by default.
- Int6 export is now **alias-aware**: shared parameters are detected and stored only once, with lightweight alias metadata for round-trip restoration.
- The default `BIGRAM_VOCAB_SIZE` is increased from `2048` to **`3072`** to spend some of the saved artifact budget on a bigger cheap lexical prior.
- Logging now reports both `shared_kv_groups` and the number of shared parameter aliases found during export.

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603311600_paired-kv-banks
RUN_ID=paired_kv_banks torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The main knobs for ablations are:

```bash
SHARED_KV_START=7
SHARED_KV_BANKS=2
BIGRAM_VOCAB_SIZE=3072
MLP_LEAKY_SLOPE=0.5
```

Set `SHARED_KV_BANKS=0` to disable the new sharing idea while keeping the rest of the candidate stack intact.

## Validation

The following lightweight validation was run in this workflow:

```bash
python -m compileall candidates/202603311600_paired-kv-banks/train_gpt.py
```

Outcome: **passed**.

I also attempted a CPU-only import/forward smoke test using a temporary `flash_attn_interface` stub so the shared-KV export round-trip could be exercised without GPUs. That smoke test was **not feasible in this runner** because the available Python interpreter does not have `torch` installed, and I did not install heavyweight dependencies just to simulate the environment.

## Main expected risks and tradeoffs

- Sharing late-layer KV projections may over-regularize the deepest layers and erase some of the gains from XSA/partial-RoPE specialization.
- The larger BigramHash table is only a good trade if the alias-aware export really converts shared weights into meaningful compressed-size savings.
- This is a deliberately minimal MLA-adjacent experiment, not a full latent-attention redesign, so the upside may be smaller than a more invasive architecture change.

## Suggested next experiments

If this candidate is directionally positive, the next ablations I would run are:

1. `SHARED_KV_BANKS=1` versus `2` versus `0`.
2. `BIGRAM_VOCAB_SIZE=3072` versus `4096` once artifact size is measured.
3. Sharing only `K` or only `V`, to see which side carries more useful late-layer specialization.
4. Applying the same alias-aware export idea to other explicitly shared modules if future candidates tie additional blocks.
