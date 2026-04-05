# Candidate: Shared Middle Block Cores + Bigram3072

## Hypothesis

The strongest clean pre-TTT line in this repo is already deep, thin, and compression-aware, but it still pays full parameter cost for every transformer block. Recent small-model work suggests selective block sharing can improve parameter efficiency without giving up depth. This candidate shares only the heavy **middle block cores** (attention + MLP weights), keeps the cheap per-layer adapters unique, leaves the late **XSA** and **value-embedding** layers untied, and spends part of the recovered budget on a larger **3072-bucket BigramHash** table.

## Why this is promising for this repository

- The record progression consistently rewards **deep/thin stacks**, **BigramHash**, **EMA**, **XSA**, **Partial RoPE**, and **GPTQ-lite** style compression-aware export.
- No reviewed record uses real **cross-layer / block weight sharing**.
- The non-record RTX 5090 study found full recurrence to be a bad direction, so this candidate uses a milder **MobileLLM-LS-style immediate sharing** only in the middle of the stack instead of turning the whole model into a recurrent block.
- Keeping the last 4 XSA layers and VE layers 9-10 unique preserves the parts that current best runs seem to depend on most.

## Records and prior work that influenced this candidate

- **Primary base:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - strongest non-TTT stack in the repo,
  - already includes EMA, GPTQ-lite export, Partial RoPE, XSA, SmearGate, BigramHash, and shared value embeddings.
- **Activation + lexical headroom influence:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - LeakyReLU(0.5)^2 improved pre-TTT quality,
  - its ablation also showed a useful gain from raising BigramHash capacity.
- **Constraint to keep late layers unique:** `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - reinforced that the late XSA / Partial RoPE stack is worth preserving.
- **Prior candidates:** there was no existing `candidates/` directory at review time, so this is the first candidate folder.

## External research that informed it

- **MobileLLM / MobileLLM-LS** ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905)): argues that sub-billion LMs benefit from **deep-and-thin architectures**, **embedding sharing**, **GQA**, and an **immediate block-wise weight-sharing** variant with only marginal latency overhead.
- **ALBERT** ([arXiv:1909.11942](https://arxiv.org/abs/1909.11942)): classic evidence that **cross-layer parameter sharing** can cut memory/parameter cost while retaining useful depth.
- **Universal Transformer** ([arXiv:1807.03819](https://arxiv.org/abs/1807.03819)): motivates **shared transition functions over depth** as a way to keep iterative refinement without paying for every layer independently.

## What changed versus the chosen base implementation

1. The model now keeps **11 logical layers** but uses only **8 physical block cores** by default with:
   - `SHARED_BLOCK_GROUPS="1,2;3,4;5,6"`
2. Each logical layer is split into:
   - a shared **`BlockCore`** containing the expensive attention + MLP weights,
   - a unique **layer adapter** containing norms, residual mixing, and learned scaling.
3. The shared groups are explicitly prevented from touching:
   - the final **XSA** layers,
   - the **VE** layers listed in `VE_LAYERS`.
4. Default `BIGRAM_VOCAB_SIZE` is increased from **2048 -> 3072**.
5. The MLP activation defaults to **LeakyReLU(0.5)^2** instead of ReLU^2.
6. `flash_attn_interface` is now optional; if unavailable, attention falls back to PyTorch SDPA.
7. `LATE_QAT_THRESHOLD` defaults to **0.0** in this candidate so the experiment does not rely on brittle compile-time fake-quant toggling; the main compression path is still the base script's **GPTQ-lite/int6 export**.

## How to run / evaluate

From this candidate directory:

```bash
cd candidates/202604052019_shared-core-bigram3072
RUN_ID=shared_core_bigram3072 \
DATA_PATH=../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful end-of-run log lines:

- `DIAGNOSTIC post_ema ...`
- `final_int6_roundtrip_exact ...`
- `final_int6_sliding_window_exact ...`
- `final_int6_sliding_window_s64_exact ...`

## Main risks / tradeoffs

- The middle stack may still need more layer-specific freedom than these shared cores allow, especially across the encoder/decoder boundary.
- The bigger bigram table spends some of the recovered bytes on lexical capacity; if that trade is wrong, the candidate could be worse than the base despite better parameter efficiency.
- Because the shared cores include attention `q_gain` and MLP weights, each shared group may converge to an average role rather than two specialized ones.
- If this underperforms, the next follow-ups should try:
  - sharing only one or two middle pairs,
  - sharing only MLP cores,
  - or reintroducing late fake-quant only after verifying a compile-safe switch.

## Validation run in this workflow

- `python -m compileall candidates/202604052019_shared-core-bigram3072/train_gpt.py` — **passed**
- A runtime smoke test was **not feasible in this workflow environment** because the Python runtime here does not have the repo's training dependencies installed (`torch`, `numpy`, and `sentencepiece` were all missing), so `train_gpt.py` could not be started meaningfully without additional setup.
