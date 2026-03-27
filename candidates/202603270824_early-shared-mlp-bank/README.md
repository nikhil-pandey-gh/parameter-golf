# Early Shared MLP Bank

## Hypothesis

The current static 11-layer stack is already strong on optimization, evaluation, and quantization, but it still spends a large fraction of its artifact budget on repeated MLP weights. This candidate tests whether **pairwise sharing the early-layer MLPs**, while keeping per-layer norms, residual mixing, attention, and scaling untied, can improve parameter efficiency enough to fund a larger token-pair memory without giving up too much modeling capacity.

Concretely, the candidate shares the MLP sublayer across the first 8 layers in groups of 2, keeps attention unique in every layer, and reinvests some of the saved artifact budget into a larger `BigramHash` table.

## Why this is promising for this repository

The repository history is dominated by three trends:

- the best static stacks keep finding small gains from better use of the 16 MB artifact budget,
- `BigramHash` repeatedly helps when capacity can be afforded,
- no prior record or candidate in this repo appears to have tried serious cross-layer parameter sharing.

That makes selective sharing a good next bet here. The idea is most attractive in this challenge because the target metric is **compressed artifact quality**, not raw parameter count alone.

## Prior repository work that influenced this candidate

This implementation is based directly on:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`

That base already contains the strongest non-TTT static recipe I found in the repo:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- U-Net skips
- XSA on the last 4 layers
- partial RoPE + LN scaling
- shared value embeddings on top layers
- SmearGate + BigramHash
- EMA + tight SWA
- GPTQ-lite int6 export + sliding-window evaluation

Additional repo evidence that informed this candidate:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` showed `LeakyReLU(0.5)^2` is a cheap, real gain on the modern stack.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md` showed that spending saved artifact budget on a larger `BigramHash` table is worthwhile.
- Across the full `records/` history, I did not find a serious weight-sharing or recurrent-depth attempt on the modern 11-layer family.

## External research that informed the design

- **ALBERT** (`arXiv:1909.11942`) motivates cross-layer parameter sharing as a practical way to reduce transformer parameter count while preserving useful depth.
- **Universal Transformer** (`arXiv:1807.03819`) motivates depth reuse as a way to combine recurrent inductive bias with transformer-style parallel sequence processing.
- **Intra-Layer Recurrence in Transformers for Language Modeling** (`arXiv:2505.01855`) is especially relevant here: it argues that selective recurrence is better than indiscriminate whole-block repetition, and reports that extra reuse in earlier layers is the most promising regime.
- **A Survey on Transformer Compression** (`arXiv:2402.05964`) reinforces the high-level strategy: architecture-level compression and quantization should be designed together rather than treated as separate phases.

This candidate translates those ideas into a repo-friendly change: **share only the heaviest sublayer (the MLP), and only in the earlier half of the network**, while preserving the rest of the stack that the records already validated.

## What changed versus the chosen base implementation

Relative to `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. switches the MLP activation from `relu^2` to `LeakyReLU(0.5)^2`,
2. shares the first 8 layer MLPs in pairs (`0-1`, `2-3`, `4-5`, `6-7`),
3. keeps all attention modules, norms, residual mix parameters, and layer scales untied,
4. increases `BIGRAM_VOCAB_SIZE` from `2048` to `4096`,
5. resolves default `DATA_PATH` and `TOKENIZER_PATH` relative to the repository root so the script can be run from this candidate directory,
6. keeps `final_model.pt` directly reloadable, while deduplicating shared-weight aliases only for the compressed export path and reconstructing them during the int6 reload path so shared MLPs actually save artifact bytes.

## How to run or evaluate

From the candidate directory:

```bash
cd candidates/202603270824_early-shared-mlp-bank
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
SEED=1337 \
SHARED_MLP_EARLY_LAYERS=8 \
SHARED_MLP_GROUP_SIZE=2 \
BIGRAM_VOCAB_SIZE=4096 \
LEAKY_RELU_SLOPE=0.5 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Because this script resolves dataset paths relative to the repository root, it can be launched from inside this candidate folder without extra path arguments as long as the standard dataset cache lives under `repo_root/data/`.

## Main expected risks and tradeoffs

- Sharing early MLPs may remove too much layer-specific capacity, especially if the bigger `BigramHash` table does not pay back the saved bytes.
- The savings are concentrated in artifact size, not training FLOPs, so this is primarily a parameter-efficiency bet rather than a throughput bet.
- Export/reload is slightly more complex than the base because shared tensors must be deduplicated for compression and then expanded again before loading.
- I did not validate a real CUDA run in this environment, so the hypothesis is implemented and syntax-checked, but not benchmarked.

## Validation run here

Successful:

- `python -m compileall candidates/202603270824_early-shared-mlp-bank/train_gpt.py`

Attempted but blocked by environment:

- A CPU-only import/forward smoke test was attempted with a stubbed `flash_attn_interface`, but this runner does not have `torch`, `numpy`, or `sentencepiece` installed in its Python environment.
- That means the exact risky path introduced here — `state_dict()` dedupe for the compressed artifact, alias expansion on reload, and `load_state_dict()` into the shared-MLP model — could not be executed in this runner.
- Validation command used to confirm that limitation:

```bash
python - <<'PY'
import importlib.util
mods = ['torch', 'numpy', 'sentencepiece']
print({m: bool(importlib.util.find_spec(m)) for m in mods})
PY
```

Observed result:

```python
{'torch': False, 'numpy': False, 'sentencepiece': False}
```

So the lightweight runtime smoke test was **not feasible here** without installing the full training stack.
