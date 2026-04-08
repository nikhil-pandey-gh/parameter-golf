# Mirror-Shared MLP + BigramHash(3072)

## Hypothesis

The current 11-layer record line looks more artifact-limited than FLOP-limited: recent wins mostly come from compression, evaluation, or near-zero-parameter tweaks. This candidate tests whether **ALBERT-style cross-layer sharing**, applied conservatively to only the heavy MLP weights, can free enough serialized bytes to reallocate capacity into the already successful hashed lexical path without paying the step-time penalty that hurt naive recurrence.

Concretely, the model mirror-shares MLP weights across the U-Net encoder/decoder pairs while keeping per-layer attention, norms, scales, residual mixing, skip weights, and value-embedding controls untied. The saved bytes are spent on a larger **BigramHashEmbedding** (`2048 -> 3072` buckets).

## Why this is promising here

1. Recent records improved by shrinking quantization error and reallocating bytes, not by adding expensive new infrastructure.
2. The non-record 1x5090 exploration showed that naive recurrent depth was harmful under a fixed wall-clock budget; this candidate avoids extra forward passes and instead shares weights across the existing 11-layer pass.
3. The latest record's ablation reported a gain from increasing `BigramHash` capacity, so using sharing to fund a larger bigram table is directly aligned with repository evidence.

## Influential prior experiments

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - chosen base implementation because it is the strongest pre-TTT stack in-repo and already combines EMA, XSA, Partial RoPE, LN scale, VE, and GPTQ-lite export.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - influenced the decision to treat `BigramHash` capacity as a real lever; its README includes a positive `2048 -> 3072` bigram ablation.
- `records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd/`
  and `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
  - established the broader SmearGate + bigram/local-bias direction that this candidate extends.

No earlier `candidates/` directory existed, so there were no prior candidate folders to avoid duplicating.

## External research grounding

- **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations**  
  Lan et al., 2019 — <https://arxiv.org/abs/1909.11942>  
  Main takeaway used here: cross-layer parameter sharing can cut large matrix footprints substantially while preserving layer-specific behavior through untied normalization/control parameters.

- **Universal Transformers**  
  Dehghani et al., 2018 — <https://arxiv.org/abs/1807.03819>  
  Main takeaway used here: repeated transformation structure can be effective when the recurrent/shared component is wrapped in layer- or step-specific state updates instead of making every layer fully identical.

I considered embedding factorization from ALBERT as well, but repository history suggests the tied embedding/output path is unusually sensitive, so this candidate keeps embeddings unchanged and only shares MLP weights.

## What changed vs. the base implementation

Starting from `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes two deliberate changes:

1. **Mirror-shared MLP bank**
   - adds `SHARED_MLP_MODE` (default `mirror`);
   - stores MLP weights once in `self.shared_mlps`;
   - maps the 11 logical blocks to the shared bank with the default pattern:
     - encoder: `[0, 1, 2, 3, 4]`
     - decoder: `[4, 3, 2, 1, 0, 5]`
   - keeps attention blocks, layer scales, norms, residual mixing, skip weights, and VE controls per-layer.

2. **Larger lexical hash memory**
   - raises the default `BIGRAM_VOCAB_SIZE` from `2048` to `3072`.

The key serialization detail is that the shared MLP weights live outside `self.blocks`, so they appear only once in the `state_dict` and therefore only once in the quantized export.

## How to run

From this directory:

```bash
RUN_ID=mirror_mlp_bigram3072 \
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=3072 SHARED_MLP_MODE=mirror XSA_LAST_N=4 \
SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
ITERATIONS=9000 TRAIN_BATCH_TOKENS=786432 TRAIN_SEQ_LEN=2048 \
MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For an ablation against the same codepath without sharing, set `SHARED_MLP_MODE=none`. EMA remains enabled by default in this base codepath, matching the chosen source implementation.

## Evaluation notes

- The script retains the base record's int6 mixed quantization path, EMA application, and stride-64 sliding-window evaluation.
- Expect the main tradeoff to show up in post-quant bytes and not necessarily in pre-quant loss alone.

## Risks and tradeoffs

1. MLP sharing may over-regularize the model and reduce the value of deeper decoder specialization.
2. The extra bigram capacity could overfit local statistics if the saved bytes were better spent elsewhere (for example, more VE capacity or larger untied MLPs).
3. Because only the MLP stack is shared, the candidate may land in an awkward middle ground: not enough sharing to buy a large artifact win, but enough to hurt optimization.
4. This keeps the CUDA + FlashAttention 3 dependency profile of the chosen base implementation.

## Validation

Commands run in this environment:

- `python -m compileall candidates/202604080307_mirror-mlp-bigram3072/train_gpt.py`

Outcome:

- syntax compilation passed.
- code review completed cleanly with no substantive findings.
- A CPU smoke test was not feasible in this environment because the available runtime does not include the required PyTorch/FlashAttention stack, and this script intentionally preserves the record codepath's CUDA-specific training setup.
