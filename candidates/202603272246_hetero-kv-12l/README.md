# Heterogeneous KV Heads + 12 Layers

## Hypothesis

A tiny model under a hard artifact cap should not spend the same number of key/value heads in every layer.

This candidate keeps richer grouped-query attention in early layers, tapers to fewer KV heads in deeper layers, and uses the saved attention parameters to fund a 12th transformer block. The working assumption is that early layers benefit more from richer attention structure, while later layers can tolerate MQA-like compression if the model keeps the rest of the strong 11-layer record stack intact.

## Why this looks promising for this repo

The repository history shows a consistent pattern: more useful depth, stronger MLPs, and better compression-aware design keep winning, while broad architecture rewrites and expensive recurrence have been much riskier.

Relevant repo evidence:

- `records/track_10min_16mb/2026-03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271/README.md` showed that moving from smaller stacks to an 11-layer / 3x-MLP / EMA / XSA configuration was a real step forward.
- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md` showed that cheap, zero- or near-zero-byte architectural refinements still matter on top of that 11-layer base.
- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md` established the strongest non-TTT training stack in this repo and is the direct code base for this candidate.
- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` showed that LeakyReLU(0.5)^2 is a meaningful MLP upgrade, so this candidate carries that activation forward even though the main new idea is the heterogeneous KV schedule.

## Prior work that influenced this candidate

### Repository influences

- Base implementation: `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
- Activation choice: `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
- There were no prior `candidates/` directories in the repository when this candidate was created.

### External research influences

- **Multi-Query Attention**: Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (`arXiv:1911.02150`) argues that sharing K/V across heads can preserve quality with only minor degradation while reducing K/V cost.
- **Grouped-Query Attention**: Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints* (`arXiv:2305.13245`) shows that intermediate KV-head counts can keep quality close to full multi-head attention while retaining much of MQA's efficiency.
- **Transformer compression survey**: Tang et al., *A Survey on Transformer Compression* (`arXiv:2402.05964`) is a useful framing reference for using architectural efficiency, not just quantization, as part of a compact-model design.

## What changed versus the chosen base implementation

Compared with `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate makes four targeted changes:

1. **12 layers by default**
   - `NUM_LAYERS` now defaults to `12` instead of `11`.

2. **Per-layer KV-head schedule**
   - Added `KV_HEADS_SCHEDULE` with default:
     - `4,4,4,2,2,2,2,1,1,1,1,1`
   - Early layers keep fuller GQA capacity.
   - Mid layers taper to 2 KV heads.
   - Deep layers switch to MQA-style 1 KV head.
   - The schedule is parsed once and wired block-by-block, so each attention layer can have its own KV width.

3. **LeakyReLU(0.5)^2 MLPs**
   - The MLP activation is changed from ReLU^2 to LeakyReLU(0.5)^2.
   - This follows the repo's latest activation win while keeping the rest of the MLP structure unchanged.

4. **Portability / smoke-test support**
   - Defaults for `DATA_PATH` and `TOKENIZER_PATH` now resolve relative to the repository root inferred from `__file__`, so the script can be launched from this candidate directory.
   - `flash_attn_interface` is optional; the script falls back to PyTorch SDPA when FlashAttention is unavailable.
   - Added `CPU_SMOKE_TEST=1` for a no-dataset synthetic forward/backward smoke run.

What stays the same from the base stack:

- SmearGate + BigramHash input enrichment
- partial RoPE
- LN scale
- XSA on the last 4 layers
- shared value embeddings on the deepest layers
- EMA + tight SWA behavior
- GPTQ-lite-style int6 export path
- sliding-window evaluation support

## How to run or evaluate it

From this candidate directory:

```bash
cd candidates/202603272246_hetero-kv-12l

RUN_ID=hetero_kv_12l \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful overrides:

```bash
KV_HEADS_SCHEDULE=4,4,4,4,2,2,2,2,1,1,1,1
NUM_LAYERS=12
VE_LAYERS=10,11
LEAKY_RELU_SLOPE=0.5
```

CPU-only synthetic smoke check:

```bash
CPU_SMOKE_TEST=1 python train_gpt.py
```

## Validation run in this workflow

Baseline repository syntax check:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py data
```

Outcome: succeeded.

Candidate syntax check:

```bash
python -m compileall candidates/202603272246_hetero-kv-12l/train_gpt.py
```

Outcome: succeeded.

Candidate CPU smoke check executed from inside this candidate directory in a temporary venv (the workflow runner did not have the repo's Python dependencies preinstalled in the system interpreter):

```bash
cd candidates/202603272246_hetero-kv-12l
CPU_SMOKE_TEST=1 /tmp/gh-aw/agent/pg-venv/bin/python train_gpt.py
```

Outcome:

```text
cpu_smoke_ok loss:6.9003
```

## Main expected risks / tradeoffs

- **Step-time risk**: the 12th layer increases compute even though the KV schedule reduces attention parameter count.
- **Capacity-allocation risk**: moving deep layers to 1 KV head may save too aggressively and partially erase the benefit of the added depth.
- **Value-embedding interaction risk**: value embeddings are sliced to match each layer's KV width; that keeps the implementation small, but it may not be the optimal way to mix VE with heterogeneous KV heads.
- **No full GPU train/eval run in this workflow**: this candidate was syntax-checked and smoke-tested, but not benchmarked end-to-end on challenge hardware here.

## Suggested next experiments

1. Sweep schedules that keep one more 2-KV layer before switching to 1-KV.
2. Compare `12L + heterogeneous KV` against `11L + larger MLP` using the same schedule idea.
3. If 1-KV deep layers are too lossy, try `4,4,4,2,2,2,2,2,1,1,1,1`.
4. If step time is too high, keep the KV schedule but revert to 11 layers to isolate whether the gain comes from better parameter placement alone.
