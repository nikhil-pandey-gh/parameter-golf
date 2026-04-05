# Shared MLP Banking + FP16 Embedding Reinvestment

## Hypothesis

The current 11-layer EMA + GPTQ-lite stack is already strong on raw training quality, but the repo history says post-training compression still decides a lot of the final score. Sharing only the **middle-layer MLPs** should cut artifact bytes without adding recurrence-time compute, and those bytes can be reinvested into two knobs that already looked promising in-repo: a **larger BigramHash table** and **FP16 tied-embedding export**.

In short: keep attention unique, share only the least position-specific MLPs, and spend the saved bytes on tensors that are most sensitive to quantization.

## Why this looks promising here

Several existing records point in the same direction:

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/` is the best non-TTT pure-training base in the repo and already shows that quantization-aware export work still buys real BPB.
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/` showed that the tied embedding is unusually precision-sensitive.
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/` and `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/` both found that larger BigramHash tables can still help.
- `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/` found that **looped recurrence** regressed badly in this challenge, which makes **static parameter sharing** more attractive than extra unrolled compute.
- `records/track_non_record_16mb/2026-03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3/` still paid a large post-quant penalty after much more training, which reinforces that artifact shaping remains important.

## Prior records that influenced this candidate

1. **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`
2. **Activation choice:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md`
3. **Embedding precision motivation:** `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600/README.md`
4. **Bigram scaling motivation:** `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md`

There were no prior `candidates/` directories in this repository when this candidate was created.

## External research that informed it

- **ALBERT** (cross-layer parameter sharing): https://arxiv.org/abs/1909.11942
- **Universal Transformer** (iterative layer reuse / shared depth): https://arxiv.org/abs/1807.03819
- **Looped Transformers** (modern evidence that carefully reused depth can stay competitive): https://arxiv.org/abs/2409.15647
- Secondary quantization motivation:
  - **QuaRot**: https://arxiv.org/abs/2404.00456
  - **SpinQuant**: https://arxiv.org/abs/2405.16406

The implemented idea is closer to ALBERT-style sharing than to explicit recurrence: no extra unrolled steps, just fewer unique middle MLP weights.

## What changed versus the chosen base

Compared with `2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`, this candidate:

1. switches the MLP activation to **LeakyReLU(0.5)^2** by default;
2. raises the default **BigramHash** table from `2048` to `4096`;
3. shares **middle-layer MLPs** over a 3-bank cycle by default:
   - `MLP_SHARE_START=3`
   - `MLP_SHARE_END=8`
   - `MLP_SHARED_BANKS=3`
4. teaches the mixed int6 exporter to **deduplicate aliased shared weights**, so sharing actually reduces artifact bytes instead of only reducing optimizer parameters;
5. exports the tied embedding in **FP16** by default (`FP16_EMBED_EXPORT=1`);
6. resolves the default dataset/tokenizer paths relative to the script location so `train_gpt.py` can be launched from inside this candidate directory.

## How to run

From the candidate directory:

```bash
cd candidates/202604050601_shared-mlp-banking
SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Key defaults already baked in:

- 11 layers, 512 dim, 8 heads / 4 KV heads
- seq len 2048
- XSA on last 4 layers
- partial RoPE (`ROPE_DIMS=16`)
- LN scaling
- EMA + SWA + GPTQ-lite style per-row clip search
- LeakyReLU(0.5)^2 MLPs
- middle-layer MLP sharing
- BigramHash 4096
- FP16 tied-embedding export

To disable the new sharing idea for ablations:

```bash
MLP_SHARED_BANKS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## How to evaluate

The script keeps the base record flow: it trains, exports the compressed artifact, reloads the dequantized model, and prints both roundtrip and sliding-window validation BPB. The most important final line remains:

```bash
final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...
```

That legacy log label is still emitted by the base script even though the candidate uses mixed int6 export.

## Main risks and tradeoffs

1. **Underfitting risk:** sharing MLPs may over-regularize the middle of the network and erase more capacity than the larger bigram table gives back.
2. **Interaction risk:** LeakyReLU^2, FP16 embedding export, and MLP sharing are individually motivated, but their combined effect has not been measured in this repo yet.
3. **Export-path risk:** the byte savings depend on alias-aware export; if a later refactor materializes copied tensors instead of true shared modules, the savings disappear.
4. **Challenge-fit risk:** this avoids recurrence-time compute, but it still changes the model’s inductive bias more than a pure quantization tweak would.

## Validation

Commands run in this environment:

```bash
python -m compileall candidates/202604050601_shared-mlp-banking/train_gpt.py
```

Outcome:

- passed

Attempted smoke check:

- A minimal CPU harness was attempted, but this container does not have `torch` installed for either `python` or `python3`, so a real model-start smoke test was not feasible here (`ModuleNotFoundError: No module named 'torch'`).
