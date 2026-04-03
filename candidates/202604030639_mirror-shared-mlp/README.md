# Mirror-Shared MLP

## Hypothesis

The strongest next low-infrastructure bet is to keep the current 11-layer attention stack mostly intact, but replace the per-layer MLP weights with **mirror-shared MLP banks** across the U-Net-shaped encoder/decoder pairs. This should cut a large fraction of the repeated feed-forward weights while preserving layer-specific attention, and the saved capacity can be spent on a slightly richer lexical side path without breaking the 10-minute / 16MB regime.

Concretely, this candidate ties the MLP weights for layers `(0,10) (1,9) (2,8) (3,7) (4,6)` and keeps layer `5` unique. Each layer still has its own learned hidden/output modulation vectors, residual scales, norms, and attention block.

## Why it is promising here

- The repository's best **core** stack is the 11-layer EMA + GPTQ-lite + warmdown3500 line, which already extracted most of the obvious wins from quantization, XSA, partial RoPE, LN scale, and sliding evaluation.
- The repo also contains a clear warning that **naive full layer recurrence** was bad in a 10-minute budget because it paid extra step-cost for every reused layer.
- This candidate avoids that failure mode by keeping the same 11 forward passes and unique attention layers while sharing only the largest repeated MLP matrices.
- The latest top record also shows that **LeakyReLU(0.5)^2** is a cheap orthogonal MLP improvement, so this candidate carries that activation into the shared-MLP design.

## Prior repository work that influenced this candidate

- **Base implementation:** `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - Best clean train/export stack without depending on heavy eval-time TTT.
- **Activation carry-over:** `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - Shows LeakyReLU(0.5)^2 is a meaningful low-cost MLP improvement.
- **Local dead-end informing the design:** `records/track_non_record_16mb/2026-03-19_SwiGLU_WarmdownFix_QuarterBatch_1x5090/`
  - Reported that naive layer recurrence was net-negative under fixed wallclock.

There were **no prior `candidates/` directories** in the repository when this candidate was created.

## External research that informed it

- **ALBERT** (Lan et al., 2019, arXiv:1909.11942): strong evidence that cross-layer parameter sharing can improve parameter efficiency without collapsing model quality.
- **Universal Transformer** (Dehghani et al., 2018, arXiv:1807.03819): motivates recurrent-depth style reuse as an inductive bias, but with an explicit warning that reuse changes the compute/expressivity trade-off.
- **Subformer** (Reid et al., 2021, arXiv:2101.00234): especially relevant here because it argues that **sandwich-style sharing outperforms naive cross-layer sharing in generative transformers**.
- **Basis Sharing** (Wang et al., 2024, arXiv:2410.03765): suggests that shared bases with small layer-specific coefficients can outperform naive sharing under aggressive compression.
- **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach** (Geiping et al., 2025, arXiv:2502.05171): a recent signal that recurrent-depth ideas are still worth exploring, even if this repository needs a cheaper training-time approximation than a full recurrent-depth pretraining recipe.

This candidate is effectively a repository-constrained version of those ideas: **share the heavy MLP basis, keep the attention path unique, and restore layer specialization with tiny learned modulation vectors.**

## What changed versus the chosen base implementation

1. **Mirror-shared MLP bank**
   - The 11 per-layer MLPs are replaced by 6 unique MLP banks shared across mirrored layer pairs.
   - Each block keeps its own `shared_mlp_hidden_scale` and `shared_mlp_out_scale` vectors so sharing behaves more like a shared basis plus per-layer coefficients than strict tying.
2. **LeakyReLU(0.5)^2 activation**
   - Replaces ReLU^2 inside the shared MLPs.
3. **Runtime-safe late QAT toggle**
   - The base script used a class-level boolean that earlier repo analysis showed could be constant-folded away under `torch.compile`.
   - This candidate moves QAT enablement onto per-module buffers so the late-QAT switch remains a runtime value.
4. **BigramHash default**
   - Default `BIGRAM_VOCAB_SIZE` is increased from `2048` to `3072`, following the latest record's positive bigram ablation.

## How to run

From this candidate directory:

```bash
RUN_ID=mirror_shared_mlp \
DATA_PATH=../../data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../../data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful knobs:

```bash
MIRROR_SHARE_MLP=1        # default; set 0 to fall back to one MLP per layer
BIGRAM_VOCAB_SIZE=3072    # candidate default
QAT_ENABLED=0             # leave off initially; late QAT turns on near warmdown if enabled by threshold
LATE_QAT_THRESHOLD=0.15   # default candidate threshold
```

## Evaluation

The script preserves the base flow:

1. train under the wallclock cap,
2. apply EMA,
3. export a compressed int6 artifact,
4. reload it,
5. report roundtrip and sliding-window `val_bpb`.

## Main expected risks and tradeoffs

- **Too much sharing may reduce expressivity.** The per-layer modulation vectors may be too weak to fully recover the benefit of separate MLPs.
- **Step-time vs quality trade-off remains uncertain.** This candidate avoids adding more layers, but any quality gain must still outweigh the inductive bias change from tying.
- **LeakyReLU^2 and sharing may interact nonlinearly.** The activation helped in the top TTT-heavy stack, but its effect here is still unverified on the pure 3/22 base.
- **The larger BigramHash may be neutral or slightly negative** if the saved artifact budget should instead have been reinvested elsewhere.

## Validation

Commands run during candidate creation:

```bash
python -m compileall candidates/202604030639_mirror-shared-mlp/train_gpt.py
python -m compileall train_gpt.py train_gpt_mlx.py data
```

Outcomes:

- `compileall` succeeded for the candidate script.
- The repository baseline `train_gpt.py`, `train_gpt_mlx.py`, and `data/` also compiled cleanly before the candidate changes.
- A true CPU smoke run was **not feasible** in this environment without changing the submission path, because the script is intentionally CUDA + FlashAttention based and the environment does not provide the training dependencies needed for import-and-run execution.
