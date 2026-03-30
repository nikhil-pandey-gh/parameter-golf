# Single-Head MTP on the LeakyReLU² + TTT Stack

## Hypothesis

Add a **train-only single-head multi-token prediction (MTP)** auxiliary loss to the strongest existing stack so the shared trunk learns a slightly richer "look one step farther" objective during the fixed 10-minute training window.

The expected upside is **better sample efficiency and stronger induction/lookahead behavior** with effectively **zero artifact cost**, because the auxiliary head is excluded from export and never used for evaluation.

## Why this is promising for this repository

The repo has already pushed hard on the obvious levers:

- bigger-but-compressible models,
- quantization/export quality,
- EMA/SWA,
- partial RoPE,
- XSA,
- SmearGate + BigramHash,
- and legal TTT.

That makes the **training objective** one of the cleanest remaining places to add signal without changing the artifact budget or the evaluator.

This candidate is especially attractive here because the `2026-03-23_LeakyReLU_LegalTTT_ParallelMuon` trainer already contained a dormant MTP path, but the auxiliary heads were **not actually wired into any optimizer**. The code could compute an auxiliary loss, but the MTP head weights never stepped. This candidate turns that path into a real experiment.

## Prior records and candidates that influenced this

There were no prior `candidates/` directories in the repo when this was created.

The main record influences were:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/`
  - strongest overall stack in the repo,
  - source of the dormant MTP code path,
  - provides the LeakyReLU(0.5)^2 + Parallel Muon + legal TTT base.

- `records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/`
  - reinforces that **small train/export improvements still matter** on top of the mature 11-layer stack,
  - useful reminder that the best ideas here often stack rather than replace the current recipe.

- `records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/`
  - explicit example of a feature flag that looked live but was actually dead under `torch.compile`,
  - motivated verifying that the new auxiliary path is truly wired into optimization.

## External research that informed it

- Fabian Gloeckle et al., **"Better and Faster Large Language Models via Multi-Token Prediction"**, arXiv:2404.19737
  - argues that predicting multiple future tokens with independent heads improves sample efficiency and induction-style behavior.

- Guoliang Zhao et al., **"Self-Distillation for Multi-Token Prediction"**, arXiv:2603.23911
  - shows that even **1-head MTP** is practically useful and can preserve main-head quality with low added cost.

- Lorenzo Noci et al., **"Thinking into the Future: Latent Lookahead Training for Transformers"**, arXiv:2603.20219
  - supports the broader idea that supervising limited future lookahead can improve hard next-token decisions.

## What changed versus the chosen base implementation

Base fork:

- `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py`

Changes in this candidate:

1. **Enabled conservative MTP defaults**
   - `MTP_NUM_HEADS=1`
   - `MTP_LOSS_WEIGHT=0.10`

2. **Actually optimized the MTP head**
   - added each `mtp_head.weight` to the AdamW-managed small-parameter group,
   - so the auxiliary loss now updates the auxiliary head instead of being effectively dead.

3. **Baked in two record-faithful defaults from the base run**
   - `BIGRAM_VOCAB_SIZE=1536`
   - `TTT_FREEZE_BLOCKS=0`

4. **Left late QAT off by default**
   - `LATE_QAT_THRESHOLD=0.0`
   - keeps this candidate focused on the MTP change instead of relying on an inherited compiled late-QAT toggle.

5. **Kept export behavior unchanged**
   - `mtp_heads.*` are still excluded from the exported state dict,
   - the eval model is still reconstructed with `mtp_num_heads=0`,
   - so the final artifact cost stays aligned with the original stack.

## How to run or evaluate it

From this directory:

```bash
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The script resolves its default `DATA_PATH` and `TOKENIZER_PATH` back to the repository root, so these commands are intended to work when launched directly from `candidates/202603300616_single-head-mtp/`.

For leaderboard-style evaluation with legal TTT enabled:

```bash
SEED=1337 TTT_ENABLED=1 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Useful ablations:

```bash
# Turn MTP off to recover the non-MTP trunk behavior
MTP_NUM_HEADS=0 torchrun --standalone --nproc_per_node=8 train_gpt.py

# Keep MTP on but reduce its influence
MTP_NUM_HEADS=1 MTP_LOSS_WEIGHT=0.05 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Main expected risks and tradeoffs

- The auxiliary loss adds extra output-head compute during training, so **fewer steps** may offset any sample-efficiency gain.
- A single auxiliary future-token head may be too weak, while a heavier MTP setup may be too expensive for this 600-second regime.
- Because the auxiliary head is dropped at export, any improvement must transfer into the **shared trunk** rather than rely on the extra head at eval time.
- This candidate inherits the operational complexity of the `2026-03-23` base stack, including Parallel Muon and optional legal TTT.

## Validation

Commands run locally in this workflow:

```bash
python -m compileall train_gpt.py train_gpt_mlx.py \
  candidates/202603300616_single-head-mtp/train_gpt.py
```

Outcome:

- passed for all three files.

Attempted import smoke:

```bash
python - <<'PY'
import importlib.util
from pathlib import Path
path = Path('candidates/202603300616_single-head-mtp/train_gpt.py')
spec = importlib.util.spec_from_file_location('candidate_train', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
PY
```

Outcome:

- failed in this runner because the repository's Python dependencies are not installed here (`ModuleNotFoundError: No module named 'numpy'`).

Dependency-free structural check:

```bash
python - <<'PY'
import ast
from pathlib import Path
path = Path('candidates/202603300616_single-head-mtp/train_gpt.py')
text = path.read_text(encoding='utf-8')
tree = ast.parse(text, filename=str(path))
assert "REPO_ROOT = Path(__file__).resolve().parents[2]" in text
assert 'str(REPO_ROOT / "data/datasets/fineweb10B_sp1024")' in text
assert 'str(REPO_ROOT / "data/tokenizers/fineweb_1024_bpe.model")' in text
values = {}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name == 'Hyperparameters':
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                name = stmt.targets[0].id
                if name not in {'mtp_num_heads', 'mtp_loss_weight', 'bigram_vocab_size', 'ttt_freeze_blocks', 'late_qat_threshold'}:
                    continue
                outer = stmt.value
                inner = outer.args[0]
                values[name] = ast.literal_eval(inner.args[1])
assert values == {
    'mtp_num_heads': 1,
    'mtp_loss_weight': 0.1,
    'bigram_vocab_size': 1536,
    'ttt_freeze_blocks': 0,
    'late_qat_threshold': 0.0,
}
assert 'for mtp_head in base_model.mtp_heads:' in text
assert 'scalar_params.append(mtp_head.weight)' in text
assert 'export_sd = {k: v for k, v in full_state_dict.items() if "mtp_heads" not in k}' in text
print(values)
PY
```

Outcome:

- passed; confirmed the MTP defaults, repo-root data/tokenizer path resolution, optimizer wiring, and export exclusion logic.

A true CPU forward-pass smoke test was **not feasible** in this workflow because:

- the runner does not have the repo's Python deps installed, and
- the real training/eval path is CUDA-centric and built around the existing FlashAttention-based stack.
