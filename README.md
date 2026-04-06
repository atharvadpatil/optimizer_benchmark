# Do LLM-Era Optimizers Transfer to Small-Scale Training?

A controlled benchmark of 7 optimizers — SGD+Momentum, AdamW, Lion, AdEMAMix, SOAP, Muon, and Schedule-Free AdamW — at small scale (ResNet-18 on CIFAR-10, 2-layer GPT on Penn Treebank) to test whether gains reported at 124M–720M parameters transfer to models under 15M parameters.

## Research Questions

- **RQ1**: Do optimizer rankings from large-scale LLM training hold at small scale?
- **RQ2**: Does AdEMAMix's gradient-history advantage activate within short training horizons (~20K steps)?
- **RQ3**: How does per-step compute overhead (SOAP, Muon) trade off against accuracy at small batch sizes?
- **RQ4**: Which optimizers are most robust to hyperparameter choices?
- **RQ5**: Do architecture-specific optimizers (Muon, SOAP) behave differently on CNNs vs Transformers?

## Project Structure

```
optimizer_benchmark/
├── configs/grid.yaml            # Hyperparameter grids (3 LR × 3 WD per optimizer)
├── models/
│   ├── resnet_cifar.py          # CIFAR-adapted ResNet-18 (~11.2M params)
│   └── gpt_small.py            # 2-layer GPT (~4.2M params)
├── data/
│   ├── cifar10.py               # CIFAR-10 loaders with standard augmentation
│   └── ptb.py                   # Penn Treebank word-level loader (auto-downloads)
├── trainers/
│   ├── device.py                # Auto-detect MPS > CUDA > CPU, seed_everything()
│   ├── optimizer_factory.py     # Unified get_optimizer() for all 7 optimizers
│   └── trainer.py               # Training loops for classification and language modelling
├── scripts/
│   ├── run_single.py            # Run one experiment (for debugging)
│   ├── run_grid.py              # Run full HP grid with skip-if-exists
│   ├── run_diagnostics.py       # Horizon (RQ2) and batch-size (RQ3) diagnostics
│   └── analyze.py               # Generate all figures and tables from results/
├── results/                     # JSON logs (one file per run)
├── figures/                     # Generated plots
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11+
- macOS with Apple Silicon (MPS), or CUDA GPU, or CPU

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd optimizer_benchmark

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check device detection
python -c "from trainers.device import get_device; get_device()"

# Check all optimizers load
python -c "
import torch.nn as nn
from trainers.optimizer_factory import get_optimizer
model = nn.Linear(64, 10)
for name in ['sgd','adamw','lion','ademamix','soap','muon','schedulefree_adamw']:
    opt = get_optimizer(name, model, lr=1e-3, weight_decay=0.01)
    print(f'{name}: {opt.__class__.__name__}')
"
```

## Running Experiments

### 1. Smoke Test (single run)

Verify everything works end-to-end with a quick 3-epoch run:

```bash
python scripts/run_single.py \
    --task cifar10 \
    --optimizer adamw \
    --lr 3e-3 \
    --wd 0.05 \
    --epochs 3
```

For PTB:

```bash
python scripts/run_single.py \
    --task ptb \
    --optimizer adamw \
    --lr 3e-3 \
    --wd 0.05 \
    --max_steps 2000
```

### 2. Full Hyperparameter Grid

Each optimizer is evaluated over a 3×3 grid of learning rates and weight decays (9 runs per optimizer per task, 63 runs total per task). Grids are defined in `configs/grid.yaml`.

```bash
# Run all optimizers on CIFAR-10 (50 epochs each, ~21 GPU-hours)
python scripts/run_grid.py --task cifar10

# Run all optimizers on PTB (50K steps each, ~10 GPU-hours)
python scripts/run_grid.py --task ptb
```

**Run a subset of optimizers:**

```bash
python scripts/run_grid.py --task cifar10 --optimizers adamw lion sgd
```

**Resumability:** The grid runner checks for existing result files before each run. If you stop and restart, it picks up where it left off. Failed runs are saved as `_FAILED.json` and won't be retried.

### 3. Diagnostic Experiments

Diagnostics require completed main grid results (they read best hyperparameters from `results/`).

**Training Horizon (RQ2)** — tests AdEMAMix, AdamW, SGD at 10%/25%/50%/100% of training budget:

```bash
python scripts/run_diagnostics.py --task cifar10 --diagnostic horizon
```

**Batch Size Sensitivity (RQ3)** — tests all optimizers at batch sizes {32, 64, 128, 256} with fixed 20K steps:

```bash
python scripts/run_diagnostics.py --task cifar10 --diagnostic batchsize
```

**Run both:**

```bash
python scripts/run_diagnostics.py --task cifar10 --diagnostic all
```

### 4. Generate Figures and Tables

```bash
python scripts/analyze.py
```

This produces in `figures/`:

| Output | Description |
|--------|-------------|
| `figure1_convergence.png` | Convergence curves (best run per optimizer, both tasks) |
| `figure2_pareto.png` | Pareto frontier: test metric vs wall-clock time |
| `figure3_heatmaps_*.png` | HP sensitivity heatmaps (LR × WD per optimizer) |
| `figure4_horizon.png` | Training horizon curves (RQ2) |
| `table1_*.json` | Final accuracy/perplexity: best, mean, std |
| `table2_recommendations.json` | Practitioner optimizer selection guide |
| `sse_*.json` | Small-Scale Efficiency metric per optimizer |

## Optimizers

| Optimizer | Source | Notes |
|-----------|--------|-------|
| SGD+Momentum | PyTorch built-in | Baseline, momentum=0.9 |
| AdamW | PyTorch built-in | Baseline |
| Lion | `pip install lion-pytorch` | Uses 3–10× lower LR than AdamW |
| AdEMAMix | Inline implementation | alpha=5.0, beta3=0.9999 (paper defaults) |
| SOAP | Inline implementation | Kronecker-factored preconditioner, freq=10 |
| Muon | `pip install muon-optimizer` | SVD-based, 2D weights only (biases → aux Adam) |
| Schedule-Free AdamW | `pip install schedulefree` | No external LR scheduler; uses `optimizer.train()`/`eval()` |

## SSE Metric

**Small-Scale Efficiency (SSE)** balances accuracy against compute cost:

- CIFAR-10: `SSE = Test Accuracy / Total Wall-Clock Seconds`
- PTB: `SSE = (1 / Val Perplexity) / Total Wall-Clock Seconds`

## CLI Reference

### run_single.py

```
--task          cifar10 | ptb (required)
--optimizer     Optimizer name (required)
--lr            Learning rate (required)
--wd            Weight decay (default: 0.05)
--epochs        Epochs for CIFAR-10 (default: 50)
--max_steps     Max steps for PTB (default: 50000)
--batch_size    Batch size (default: 128)
--device        Force device: cpu | mps | cuda (default: auto-detect)
--seed          Random seed (default: 42)
--num_workers   DataLoader workers (default: 4 for CIFAR-10, 2 for PTB)
```

### run_grid.py

```
--task          cifar10 | ptb (required)
--optimizers    Space-separated subset (default: all 7)
--config        Path to grid YAML (default: configs/grid.yaml)
--device        Force device (default: auto-detect)
--seed          Random seed (default: 42)
--num_workers   DataLoader workers
```

### run_diagnostics.py

```
--task          cifar10 | ptb (default: cifar10)
--diagnostic    horizon | batchsize | all (required)
--optimizers    Subset for batchsize diagnostic (default: all)
--device        Force device (default: auto-detect)
--seed          Random seed (default: 42)
--num_workers   DataLoader workers
```

### analyze.py

```
--results_dir   Input directory (default: results)
--figures_dir   Output directory (default: figures)
```

## Troubleshooting

- **MPS errors:** Some optimizers may fail on Apple Silicon MPS. Rerun with `--device cpu`.
- **DataLoader hangs on macOS:** Set `--num_workers 0` to disable multiprocessing.
- **Missing optimizer package:** Error messages include the exact `pip install` command needed.
- **Resuming interrupted runs:** Just re-run the same command — skip-if-exists handles it automatically.
