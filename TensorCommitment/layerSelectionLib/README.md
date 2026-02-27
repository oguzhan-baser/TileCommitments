# layerSelectionLib

Layer significance extraction, contiguous budgeted assignment, and benchmarking for verifier-aware model partitioning.

This is an optional module in `finalcode`, independent from the activation -> interpolation -> commitment pipeline.

This README consolidates the library documentation that was previously spread across integration, optimization, benchmark, and AlphaPruning helper docs.

`AlphaPruning/README.md` is intentionally kept as separate upstream project documentation.

## What This Library Does

1. Extracts per-layer significance from AlphaPruning ESD metrics.
2. Solves Multi-Verifier Budgeted Contiguous Selection (MV-BCS) optimally.
3. Benchmarks multiple objective strategies against AlphaPrun scoring.
4. Provides visualization and model-comparison tooling.

## Module Layout

```text
layerSelectionLib/
├── README.md
├── run_layer_selection.py
├── AlphaPruning/
├── layer_selection_integration/
│   ├── extract_layer_scores.py
│   └── visualize_scores.py
├── OptimizationModule/
│   ├── layer_selection_optimizer.py
│   ├── optimize_layers.py
│   └── scores_loader.py
├── benchmark/
│   ├── run_full_benchmark.py
│   ├── run_all_models.py
│   ├── run_benchmarks.py
│   ├── generate_budget_cases.py
│   ├── extract_weight_norms.py
│   └── plot_results.py
├── scores/
├── plots/
└── extract_values_from_LLMs.sh
```

## Installation

### Minimal dependencies (main workflows)

```bash
pip install numpy torch transformers matplotlib seaborn accelerate weightwatcher datasets sentencepiece
```

### AlphaPruning reference environment (from original install guide)

```bash
conda create -n prune_llm python=3.9
conda activate prune_llm
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.47.1 datasets==2.19.0 numpy==1.24.3 wandb sentencepiece
pip install accelerate==0.26.0
pip install weightwatcher
```

### Extra dependency for image classifiers

```bash
pip install timm==0.4.12
```

### Hugging Face access

For gated/private models and benchmark scripts:

```bash
export HF_TOKEN=your_token_here
```

## Primary Interface

Use the unified runner:

```bash
cd finalcode/layerSelectionLib
python run_layer_selection.py --help
```

### Extract scores

```bash
python run_layer_selection.py extract \
  --model facebook/opt-125m \
  --output scores/opt125m_scores.json \
  --pretty
```

### Visualize scores

```bash
python run_layer_selection.py visualize \
  --mode json_single \
  --json-file scores/opt125m_scores.json \
  --output plots/opt125m_plot.png
```

### Optimize contiguous assignments

```bash
python run_layer_selection.py optimize \
  --scores scores/opt125m_scores.json \
  --budgets 10000000 20000000 15000000 \
  --output scores/opt125m_optimized.json \
  --pretty
```

### Run full benchmark pipeline

```bash
python run_layer_selection.py benchmark \
  --model facebook/opt-125m \
  --scores-file scores/opt125m_scores.json \
  --num-cases 100 \
  --num-verifiers 3
```

## Layer Score Pipeline

### Conceptual flow

```text
weights -> ESD metrics (alpha_peak/...) -> pruning ratios -> significance scores
```

### Relationship used by this library

- Lower pruning ratio means more critical layer.
- Default significance mapping is:
  - `significance_score = 1.0 - pruning_ratio`

### Why values can look compressed or near 0-1

1. Visualization in `data_dir` mode normalizes to [0, 1].
2. Raw scores from JSON are `1 - ratio`, often naturally clustered.
3. Default `block_wise` mapping assigns same metric across layers in a block.
4. `epsilon` and global sparsity scaling constrain ratio spread.

To increase differentiation:
- use `--mapping_type layer_wise`
- increase `--epsilon`
- inspect raw ESD metrics directly (`--use_raw_metrics` in visualization)

## ESD `.npy` Metric Files

Precomputed files in `AlphaPruning/data/<model>/<metric>.npy` are 1D arrays of per-prunable-layer metrics.

Typical meaning for `alpha_peak`:
- Higher alpha -> better trained -> more prunable
- Lower alpha -> less prunable -> more significant

If metric files are missing, extraction scripts can compute them on the fly (can take minutes to an hour depending on model size).

## Block vs Layer Definitions

### OPT-125M

- 12 transformer blocks
- 6 prunable layers per block:
  - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.out_proj`, `fc1`, `fc2`
- total 72 prunable layers

### LLaMA-style (e.g., 7B)

- 32 transformer blocks
- 7 prunable layers per block:
  - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`, `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`
- total 224 prunable layers for 7B

`*_scores.json` includes both per-layer entries and per-block aggregates.

## Score Extraction Details

Direct script (without wrapper):

```bash
cd layer_selection_integration
python extract_layer_scores.py \
  --model facebook/opt-125m \
  --ww_metric alpha_peak \
  --sparsity_ratio 0.7 \
  --epsilon 0.3 \
  --output scores.json \
  --pretty
```

### Key extraction arguments

| Argument | Meaning | Default |
|---|---|---|
| `--model` | HF model id or local path | required |
| `--ww_metric` | ESD metric | `alpha_peak` |
| `--cache_dir` | model cache | `llm_weights` |
| `--ww_metric_cache` | metric cache dir | auto |
| `--sparsity_ratio` | target sparsity | script default |
| `--epsilon` | ratio spread control | script default |
| `--mapping_type` | `block_wise` or `layer_wise` | `block_wise` |
| `--output` | JSON output path | required |

### Output schema (`*_scores.json`)

- `model_name`, `total_parameters`, `num_prunable_layers`, `num_blocks`
- `layers[]` with:
  - `index`
  - `name`
  - `num_parameters` (cost)
  - `significance_score` (benefit)
  - `pruning_ratio`
  - `weight_shape`
- `blocks[]` with per-block summary
- `metadata` with min/max/mean statistics

## Optimization Module (MV-BCS)

Problem:
- `L` layers with value `v_i >= 0` and cost `c_i > 0`
- `M` verifiers with budgets `B_k`
- choose one contiguous interval per verifier (possibly empty), pairwise disjoint, maximize total value under per-verifier budgets

Algorithm:
- Dynamic Programming with sliding-window deque optimization
- Time complexity: `O(M * L)`
- Deterministic and optimal for defined objective

### CLI usage

```bash
python -m OptimizationModule.optimize_layers \
  --scores scores/opt125m_scores.json \
  --budgets 10000000 20000000 15000000 \
  --output results.json \
  --pretty
```

### Python API usage

```python
from OptimizationModule import LayerSelectionOptimizer, load_scores_from_json

values, costs, metadata = load_scores_from_json("scores/opt125m_scores.json")
optimizer = LayerSelectionOptimizer(values, costs, [10000000, 20000000, 15000000])
optimal_benefit, assignments = optimizer.solve()
```

### Validation/testing

```bash
python OptimizationModule/test_optimizer.py
```

## Benchmark Suite

Benchmarks compare multiple objective choices while evaluating outcomes on AlphaPrun benefit.

### Full pipeline in one command

```bash
cd benchmark
python run_full_benchmark.py \
  --model facebook/opt-125m \
  --scores_file ../scores/opt125m_scores.json \
  --num_cases 100 \
  --num_verifiers 3
```

### Step-by-step

```bash
python generate_budget_cases.py --num_cases 100 --num_verifiers 3 --output_dir cases

python extract_weight_norms.py \
  --model facebook/opt-125m \
  --scores_file ../scores/opt125m_scores.json \
  --output weight_norms/opt125m_norms.json \
  --cache_dir ../layer_selection_integration/llm_weights

python run_benchmarks.py \
  --scores ../scores/opt125m_scores.json \
  --norms weight_norms/opt125m_norms.json \
  --cases_dir cases \
  --output_dir results \
  --model_name opt125m

python plot_results.py \
  --results results/opt125m_all_results.json \
  --output results/opt125m_comparison.png
```

### Run all configured models

```bash
python run_all_models.py
```

Useful flags:
- `--skip_norms`
- `--skip_cases`
- `--skip_individual`
- `--output`
- `--num_cases`
- `--cache_dir`

### Outputs

- `cases/case_XXX.json`, `cases/all_cases.json`
- `weight_norms/<model>_norms.json`
- `results/case_XXX_results.json`
- `results/<model>_all_results.json`
- `results/<model>_comparison.png`
- `results/combined_comparison.png`

## Image Classifier Pruning (AlphaPruning submodule)

Supports ConvNeXt/DeiT/ViT pruning workflows in `AlphaPruning/image_classifiers/`.

```bash
python main.py --model convnext_base \
  --data_path /path/to/imagenet \
  --resume /path/to/pretrained/weights \
  --prune_metric wanda_ww \
  --sparsity 0.8 \
  --save_dir /path/to/results \
  --WW_metric alpha_mid \
  --epsilon 0.2
```

Supported `--model` values include:
- `convnext_base`
- `deit_small_patch16_224`
- `deit_base_patch16_224`
- `vit_base_patch16_224`
- `vit_large_patch16_224`

## CVXPY Integration Example

```python
import json
import cvxpy as cp

with open("scores.json") as f:
    data = json.load(f)

layers = data["layers"]
n = len(layers)
values = [x["significance_score"] for x in layers]
costs = [x["num_parameters"] for x in layers]
budget = int(data["total_parameters"] * 0.5)

x = cp.Variable(n, boolean=True)
objective = cp.Maximize(sum(values[i] * x[i] for i in range(n)))
constraints = [sum(costs[i] * x[i] for i in range(n)) <= budget]
cp.Problem(objective, constraints).solve()
```

## Troubleshooting

- `HF_TOKEN environment variable is not set`:
  - `export HF_TOKEN=...` in the same shell session.
- Missing norms file:
  - run `extract_weight_norms.py` first.
- Missing metric file:
  - extraction will compute metrics automatically (can be slow).
- Large model memory pressure:
  - run one model at a time, reduce concurrent workloads, use smaller test runs.
- Plot/benchmark sanity check:
  - run `python benchmark/test_benchmark.py`.

## Notes

- `run_layer_selection.py` is the maintained entrypoint.
- `extract_values_from_LLMs.sh` is legacy compatibility tooling.
- `OptimizationModule_copy/` is kept as legacy/reference; use `OptimizationModule/` for active workflows.
