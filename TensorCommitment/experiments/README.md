# Optimization Benchmarking Experiments

Reproducible experiments comparing layer-selection optimization strategies for the AlphaPrun objective.

---

## Overview

We compare **five** optimization strategies, all evaluated with the **AlphaPrun objective** (significance_score) for fair comparison:

| Method      | Description                                      |
|------------|---------------------------------------------------|
| Random     | Random (position-weighted) integer values         |
| Norm L1    | L1 norm of layer weights as value                 |
| Norm L2    | L2 norm of layer weights as value                 |
| Norm L∞    | L∞ norm of layer weights as value                 |
| AlphaPrun  | Significance score from AlphaPruning (proposed)  |

**Output:** For each budget case we report the AlphaPrun benefit (and ratio) achieved by each method. Higher is better.

---

## Environment & Dependencies & Reproducibility

- **Python:** 3.8+
- **Required packages:** `numpy`, `matplotlib`, `seaborn`, `torch`, `transformers`
- **External:** AlphaPruning (via `../layerSelectionLib`); HuggingFace model access.

Install from the repo root (or ensure `layerSelectionLib` is on `PYTHONPATH`):

```bash
cd /path/to/finalcode/experiments
pip install -r requirements.txt
```

**HuggingFace:** Set a read token for gated models (e.g. Llama):

```bash
export HF_TOKEN=your_token_here
```


**Seeds:** Budget case generation uses `np.random.seed(42)`. The benchmark runner uses `case_id` as the random seed for the Random method and a fixed seed for budget scaling. With the same `cases/` and scores, results are deterministic.

**Artifacts:** For exact reproduction, use the same `cases/`. Weight norms are deterministic given the same model and scores.

---

## Directory Layout

```
experiments/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── cases/                       # Budget cases (generated)
│   ├── case_001.json ... case_100.json
│   └── all_cases.json
├── weight_norms/                # Per-model weight norms (generated)
│   └── {model_short}_norms.json
├── results/                     # Benchmark outputs and plots
│   ├── case_*_results.json
│   ├── {model_short}_all_results.json
│   ├── {model_short}_comparison.png
│   └── combined_comparison.png  # When running all models
├── scalability/                 # Scalability experiments
│   ├── run_scalability_benchmarks.py
│   ├── plot_scalability.py
│   └── results/
|── layer_analysis_experiments
|── scalability_experiments
├── generate_budget_cases.py     # Step 1: sample budget cases
├── extract_weight_norms.py      # Step 2: extract L1/L2/L∞ norms
├── run_experiment_benchmarks.py # Step 3: run all methods on cases
├── plot_results.py              # Step 4: boxplots and summary
├── run_full_benchmark.py        # Single model: run steps 1–4
├── run_all_models.py            # All models + combined plot
├── test_benchmark.py            # Quick sanity check
└── model_utils.py               # Shared helpers
```

Scores and model cache (relative to main repo directory):

- **Scores:** `layerSelectionLib/scores/` (e.g. `llama2-7b_scores.json`, `opt125m_scores.json`).
- **Model cache:** `layerSelectionLib/layer_selection_integration/llm_weights/` (default for `--cache_dir`).

---

## Commands to Reproduce

Run from `finalcode/experiments/`. Defaults assume this layout and `../layerSelectionLib/scores` for scores.

The script to reproduce the figures in the paper.
```bash
python run_experiment_benchmarks.py --experiment_cases cases/experiment_cases.json --scores_dir ../layerSelectionLib/scores/ --norms_dir ./weight_norms/ --output_dir results/ --models llama2_70b llama2_13b llama2_7b opt_125m --plot
```

```bash
python scalability_experiments/run_scalability_benchmarks.py --cases scalability_experiments/scalability_cases.json --scores_dir ../layerSelectionLib/scores/ --norms_dir ./weight_norms/ --output_dir results/

python scalability_experiments/plot_scalability.py --results results --output results/scalability_plot.png
```



```bash
cd layer_analysis_experiments
./run_layer_analysis.sh
```

```bash
cd experiments

# Run all benchmarks (Terkle, Verkle, Merkle)
python run_all_sweeps.py --plot

# Or run individual sweeps
python sweep_terkle.py
python sweep_verkle.py
python sweep_merkle.py

# Generate plot from existing results
python plot_dimension_benchmark.py
```

### Single model (full pipeline)

```bash
cd /path/to/finalcode/experiments

python run_full_benchmark.py \
  --model facebook/opt-125m \
  --scores_file ../layerSelectionLib/scores/opt125m_scores.json \
  --num_cases 100 \
  --num_verifiers 3
```

Optional: `--skip_cases` to reuse existing `cases/`, `--skip_norms` to reuse existing `weight_norms/`, `--cache_dir ...` to override model cache.

### All models + combined plot

```bash
python run_all_models.py --num_cases 100 --num_verifiers 3
```

Use `--scores_dir ../layerSelectionLib/scores` if not already the default. Skip steps if needed: `--skip_norms`, `--skip_cases`, or `--skip_individual` (only build combined plot from existing `*_all_results.json`).


### Step-by-step (single model)

```bash
# 1) Generate budget cases (seed 42)
python generate_budget_cases.py --num_cases 100 --num_verifiers 3 --output_dir cases

# 2) Extract weight norms (one-time per model)
python extract_weight_norms.py \
  --model facebook/opt-125m \
  --scores_file ../layerSelectionLib/scores/opt125m_scores.json \
  --output weight_norms/opt_125m_norms.json

# 3) Run benchmarks
python run_benchmarks.py \
  --scores ../layerSelectionLib/scores/opt125m_scores.json \
  --norms weight_norms/opt_125m_norms.json \
  --cases_dir cases \
  --output_dir results \
  --model_name opt_125m

# 4) Plot
python plot_results.py \
  --results results/opt_125m_all_results.json \
  --output results/opt_125m_comparison.png
```

### Combined plot only (when results are ready)

When you already have per-model summary files (`*_all_results.json`) in `results/`, you can build or refresh the combined plot without re-running benchmarks.

**Option A — use `run_all_models.py` (finds all `*_all_results.json` in `results/`):**

```bash
python run_all_models.py --skip_individual
```

This only runs the plotting step and writes `results/combined_comparison.png` and `results/combined_comparison.txt`.

**Option B — use `plot_results.py` directly:**

```bash
# Pass the results directory; script discovers all *_all_results.json
python plot_results.py --results results --output results/combined_comparison.png --combined
```

Or list specific summary files:

```bash
python plot_results.py \
  --results results/llama2_70b_experiment_results.json,results/llama2_13b_experiment_results.json,results/llama2_7b_experiment_results.json,results/opt_125m_experiment_results.json \
  --output results/combined_comparison.png \
  --combined
```


---

### Expected Outputs

- **cases/**  
  `case_001.json` … `case_100.json`, `all_cases.json` (budgets and total budget per case).

- **weight_norms/**  
  `{model_short}_norms.json`: per-layer `norm_1`, `norm_2`, `norm_inf`.

- **results/**  
  - `case_XXX_results.json`: per-case results for all methods.  
  - `{model_short}_all_results.json`: aggregated results.  
  - `{model_short}_comparison.png` / `.txt`: boxplot and summary.  
  - `combined_comparison.png`: all models (when using `run_all_models.py`).

Metrics in plots: distribution of AlphaPrun benefit (or ratio) per method across cases (mean, median, quartiles, outliers).

---

### Notes

- All methods use the same cost function (number of parameters).  
- Weight norm extraction loads the full model once per model; large models can be slow.  
- For out-of-memory issues: reduce `--num_cases` or run one model at a time.
