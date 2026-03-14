# Adaptive Cross-Project Defect Prediction
### with Similarity-Based Source Selection

**Base paper:** Zou & Wang, *TriStage-CPDP* — Complex & Intelligent Systems, 2025  


---

## Project Structure

```
CPDP-3S/
├── src/                        # Baseline & utility scripts
│   ├── check.py                # Validate CSV schema across all data files
│   ├── cross_project_baseline.py   # Simple XGBoost cross-project baseline
│   ├── leave_one_project_out.py    # LOPO evaluation with CORAL + cosine selection
│   ├── leave_one_project_out_random.py  # LOPO with random source selection
│   ├── similarity_selection.py     # Standalone cosine similarity ranker
│   ├── train_single_project.py     # Within-project XGBoost baseline
│   ├── row.py                      # Combine all CSVs into one array
│   └── main.py                     # Environment check
│
├── src_new/                    # Full adaptive pipeline (main system)
│   ├── pipeline.py             # Main orchestrator — run this
│   ├── source_selector.py      # Stage 0: similarity-based source selection
│   ├── feature_extraction.py   # Stage 1: feature extraction orchestrator
│   ├── feature_codet5.py       # Stage 1a: CodeT5+ semantic embeddings
│   ├── feature_graphsage.py    # Stage 1b: GraphSAGE structural embeddings
│   ├── feature_selection.py    # Stage 2: LASSO-BIC + RFE
│   ├── coral.py                # Stage 3: CORAL domain adaptation
│   ├── smote.py                # Stage 4: SMOTE oversampling
│   ├── ensemble.py             # Stage 5: stacked ensemble + PyTorch MLP
│   ├── smoke_test.py           # End-to-end sanity check on synthetic data
│   └── debug.py                # Step-by-step import and load tester
│
├── data/                       # PROMISE CSV files (place yours here)
├── embeddings/                 # Auto-created — cached .npy embedding files
├── results.csv                 # Output after benchmark run
└── README.md
```

---

## Part 1 — `src/` Baseline Scripts

These are standalone scripts for quick experiments and validation. They use XGBoost and simple cosine-based source selection — no deep learning, no stacking. Use them to sanity-check data and establish a quick performance baseline.

### What each file does

**`check.py`** — validates that every CSV in `data/` has exactly the 24 expected columns (`name`, `version`, `classname`, 20 metrics, `bug`) in the right order. Run this first whenever you add new data files.

**`train_single_project.py`** — within-project baseline. Trains XGBoost on 80% of `ant.csv` and tests on the remaining 20%. Gives an upper-bound reference for what's achievable with labeled data from the same project.

**`cross_project_baseline.py`** — simple CPDP baseline. Trains on xalan-2.4, xalan-2.5, xalan-2.6 combined and tests on `ant.csv`. No domain adaptation. Shows raw cross-project performance without any alignment.

**`similarity_selection.py`** — ranks all projects in `data/` by cosine similarity to a chosen target (`ant.csv` by default). Prints a ranked list. Useful for inspecting which projects are most similar before running the full pipeline.

**`leave_one_project_out.py`** — LOPO evaluation loop. For each project as the target, selects top-3 sources by cosine similarity, applies CORAL alignment, trains XGBoost, and reports F1/AUC/MCC. Averages across all projects at the end.

**`leave_one_project_out_random.py`** — same LOPO loop but with random source selection instead of similarity-based. Run this alongside `leave_one_project_out.py` to quantify how much the similarity selection actually helps.

**`row.py`** — combines all CSVs into a single numpy array. Utility script for quick data inspection.

**`main.py`** — prints "Environment is working correctly!" — use to verify Python and pandas are installed.

### Requirements for `src/`

```bash
pip install pandas numpy scikit-learn xgboost
```

### Running `src/` scripts

```bash
# 1. Check all CSV files have correct schema
python src/check.py

# 2. Within-project baseline (upper bound reference)
python src/train_single_project.py

# 3. Simple cross-project baseline (no adaptation)
python src/cross_project_baseline.py

# 4. See which projects are most similar to ant
python src/similarity_selection.py

# 5. LOPO with cosine-based source selection + CORAL
python src/leave_one_project_out.py

# 6. LOPO with random source selection (comparison baseline)
python src/leave_one_project_out_random.py
```

**Expected output from `leave_one_project_out.py`:**
```
ant.csv:     F1=0.xxx, AUC=0.xxx, MCC=0.xxx
camel-1.0.csv: F1=0.xxx, AUC=0.xxx, MCC=0.xxx
...
===== Overall Average Performance =====
Avg F1:  0.xxx
Avg AUC: 0.xxx
Avg MCC: 0.xxx
```

---

## Part 2 — `src_new/` Full Adaptive Pipeline

This is the complete system submitted as the major project. It improves on the base paper (TriStage-CPDP) with similarity-based source selection, CORAL domain adaptation, SMOTE imbalance handling, and a GPU-accelerated stacked ensemble.

### Pipeline overview

```
Stage 0  Source selection     Cosine + MMD + A-Distance + defect-rate similarity
Stage 1  Feature extraction   CodeT5+ (256-d) + GraphSAGE (32-d) + metrics (20-d) = 308-d
Stage 2  Feature selection    LASSO with BIC-guided alpha, then RFE
Stage 3  Domain adaptation    CORAL — aligns source covariance to target (zero labels needed)
Stage 4  Imbalance handling   SMOTE — synthetic minority oversampling on training set only
Stage 5  Stacked ensemble     RF + HGB×2 + TorchMLP (GPU) → Logistic Regression meta-learner
```

### Improvements over the base paper

| Aspect | TriStage-CPDP (base paper) | This project |
|--------|---------------------------|--------------|
| Source selection | Uses all 9 projects | Top-K filtered by 4-signal similarity score |
| Domain adaptation | LPP (needs 10% target labels) | CORAL (zero target labels needed) |
| Classifier | Nearest-centroid | Stacked ensemble: RF + HGB×2 + MLP |
| GPU support | None | PyTorch MLP uses CUDA on Windows (RTX 4050) |
| Imbalance | Basic oversampling | SMOTE with adaptive k-neighbours |

### What each file does

**`pipeline.py`** — the main orchestrator. Loads all CSVs from `data/`, runs 30 parallel trials per target project using joblib, prints a results table comparing your metrics against TriStage-CPDP, and saves output to `results.csv`.

**`source_selector.py`** — Stage 0. For each candidate source project, computes four similarity signals: cosine similarity (0.35), MMD (0.35), A-Distance (0.15), and defect-rate similarity (0.15). Returns the top-K highest-scoring projects. This directly addresses the negative transfer problem in the base paper.

**`feature_extraction.py`** — Stage 1 orchestrator. Combines CodeT5+, GraphSAGE, and traditional metrics into a single joint feature vector `XP = concat(XC, XS, XT)`. Supports four modes. Handles caching, alignment to CSV row order, and fallback when source files are unavailable.

**`feature_codet5.py`** — Stage 1a. Loads `Salesforce/codet5p-110m-embedding` and produces 256-d embeddings per Java file by mean-pooling the last hidden state. Results are cached as `.npy` files — runs once per project.

**`feature_graphsage.py`** — Stage 1b. Parses Java source files using regex to detect `import`/`extends`/`implements` dependencies, builds a Class Dependency Network (CDN) as a directed graph, initialises nodes with SVD-based embeddings, then trains a 2-layer GraphSAGE with contrastive loss to produce 32-d structural embeddings. Pure numpy — no torch_geometric needed.

**`feature_selection.py`** — Stage 2. BIC-guided LASSO for fast coarse elimination (shrinks irrelevant coefficients to zero), followed by RFE with Random Forest importance for fine-grained selection. Fitted only on source data, applied to source and target separately to prevent leakage.

**`coral.py`** — Stage 3. Computes source and target covariance matrices and finds the closed-form transformation `A = C_s^{-1/2} · C_t^{1/2}` that maps source features into the target's statistical space. Deterministic, fast, no training instability.

**`smote.py`** — Stage 4. Generates synthetic defective-class samples by interpolating between real minority-class instances and their K nearest neighbours: `x_new = x_i + λ·(x_nn − x_i)`. Applied only after the train/test split — test data is never touched.

**`ensemble.py`** — Stage 5. Trains Random Forest, two HistGradientBoosting variants, and a PyTorch MLP as base learners using 3-fold OOF predictions to generate unbiased meta-features, then trains a Logistic Regression meta-learner on top. The MLP automatically uses CUDA GPU on Windows if torch is installed.

**`smoke_test.py`** — runs all 5 stages on small synthetic data and prints a pass/fail for each. Run this first after any code change.

**`debug.py`** — prints progress at each import and loading step. Run this if `pipeline.py` produces no output — it will show exactly which step is failing.

### Requirements for `src_new/`

```bash
# Core (required)
pip install scikit-learn scipy pandas numpy joblib networkx

# GPU acceleration — PyTorch with CUDA 12.1 (Windows + NVIDIA RTX 4050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Full features — CodeT5+ and GraphSAGE embeddings (optional)
pip install transformers torch
```

> **Note:** cuML does not support Windows. PyTorch is the correct GPU option for Windows + NVIDIA. The MLP falls back to CPU silently if torch is not installed.

### Data setup

Place your PROMISE CSV files in `data/`. Any filename works — the pipeline reads all `.csv` files in the folder.

Each CSV must have:
- 20 traditional metric columns: `wmc, dit, noc, cbo, rfc, lcom, ca, ce, npm, lcom3, loc, dam, moa, mfa, cam, ic, cbm, amc, max_cc, avg_cc`
- A label column named `bug`, `defects`, `Defective`, `class`, or `bugs`
- Optional metadata columns (`name`, `version`, `classname`) — these are automatically dropped

Projects with different column counts are automatically aligned to their common column intersection — no manual fixing needed.

### Running `src_new/`

```bash
# Step 0: Verify everything works (always do this first)
python src_new/smoke_test.py

# Step 1: If smoke test passes, run the full benchmark
python src_new/pipeline.py --data_dir data/ --feature_mode metrics --trials 30

# Run a single target project only (faster for testing)
python src_new/pipeline.py --data_dir data/ --target ant --feature_mode metrics --trials 30

# Full features with Java source (CodeT5+ + GraphSAGE + metrics)
python src_new/pipeline.py --data_dir data/ --src_root java_src/ --feature_mode full --trials 30

# If pipeline.py shows no output, run the debug script first
python src_new/debug.py
```

**Expected output:**
```
Loading projects (feature_mode='metrics') ...
  Loaded ant                     | X=(745, 20) | defect rate 22.3%
  Loaded camel-1.0               | X=(339, 20) | defect rate 18.9%
  ...
  -> 17 projects ready. Feature dim: 20

  Target: ant | 30 trials
    Done 30/30 | F1=0.523 AUC=0.785 MCC=0.371

======================================================================
RESULTS SUMMARY
======================================================================
 project  f1_mean  f1_std  auc_mean  auc_std  mcc_mean  mcc_std
     ant    0.523   0.041     0.785    0.032     0.371    0.058
  ...

Vs TriStage-CPDP (base paper):
  F1:  Ours=0.xxx  Paper=0.627  ▲x.x%
  AUC: Ours=0.xxx  Paper=0.663  ▲x.x%
  MCC: Ours=0.xxx  Paper=0.245  ▲x.x%

Saved to results.csv
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/` | Directory containing PROMISE CSV files |
| `--src_root` | None | Parent dir of Java source folders (for CodeT5+/GraphSAGE) |
| `--feature_mode` | `metrics` | `full` · `metrics` · `metrics+graphsage` · `metrics+codet5` |
| `--target` | None | Single target project name — omit to run all |
| `--top_k` | `3` | Number of source projects to select per trial |
| `--trials` | `30` | Independent trials per target (30 matches base paper) |
| `--graphsage_epochs` | `100` | GraphSAGE training epochs |
| `--cache_dir` | `embeddings/` | Where to store precomputed .npy embedding files |
| `--out` | `results.csv` | Output path for results table |

### Feature modes

| Mode | Features included | Dim | Requires |
|------|------------------|-----|----------|
| `metrics` | Traditional PROMISE metrics only | 20 | Just CSVs |
| `metrics+graphsage` | CDN structural embeddings + metrics | 52 | CSVs + Java source |
| `metrics+codet5` | CodeT5+ semantic embeddings + metrics | 276 | CSVs + Java source + torch |
| `full` | CodeT5+ + GraphSAGE + metrics | 308 | CSVs + Java source + torch |

### Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `top_k` | 3 | Number of source projects selected per trial |
| `target_label_ratio` | 0.10 | Fraction of target used as labeled guide — matches base paper |
| `w_cos / w_mmd` | 0.35 each | Cosine and MMD weights in source selection |
| `w_adist / w_defect_rate` | 0.15 each | A-Distance and defect-rate weights |
| `CORAL reg` | 1e-3 | Regularisation added to covariance matrices |
| `SMOTE k` | 5 | Nearest neighbours for synthetic sample generation |
| `n_folds` | 3 | OOF folds for stacked ensemble (3=fast, 5=more accurate) |
| `MLP hidden_dim` | 128 | PyTorch MLP hidden layer size |
| `MLP epochs` | 50 | PyTorch MLP training epochs |

---

## Expected Results vs Base Paper

Results on the PROMISE dataset (average across 10 target projects, 30 trials each):

| Metric | TriStage-CPDP (paper) | This project (target) |
|--------|-----------------------|-----------------------|
| F1 (avg) | 0.627 | > 0.640 |
| AUC (avg) | 0.663 | > 0.675 |
| MCC (avg) | 0.245 | > 0.260 |

Sample result observed (target = ant):

| Metric | TriStage-CPDP | Ours |
|--------|---------------|------|
| F1 | 0.632 | 0.523 |
| AUC | 0.585 | **0.785** |
| MCC | 0.214 | **0.371** |

Best gains expected on **log4j** (92% defect rate, extreme imbalance) and **jedit** (small dataset) — exactly where negative transfer hurts TriStage-CPDP most.

---

## Statistical Validation

To match the base paper's validation protocol exactly, report across 30 trials:

- **Wilcoxon signed-rank test** — p < 0.05 confirms statistically significant difference between methods
- **Cliff's δ** — effect size: negligible (<0.147) / small / medium / large (>0.474)
- **Scott-Knott ESD test** — hierarchical clustering to rank all methods into statistically distinct groups

These three tests are used in the base paper's Tables 5, 6, and 7 and allow direct side-by-side comparison.

---

## Troubleshooting

**Pipeline shows no output:**
```bash
python src_new/pipeline.py --data_dir data/ --feature_mode metrics 2>&1
python src_new/debug.py
```

**Dimension mismatch error (`array at index 0 has size 21 and index 16 has size 20`):**  
Fixed automatically — projects are aligned to their common column intersection at load time. If it still occurs, run `src/check.py` to find the mismatched file.

**ConvergenceWarning from LASSO:**  
Already suppressed internally via `warnings.catch_warnings()` with `max_iter=10000`.

**`BatchNorm` error on small datasets:**  
Fixed — the MLP uses `LayerNorm` instead of `BatchNorm1d`, which works with any batch size.

**cuML installation fails on Windows:**  
cuML is Linux-only. Use PyTorch instead — install with the command in Requirements above. The MLP uses your RTX 4050 automatically via CUDA.

**`ImportError: cannot import name 'load_all_projects' from 'pipeline'`:**  
You are using the old `src/pipeline.py` instead of the new `src_new/pipeline.py`. Make sure you are running from the right folder.

---

## References

1. Zou, Y., & Wang, H. (2025). A three-stage cross-project defect prediction framework based on feature representation and knowledge transfer. *Complex & Intelligent Systems*, 11(459).
2. Sun, B., & Saenko, K. (2016). Return of frustratingly easy domain adaptation. *AAAI*.
3. Wang, Y. et al. (2023). CodeT5+: Open code large language models for code understanding and generation. *arXiv:2305.07922*.
4. Hamilton, W. et al. (2017). Inductive representation learning on large graphs (GraphSAGE). *NeurIPS*.
5. Chawla, N. et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
6. Kiran, U. et al. (2025). Improving cross-project defect prediction through feature selection and model optimization. *GIET 2025*.