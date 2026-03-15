# How to Run — Adaptive Cross-Project Defect Prediction

This guide has every command you need to reproduce our results on any machine.
If your machine has a GPU, the MLP base learner in the ensemble will automatically use CUDA — no code changes needed.

---

## System Requirements

| Item | Minimum | Recommended (GPU machine) |
|------|---------|--------------------------|
| Python | 3.10+ | 3.10 – 3.12 |
| RAM | 8 GB | 16 GB |
| GPU | not needed | NVIDIA (CUDA 12.1+) |
| Storage | ~500 MB | ~500 MB |

---

## 1. Clone / Get the Code

```bash
git clone <your-repo-url>
cd Adaptive-Cross-Project-Defect-Prediction-with-Similarity-Based-Source-Selection
```

---

## 2. Create a Virtual Environment & Install Dependencies

```bash
# Create fresh venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux / Mac)
source venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

### If you have a GPU (NVIDIA) — install PyTorch with CUDA

Run this **after** the step above (replaces the CPU-only torch):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Without torch installed the MLP base learner gracefully falls back to sklearn
> LogisticRegression — results are still valid, just slightly lower quality.

---

## 3. Verify Installation (Smoke Test)

```bash
cd src_new
python -c "from pipeline import load_all_projects; p = load_all_projects('../data/'); print(len(p), 'projects loaded')"
```

Expected output: `25 projects loaded`

---

## 4. Quick Single-Project Test (~1–2 minutes)

Useful to confirm everything works before the full run.

```bash
cd src_new
python pipeline.py --data_dir ../data/ --target ant --trials 5 --feature_mode metrics
```

You will see:
- Each project loaded with feature shape and defect rate
- Ensemble GPU/CPU status
- Final: `F1=X.XXX  AUC=X.XXX  MCC=X.XXX`

---

## 5. Full Benchmark — All 25 Projects, 30 Trials Each

This is the authoritative evaluation that generates `results_new.csv`.

**Expected runtime:**
- No GPU (CPU only): ~3–5 hours
- With GPU (NVIDIA): ~40–60 minutes

```bash
cd src_new
python pipeline.py --data_dir ../data/ --feature_mode metrics --trials 30 --out ../results_new.csv
```

Output at the end will show:
```
RESULTS SUMMARY
...per-project table...

Averages: F1=X.XXX  AUC=X.XXX  MCC=X.XXX

Vs TriStage-CPDP (base paper):
  F1:  Ours=X.XXX  Paper=0.627  ...
  AUC: Ours=X.XXX  Paper=0.663  ...
  MCC: Ours=X.XXX  Paper=0.245  ...
```

And saves `results_new.csv` in the project root.

---

## 6. Single Target — Any Project

Replace `ant` with any project name from the data folder:

```bash
cd src_new
python pipeline.py --data_dir ../data/ --target ant       --trials 30
python pipeline.py --data_dir ../data/ --target poi-3.0   --trials 30
python pipeline.py --data_dir ../data/ --target xerces-1.4 --trials 30
```

Available targets (25 total):
`ant`, `camel-1.0`, `camel-1.2`, `camel-1.4`, `camel-1.6`,
`ivy-1.1`, `ivy-1.4`, `ivy-2.0`,
`log4j-1.0`, `log4j-1.1`, `log4j-1.2`,
`lucene-2.0`, `lucene-2.2`, `lucene-2.4`,
`poi-1.5`, `poi-2.0`, `poi-2.5`, `poi-3.0`,
`xalan-2.4`, `xalan-2.5`, `xalan-2.6`, `xalan-2.7`,
`xerces-1.2`, `xerces-1.3`, `xerces-1.4`

---

## 7. Hyperparameter Grid Search (Optional — Advanced)

Sweeps `top_k`, `coral_reg`, `smote_strategy`, `n_folds` automatically.

```bash
cd src_new

# Grid search on all projects (slow — use a few targets to test)
python pipeline.py --data_dir ../data/ --grid_search --grid_trials 10

# Grid search on a single target (faster)
python pipeline.py --data_dir ../data/ --grid_search --grid_target ant --grid_trials 10
```

Results saved to `grid_results.csv`.

---

## 8. Inspect Saved Results

```bash
# From project root
python check_results.py
```

This reads `results.csv` and prints:
- Overall F1 / AUC / MCC averages
- Per-project table
- Win / loss vs TriStage-CPDP paper

---

## Key Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/` | Folder with `.csv` project files |
| `--feature_mode` | `metrics` | Feature type: `metrics`, `full`, `metrics+graphsage`, `metrics+codet5` |
| `--target` | *(all)* | Run only one target project |
| `--trials` | `30` | Number of random-seed trials per target |
| `--top_k` | `3` | Number of source projects to select |
| `--out` | `results.csv` | Output CSV filename |
| `--coral_reg` | `1e-3` | CORAL domain adaptation regularisation |
| `--smote_strategy` | `auto` | SMOTE balance ratio (`auto` = full balance) |
| `--n_folds` | `5` | Stacked ensemble OOF folds |
| `--w_cos` | `0.40` | Cosine similarity weight in source selection |
| `--w_mmd` | `0.35` | MMD weight in source selection |
| `--w_adist` | `0.15` | A-distance weight in source selection |
| `--w_dr` | `0.10` | Defect-rate similarity weight in source selection |

---

## Pipeline Stages (What Happens Internally)

```
Stage 0  →  Similarity-Based Source Selection
              (cosine + MMD + A-distance + defect-rate)
Stage 1  →  Feature Extraction (metrics / CodeT5+ / GraphSAGE)
Stage 2  →  Feature Selection  (LASSO-BIC + RFE)
Stage 3  →  Domain Adaptation  (CORAL)
Stage 4  →  Class Imbalance    (SMOTE)
Stage 5  →  Stacked Ensemble   (RF + HGB×2 + XGBoost + MLP → LogReg meta)
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: pandas` | Run `pip install -r requirements.txt` inside the venv |
| `No module named six.moves` | Run `pip install --force-reinstall pandas==2.2.3` |
| `No label column found` | Your CSV must have one of: `bug`, `defects`, `label`, `Defective`, `class`, `bugs`, `isDefective` |
| MLP falls back to sklearn LR | Install torch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| `Need at least 2 projects` | Make sure `--data_dir` points to the folder with at least 2 `.csv` files |
