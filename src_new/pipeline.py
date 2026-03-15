"""
Adaptive Cross-Project Defect Prediction Pipeline
===================================================
Stage 0 → Similarity-Based Source Selection
Stage 1 → Feature Extraction (CodeT5+ | GraphSAGE | metrics)
Stage 2 → Feature Selection (LASSO-BIC + RFE)
Stage 3 → Domain Adaptation (CORAL)
Stage 4 → Class Imbalance Handling (SMOTE)
Stage 5 → Stacked Ensemble

Usage
-----
# Metrics only (no Java source needed):
python pipeline.py --data_dir data/ --feature_mode metrics --trials 30

# Full features (Java source required):
python pipeline.py --data_dir data/ --src_root java_src/ \
    --feature_mode full --trials 30

# Single target:
python pipeline.py --data_dir data/ --target ant --feature_mode metrics
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from source_selector import SimilaritySourceSelector
from feature_selection import LASSOREFSelector
from feature_extraction import FeatureExtractor
from coral import CORAL
from smote import SMOTE
from ensemble import StackedEnsemble


LABEL_CANDIDATES = ['bug', 'defects', 'label', 'Defective',
                    'class', 'bugs', 'isDefective']


# ─────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────
def _find_label_col(df, filepath):
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"No label column in {filepath}. Cols: {list(df.columns)}")


def load_labels(data_dir: str) -> dict:
    """Load only the y labels for each project (used alongside precomputed X)."""
    labels = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
            continue
        name = fname.replace('.csv', '')
        df   = pd.read_csv(os.path.join(data_dir, fname))
        col  = _find_label_col(df, fname)
        labels[name] = (df[col] > 0).astype(int).values
    return labels


def load_all_projects(data_dir: str,
                      src_root: str = None,
                      feature_mode: str = 'metrics',
                      cache_dir: str = 'embeddings',
                      graphsage_epochs: int = 100) -> dict:
    """
    Load all projects, extracting features via Stage 1.

    Returns dict: {project_name: (X_array, y_array)}
    All X arrays are aligned to their common column intersection
    (fixes the 'dimension mismatch' error).
    """
    extractor = FeatureExtractor(
        mode=feature_mode,
        graphsage_epochs=graphsage_epochs,
        cache_dir=cache_dir)

    raw_X = {}
    raw_y = {}

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
            continue
        name     = fname.replace('.csv', '')
        csv_path = os.path.join(data_dir, fname)

        # Find Java source dir for this project
        src_dir = None
        if src_root and feature_mode != 'metrics':
            for cand in [name, name.split('-')[0], name.replace('-','_')]:
                for sub in ['', 'src', 'src/main/java', 'source']:
                    p = os.path.join(src_root, cand, sub)
                    if os.path.isdir(p):
                        src_dir = p
                        break
                if src_dir:
                    break
            if src_dir is None:
                print(f"  [WARN] No source dir found for '{name}' under "
                      f"'{src_root}'. Using metrics-only for this project.")

        try:
            X = extractor.extract(csv_path, src_dir, name)
            df  = pd.read_csv(csv_path)
            col = _find_label_col(df, csv_path)
            y   = (df[col] > 0).astype(int).values

            raw_X[name] = X
            raw_y[name] = y
            print(f"  Loaded {name:25s} | X={X.shape} | "
                  f"defect rate {y.mean():.1%}")
        except Exception as e:
            print(f"  [WARN] Skipping {fname}: {e}")

    if not raw_X:
        return {}

    # Align feature dimensions across all projects
    # (different modes or missing source dirs can cause shape mismatch)
    dims = [X.shape[1] for X in raw_X.values()]
    if len(set(dims)) > 1:
        print(f"\n  [WARN] Feature dim mismatch across projects: "
              f"{set(dims)}. Trimming to min.")
        min_dim = min(dims)
        raw_X = {k: X[:, :min_dim] for k, X in raw_X.items()}

    projects = {name: (raw_X[name], raw_y[name]) for name in raw_X}
    print(f"\n  -> {len(projects)} projects ready. "
          f"Feature dim: {list(raw_X.values())[0].shape[1]}")
    return projects


# ─────────────────────────────────────────────────────────────
# Single Trial
# ─────────────────────────────────────────────────────────────
def run_trial(target_name, target_X, target_y, all_projects,
              top_k=3, target_label_ratio=0.10, seed=42,
              coral_reg=1e-3, smote_strategy='auto', n_folds=5,
              w_cos=0.4, w_mmd=0.35, w_adist=0.15, w_dr=0.10):
    rng = np.random.RandomState(seed)

    # Split: labeled vs test (matches base paper's 10% default)
    labeled_idx = []
    for cls in np.unique(target_y):
        cls_idx  = np.where(target_y == cls)[0]
        n_lab    = max(1, int(len(cls_idx) * target_label_ratio))
        labeled_idx.extend(rng.choice(cls_idx, n_lab, replace=False).tolist())

    labeled_idx = np.array(labeled_idx)
    test_mask   = np.ones(len(target_y), dtype=bool)
    test_mask[labeled_idx] = False

    X_tgt_lab = target_X[labeled_idx]
    y_tgt_lab = target_y[labeled_idx]
    X_test    = target_X[test_mask]
    y_test    = target_y[test_mask]

    if len(np.unique(y_test)) < 2:
        raise ValueError("Test set has only one class — skipping trial.")

    # Stage 0: Source Selection (with defect-rate similarity)
    candidates   = {n: X for n, (X, _) in all_projects.items()
                    if n != target_name}
    candidates_y = {n: y for n, (_, y) in all_projects.items()
                    if n != target_name}
    selector = SimilaritySourceSelector(
        top_k=top_k, w_cos=w_cos, w_mmd=w_mmd,
        w_adist=w_adist, w_dr=w_dr)
    selected = selector.select(
        X_tgt_lab, candidates,
        target_y=y_tgt_lab, sources_y=candidates_y)

    X_source = np.vstack([all_projects[n][0] for n in selected])
    y_source = np.concatenate([all_projects[n][1] for n in selected])

    # Stage 2: Feature Selection
    n_feats = X_source.shape[1]
    fs = LASSOREFSelector(min_features=max(3, n_feats // 5), random_state=seed)
    X_src_sel  = fs.fit_transform(X_source, y_source)
    X_tgt_sel  = fs.transform(X_tgt_lab)
    X_test_sel = fs.transform(X_test)

    # Stage 3: CORAL Domain Adaptation
    coral        = CORAL(reg=coral_reg)
    X_src_ada    = coral.fit_transform(X_src_sel, X_tgt_sel)

    scaler   = StandardScaler()
    combined = scaler.fit_transform(np.vstack([X_src_ada, X_tgt_sel]))
    X_src_train   = combined[:len(X_src_ada)]
    X_tgt_labeled = combined[len(X_src_ada):]
    X_test_final  = scaler.transform(X_test_sel)

    X_train = np.vstack([X_src_train, X_tgt_labeled])
    y_train = np.concatenate([y_source, y_tgt_lab])

    # Stage 4: SMOTE
    k_smote = max(1, min(5, np.bincount(y_train).min() - 1))
    smote_strat = smote_strategy if smote_strategy == 'auto' else float(smote_strategy)
    X_bal, y_bal = SMOTE(
        k_neighbors=k_smote, random_state=seed,
        sampling_strategy=smote_strat).fit_resample(X_train, y_train)

    # Stage 5: Stacked Ensemble
    eff_folds = max(2, min(n_folds, np.bincount(y_bal).min()))
    model     = StackedEnsemble(n_folds=eff_folds, random_state=seed)
    model.fit(X_bal, y_bal)

    y_pred = model.predict(X_test_final)
    y_prob = model.predict_proba(X_test_final)[:, 1]

    return {
        'f1':  f1_score(y_test, y_pred, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'selected_sources': selected,
        'n_features': int(fs.selected_mask_.sum()),
    }


# ─────────────────────────────────────────────────────────────
# Multi-Trial & Benchmark
# ─────────────────────────────────────────────────────────────
def evaluate_target(target_name, all_projects, top_k=3, n_trials=30,
                    n_jobs=-1, **trial_kwargs):
    """Run n_trials in parallel. n_jobs=-1 uses all CPU cores."""
    from joblib import Parallel, delayed

    print(f"\n  Target: {target_name} | {n_trials} trials", flush=True)

    target_X, target_y = all_projects[target_name]
    seeds = [t * 7 + 13 for t in range(n_trials)]

    def _safe_trial(s):
        try:
            return run_trial(target_name, target_X, target_y,
                             all_projects, top_k=top_k, seed=s,
                             **trial_kwargs)
        except Exception as e:
            print(f"    [WARN] Trial seed={s} skipped: {e}")
            return None

    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(_safe_trial)(s) for s in seeds
    )

    f1s  = [r['f1']  for r in results if r and 'f1'  in r]
    aucs = [r['auc'] for r in results if r and 'auc' in r]
    mccs = [r['mcc'] for r in results if r and 'mcc' in r]

    print(f"    Done {len(f1s)}/{n_trials} | "
          f"F1={np.mean(f1s):.3f} AUC={np.mean(aucs):.3f} "
          f"MCC={np.mean(mccs):.3f}", flush=True)

    if not f1s:
        return {k: 0.0 for k in ['f1_mean','f1_median','f1_std',
                                   'auc_mean','auc_median','auc_std',
                                   'mcc_mean','mcc_median','mcc_std']}
    return {
        'f1_mean':  np.mean(f1s),  'f1_median':  np.median(f1s),  'f1_std':  np.std(f1s),
        'auc_mean': np.mean(aucs), 'auc_median': np.median(aucs), 'auc_std': np.std(aucs),
        'mcc_mean': np.mean(mccs), 'mcc_median': np.median(mccs), 'mcc_std': np.std(mccs),
    }


def run_full_benchmark(data_dir, src_root=None, feature_mode='metrics',
                       top_k=3, n_trials=30, cache_dir='embeddings',
                       graphsage_epochs=100, **trial_kwargs):
    print(f"Loading projects (feature_mode='{feature_mode}') ...")
    all_projects = load_all_projects(
        data_dir, src_root, feature_mode, cache_dir, graphsage_epochs)
    if len(all_projects) < 2:
        raise ValueError("Need at least 2 projects.")

    rows = []
    for name in sorted(all_projects.keys()):
        res = evaluate_target(name, all_projects, top_k, n_trials,
                              **trial_kwargs)
        rows.append({'project': name, **res})
        print(f"  -> {name}: F1={res['f1_mean']:.3f} "
              f"AUC={res['auc_mean']:.3f} MCC={res['mcc_mean']:.3f}")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Statistical Tests
# ─────────────────────────────────────────────────────────────
PAPER_F1 = {
    'ant': 0.632, 'camel-1.0': 0.472, 'camel-1.2': 0.496,
    'camel-1.4': 0.481, 'camel-1.6': 0.517, 'ivy-1.1': 0.604,
    'ivy-1.4': 0.377, 'ivy-2.0': 0.547, 'log4j-1.0': 0.557,
    'log4j-1.1': 0.620, 'log4j-1.2': 0.831, 'lucene-2.0': 0.667,
    'lucene-2.2': 0.665, 'lucene-2.4': 0.693, 'poi-1.5': 0.718,
    'poi-2.0': 0.501, 'poi-2.5': 0.757, 'poi-3.0': 0.791,
    'xalan-2.4': 0.463, 'xalan-2.5': 0.592, 'xalan-2.6': 0.701,
    'xalan-2.7': 0.793, 'xerces-1.2': 0.393, 'xerces-1.3': 0.502,
    'xerces-1.4': 0.903,
}


def cliffs_delta(x, y):
    """Cliff's delta effect size between two samples."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0, 'negligible'
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    d = (more - less) / (n_x * n_y)
    abs_d = abs(d)
    if abs_d < 0.147:
        level = 'negligible'
    elif abs_d < 0.33:
        level = 'small'
    elif abs_d < 0.474:
        level = 'medium'
    else:
        level = 'large'
    return d, level


def print_statistical_comparison(df):
    """Print per-project comparison and statistical tests vs base paper."""
    print("\n" + "="*70)
    print("PER-PROJECT F1 COMPARISON VS TRISTAGE-CPDP")
    print("="*70)
    print(f"{'Project':<18} {'Ours':>8} {'Paper':>8} {'Diff':>8} {'Status':>8}")
    print("-" * 52)

    our_vals = []
    paper_vals = []
    for _, row in df.iterrows():
        proj = row['project']
        ours = row['f1_mean']
        paper = PAPER_F1.get(proj, None)
        if paper is None:
            continue
        our_vals.append(ours)
        paper_vals.append(paper)
        diff = ours - paper
        status = '+' if diff >= 0 else '-'
        print(f"{proj:<18} {ours:>8.3f} {paper:>8.3f} {diff:>+8.3f} {status:>8}")

    our_arr = np.array(our_vals)
    paper_arr = np.array(paper_vals)

    if len(our_arr) >= 5:
        try:
            w_stat, p_val = stats.wilcoxon(our_arr, paper_arr)
        except ValueError:
            w_stat, p_val = 0.0, 1.0
        d, level = cliffs_delta(our_arr.tolist(), paper_arr.tolist())

        print(f"\n  Wilcoxon signed-rank: W={w_stat:.1f}, p={p_val:.4f}"
              f" {'(significant)' if p_val < 0.05 else '(not significant)'}")
        print(f"  Cliff's delta: d={d:.3f} ({level})")
        print(f"  Mean diff:  F1={np.mean(our_arr - paper_arr):+.3f}")


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Adaptive Cross-Project Defect Prediction Pipeline')

    # Data / feature args
    parser.add_argument('--data_dir',         default='data/')
    parser.add_argument('--src_root',         default=None,
        help='Parent dir of Java source folders (needed for full/graphsage/codet5 modes)')
    parser.add_argument('--feature_mode',     default='metrics',
        choices=['full','metrics','metrics+graphsage','metrics+codet5'])
    parser.add_argument('--target',           default=None)
    parser.add_argument('--graphsage_epochs', type=int, default=100)
    parser.add_argument('--cache_dir',        default='embeddings')
    parser.add_argument('--out',              default='results.csv')

    # Experiment control
    parser.add_argument('--trials',           type=int,   default=30)
    parser.add_argument('--top_k',            type=int,   default=3)
    parser.add_argument('--target_label_ratio', type=float, default=0.10)
    parser.add_argument('--n_folds',          type=int,   default=5)

    # Source selection weights
    parser.add_argument('--w_cos',            type=float, default=0.40)
    parser.add_argument('--w_mmd',            type=float, default=0.35)
    parser.add_argument('--w_adist',          type=float, default=0.15)
    parser.add_argument('--w_dr',             type=float, default=0.10)

    # Domain adaptation
    parser.add_argument('--coral_reg',        type=float, default=1e-3)

    # Imbalance handling
    parser.add_argument('--smote_strategy',   default='auto',
        help="'auto' for full balance, or float like 0.5 / 0.7")

    # Grid search
    parser.add_argument('--grid_search',      action='store_true',
        help='Run grid search over key hyperparameters')
    parser.add_argument('--grid_target',      default=None,
        help='Target project for grid search (default: use all)')
    parser.add_argument('--grid_trials',      type=int, default=10,
        help='Trials per config during grid search (fewer for speed)')

    args = parser.parse_args()

    trial_kwargs = dict(
        target_label_ratio=args.target_label_ratio,
        coral_reg=args.coral_reg,
        smote_strategy=args.smote_strategy,
        n_folds=args.n_folds,
        w_cos=args.w_cos, w_mmd=args.w_mmd,
        w_adist=args.w_adist, w_dr=args.w_dr,
    )

    all_projects = load_all_projects(
        args.data_dir, args.src_root, args.feature_mode,
        args.cache_dir, args.graphsage_epochs)
    print(f"\nProjects: {list(all_projects.keys())}")

    if args.grid_search:
        from itertools import product
        grid = {
            'top_k':          [3, 5, 7],
            'coral_reg':      [1e-4, 1e-3, 1e-2],
            'smote_strategy': ['auto', '0.5', '0.7'],
            'n_folds':        [3, 5],
        }
        keys = list(grid.keys())
        combos = list(product(*grid.values()))
        targets = [args.grid_target] if args.grid_target else sorted(all_projects.keys())

        print(f"\nGrid search: {len(combos)} configs x {len(targets)} targets "
              f"x {args.grid_trials} trials")
        grid_rows = []
        best_f1, best_cfg = 0.0, None

        for combo in combos:
            cfg = dict(zip(keys, combo))
            kw = {**trial_kwargs, **cfg}
            f1_all = []
            for tgt in targets:
                res = evaluate_target(tgt, all_projects,
                                      cfg['top_k'], args.grid_trials, **kw)
                f1_all.append(res['f1_mean'])
            avg_f1 = np.mean(f1_all)
            grid_rows.append({**cfg, 'avg_f1': avg_f1})
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_cfg = cfg
            print(f"  {cfg} -> avg F1={avg_f1:.3f}")

        grid_df = pd.DataFrame(grid_rows).sort_values('avg_f1', ascending=False)
        grid_df.to_csv('grid_results.csv', index=False)
        print(f"\nBest config (F1={best_f1:.3f}): {best_cfg}")
        print("Grid results saved to grid_results.csv")
        return

    if args.target:
        if args.target not in all_projects:
            print(f"ERROR: '{args.target}' not found.")
            sys.exit(1)
        res = evaluate_target(args.target, all_projects,
                              args.top_k, args.trials, **trial_kwargs)
        print(f"\nResults for {args.target}:")
        for k, v in res.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    else:
        df = run_full_benchmark(
            args.data_dir, args.src_root, args.feature_mode,
            args.top_k, args.trials, args.cache_dir, args.graphsage_epochs,
            **trial_kwargs)

        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(df[['project','f1_mean','f1_std',
                  'auc_mean','auc_std','mcc_mean','mcc_std']].to_string(index=False))

        print(f"\nAverages: F1={df['f1_mean'].mean():.3f}  "
              f"AUC={df['auc_mean'].mean():.3f}  "
              f"MCC={df['mcc_mean'].mean():.3f}")

        baseline = {'f1': 0.627, 'auc': 0.663, 'mcc': 0.245}
        our      = {m: df[f'{m}_mean'].mean() for m in ['f1','auc','mcc']}
        print("\nVs TriStage-CPDP (base paper):")
        for m in ['f1','auc','mcc']:
            d = (our[m] - baseline[m]) / baseline[m] * 100
            print(f"  {m.upper()}: Ours={our[m]:.3f}  Paper={baseline[m]:.3f}  "
                  f"{'▲' if d>0 else '▼'}{abs(d):.1f}%")

        print_statistical_comparison(df)

        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()