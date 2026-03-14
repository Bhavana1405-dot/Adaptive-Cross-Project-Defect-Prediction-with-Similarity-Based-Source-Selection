"""
Stage 1: Feature Extraction Orchestrator
=========================================
Combines three complementary feature types into a single joint vector:

    XP = concat(XC, XS, XT)   shape: (n_instances, 256 + 32 + 20) = (n, 308)

Where:
    XC  — CodeT5+ semantic/syntactic embeddings        (256-d)
    XS  — GraphSAGE structural embeddings from CDN     (32-d)
    XT  — Traditional PROMISE metrics                  (20-d)

This matches the base paper (TriStage-CPDP) Stage 1 exactly, so your
results are directly comparable to their Table 5/6/7.

Usage
-----
Two modes:

Mode A — Full (CodeT5+ + GraphSAGE + metrics):
    from feature_extraction import FeatureExtractor
    extractor = FeatureExtractor(mode='full')
    X = extractor.extract(
        promise_csv  = "data/ant.csv",
        src_dir      = "java_src/ant-1.7/src/",
        project_name = "ant",
        cache_dir    = "embeddings/"
    )
    # X shape: (745, 308)

Mode B — Metrics only (no Java source needed, faster):
    extractor = FeatureExtractor(mode='metrics')
    X = extractor.extract(promise_csv="data/ant.csv")
    # X shape: (745, 20)

Mode C — Partial (e.g. metrics + GraphSAGE only):
    extractor = FeatureExtractor(mode='metrics+graphsage')
    X = extractor.extract(promise_csv="data/ant.csv",
                          src_dir="java_src/ant-1.7/src/",
                          project_name="ant")

The pipeline.py calls this module — set mode via --feature_mode flag.
"""

import os
import numpy as np
import pandas as pd


LABEL_CANDIDATES = ['bug', 'defects', 'label', 'Defective',
                    'class', 'bugs', 'isDefective']

VALID_MODES = {'full', 'metrics', 'metrics+graphsage', 'metrics+codet5'}


class FeatureExtractor:
    """
    Parameters
    ----------
    mode : str
        'full'              → CodeT5+ + GraphSAGE + metrics  (308-d)
        'metrics+graphsage' → GraphSAGE + metrics             (52-d)
        'metrics+codet5'    → CodeT5+ + metrics               (276-d)
        'metrics'           → traditional metrics only         (20-d)
    codet5_batch_size : int — reduce if running out of GPU memory
    graphsage_epochs  : int — more epochs = better embeddings (slower)
    cache_dir         : str — where to store precomputed embeddings
    """

    def __init__(self,
                 mode: str = 'full',
                 codet5_batch_size: int = 8,
                 graphsage_epochs: int = 100,
                 random_state: int = 42,
                 cache_dir: str = 'embeddings'):
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}")
        self.mode              = mode
        self.codet5_batch_size = codet5_batch_size
        self.graphsage_epochs  = graphsage_epochs
        self.rs                = random_state
        self.cache_dir         = cache_dir

        self._codet5_model   = None   # lazy-loaded
        self._graphsage_ext  = None   # lazy-loaded

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────
    def extract(self,
                promise_csv: str,
                src_dir: str = None,
                project_name: str = None) -> np.ndarray:
        """
        Extract and concatenate features for one project.

        Parameters
        ----------
        promise_csv  : path to PROMISE CSV file
        src_dir      : path to Java source root (needed for CodeT5+/GraphSAGE)
        project_name : used for cache filenames (defaults to CSV stem)

        Returns
        -------
        np.ndarray of shape (n_instances, D)
        """
        if project_name is None:
            project_name = os.path.basename(promise_csv).replace('.csv', '')

        needs_src = ('codet5' in self.mode or 'graphsage' in self.mode
                     or self.mode == 'full')
        if needs_src and src_dir is None:
            raise ValueError(
                f"src_dir is required for mode='{self.mode}'. "
                f"Use mode='metrics' if you don't have Java source files.")

        # ── Traditional metrics (always extracted) ──────────
        XT = self._extract_metrics(promise_csv)
        n  = len(XT)
        print(f"  [Stage 1] {project_name}: {n} instances, "
              f"metrics shape {XT.shape}")

        feature_parts = [XT]

        # ── CodeT5+ embeddings ───────────────────────────────
        if 'codet5' in self.mode or self.mode == 'full':
            XC = self._extract_codet5(
                promise_csv, src_dir, project_name, n)
            print(f"  [Stage 1] CodeT5+ shape: {XC.shape}")
            feature_parts.insert(0, XC)   # XC first, matches paper ordering

        # ── GraphSAGE structural embeddings ──────────────────
        if 'graphsage' in self.mode or self.mode == 'full':
            XS = self._extract_graphsage(
                src_dir, promise_csv, project_name, n)
            print(f"  [Stage 1] GraphSAGE shape: {XS.shape}")
            # Insert after CodeT5+ (or at front if no CodeT5+)
            insert_pos = 1 if ('codet5' in self.mode or self.mode == 'full') else 0
            feature_parts.insert(insert_pos, XS)

        # ── Concatenate: XP = concat(XC, XS, XT) ────────────
        XP = np.hstack(feature_parts)
        print(f"  [Stage 1] Joint features: {XP.shape}  "
              f"(mode='{self.mode}')")
        return XP.astype(np.float32)

    def extract_all_projects(self,
                             data_dir: str,
                             src_root: str = None) -> dict:
        """
        Extract features for every project CSV in data_dir.

        Parameters
        ----------
        data_dir : directory containing PROMISE CSV files
        src_root : parent directory containing per-project Java source folders
                   Expected layout:
                       src_root/
                         ant-1.7/src/
                         camel-1.6/src/
                         ...
                   (None = metrics-only mode)

        Returns
        -------
        dict {project_name: np.ndarray}
        """
        projects = {}
        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.csv'):
                continue
            name = fname.replace('.csv', '')
            csv_path = os.path.join(data_dir, fname)

            src_dir = None
            if src_root:
                # Try common naming patterns
                for candidate in [name, name.replace('-', '_'),
                                  name.split('-')[0]]:
                    for sub in ['', 'src', 'src/main/java', 'source']:
                        path = os.path.join(src_root, candidate, sub)
                        if os.path.isdir(path):
                            src_dir = path
                            break
                    if src_dir:
                        break
                if src_dir is None and self.mode != 'metrics':
                    print(f"  [WARN] No src_dir found for {name} under {src_root}. "
                          f"Falling back to metrics only for this project.")

            try:
                X = self.extract(csv_path, src_dir, name)
                projects[name] = X
            except Exception as e:
                print(f"  [WARN] Skipping {name}: {e}")

        return projects

    # ─────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────
    def _extract_metrics(self, promise_csv: str) -> np.ndarray:
        """Extract the 20 traditional PROMISE metrics."""
        df = pd.read_csv(promise_csv)

        # Drop label column(s)
        drop_cols = [c for c in LABEL_CANDIDATES if c in df.columns]
        # Also drop non-numeric and identifier columns
        drop_cols += [c for c in ['name', 'version', 'name.1']
                      if c in df.columns]
        X = df.drop(columns=drop_cols).select_dtypes(include='number')

        # Fill any NaN with column median
        X = X.fillna(X.median())
        return X.values.astype(np.float32)

    def _extract_codet5(self, promise_csv, src_dir,
                        project_name, expected_n) -> np.ndarray:
        """Extract CodeT5+ embeddings, with caching."""
        cache_path = os.path.join(
            self.cache_dir, f"{project_name}_codet5.npy")

        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            if len(emb) == expected_n:
                return emb
            print(f"  [CodeT5+] Cache shape mismatch, re-extracting ...")

        # Lazy-load the extractor
        if self._codet5_model is None:
            from feature_codet5 import CodeT5Extractor
            self._codet5_model = CodeT5Extractor(
                batch_size=self.codet5_batch_size)

        from feature_codet5 import extract_and_align_to_promise
        emb = extract_and_align_to_promise(
            project_name, src_dir, promise_csv, self.cache_dir)

        if len(emb) != expected_n:
            print(f"  [WARN] CodeT5+ returned {len(emb)} rows, "
                  f"expected {expected_n}. Zero-padding.")
            emb = _pad_or_trim(emb, expected_n, emb.shape[1])

        return emb.astype(np.float32)

    def _extract_graphsage(self, src_dir, promise_csv,
                           project_name, expected_n) -> np.ndarray:
        """Extract GraphSAGE embeddings, with caching."""
        cache_path = os.path.join(
            self.cache_dir, f"{project_name}_graphsage.npy")

        if os.path.exists(cache_path):
            emb = np.load(cache_path)
            if len(emb) == expected_n:
                return emb
            print(f"  [GraphSAGE] Cache shape mismatch, re-extracting ...")

        if self._graphsage_ext is None:
            from feature_graphsage import CDNGraphSAGEExtractor
            self._graphsage_ext = CDNGraphSAGEExtractor(
                out_dim=32,
                epochs=self.graphsage_epochs,
                random_state=self.rs)

        emb = self._graphsage_ext.extract_project(
            src_dir=src_dir,
            promise_csv=promise_csv,
            project_name=project_name,
            cache_dir=self.cache_dir)

        if len(emb) != expected_n:
            print(f"  [WARN] GraphSAGE returned {len(emb)} rows, "
                  f"expected {expected_n}. Zero-padding.")
            emb = _pad_or_trim(emb, expected_n, emb.shape[1])

        return emb.astype(np.float32)


def _pad_or_trim(arr: np.ndarray, target_rows: int, cols: int) -> np.ndarray:
    """Pad with zeros or trim rows to match target_rows."""
    if len(arr) < target_rows:
        pad = np.zeros((target_rows - len(arr), cols), dtype=np.float32)
        return np.vstack([arr, pad])
    return arr[:target_rows]