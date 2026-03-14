"""
Stage 1a: Semantic & Syntactic Feature Extraction using CodeT5+
================================================================
CodeT5+ (Salesforce/codet5p-110m-embedding) extracts a 256-d embedding
for each source file, capturing:
  - Syntactic structure  (function defs, control flow, variable declarations)
  - Semantic meaning     (logical purpose, inter-function dependencies)
  - Contextual patterns  (code idioms, common bug-prone constructs)

Usage
-----
    from feature_codet5 import CodeT5Extractor

    extractor = CodeT5Extractor()                   # loads model once
    embeddings = extractor.extract_project("path/to/java/src/")
    # returns np.ndarray of shape (n_files, 256)

Offline cache
-------------
Embeddings are saved to  <project_root>/embeddings/<project_name>_codet5.npy
so the expensive model forward pass runs only once per project.

Requirements (install on your machine)
---------------------------------------
    pip install torch transformers
    # Model auto-downloads on first run (~450 MB):
    # https://huggingface.co/Salesforce/codet5p-110m-embedding
"""

import os
import glob
import numpy as np

# ── lazy imports so the module loads even if torch is absent ──
def _import_torch():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
        return torch, AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "PyTorch and transformers are required for CodeT5+ extraction.\n"
            "Install with:  pip install torch transformers")


MODEL_NAME = "Salesforce/codet5p-110m-embedding"
EMBED_DIM  = 256
MAX_TOKENS = 512


class CodeT5Extractor:
    """
    Extracts 256-d semantic-syntactic embeddings from Java source files
    using a pre-trained CodeT5+ model (no fine-tuning needed).
    """

    def __init__(self, device: str = None, batch_size: int = 8):
        """
        Parameters
        ----------
        device     : 'cuda' | 'cpu' | None (auto-detect)
        batch_size : files processed per forward pass (reduce if OOM)
        """
        torch, AutoModel, AutoTokenizer = _import_torch()
        self.torch = torch

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.batch_size = batch_size

        print(f"  [CodeT5+] Loading model on {device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
        self.model.eval()
        print(f"  [CodeT5+] Model ready.")

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────
    def extract_project(self, src_dir: str,
                        cache_path: str = None) -> np.ndarray:
        """
        Extract embeddings for all .java files in src_dir.

        Parameters
        ----------
        src_dir    : path to the project's Java source directory
        cache_path : if given, save/load numpy array here

        Returns
        -------
        np.ndarray of shape (n_files, 256)
        """
        if cache_path and os.path.exists(cache_path):
            print(f"  [CodeT5+] Loading cached embeddings from {cache_path}")
            return np.load(cache_path)

        java_files = sorted(glob.glob(
            os.path.join(src_dir, "**", "*.java"), recursive=True))

        if not java_files:
            raise FileNotFoundError(
                f"No .java files found in {src_dir}")

        print(f"  [CodeT5+] Extracting from {len(java_files)} files ...")
        embeddings = self._embed_files(java_files)

        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.save(cache_path, embeddings)
            print(f"  [CodeT5+] Saved embeddings to {cache_path}")

        return embeddings

    def extract_strings(self, code_snippets: list) -> np.ndarray:
        """
        Extract embeddings from a list of code strings directly.
        Useful when you have per-class code already extracted.

        Returns np.ndarray of shape (n_snippets, 256)
        """
        return self._embed_batch(code_snippets)

    # ─────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────
    def _read_file(self, path: str) -> str:
        for enc in ('utf-8', 'latin-1', 'cp1252'):
            try:
                with open(path, 'r', encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return ""   # fallback: empty string

    def _embed_files(self, file_paths: list) -> np.ndarray:
        snippets = [self._read_file(p) for p in file_paths]
        return self._embed_batch(snippets)

    def _embed_batch(self, snippets: list) -> np.ndarray:
        torch = self.torch
        all_embeddings = []

        for i in range(0, len(snippets), self.batch_size):
            batch = snippets[i: i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # CodeT5p-110m-embedding returns last_hidden_state
            # Mean-pool over token dimension → (batch, 256)
            hidden = outputs.last_hidden_state          # (B, T, 256)
            mask   = inputs['attention_mask'].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            all_embeddings.append(pooled.cpu().numpy())

            if (i // self.batch_size + 1) % 10 == 0:
                print(f"    {i + len(batch)}/{len(snippets)} files done")

        return np.vstack(all_embeddings).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# PROMISE-aware helper
# ─────────────────────────────────────────────────────────────
def extract_and_align_to_promise(project_name: str,
                                  src_dir: str,
                                  promise_csv_path: str,
                                  cache_dir: str = "embeddings") -> np.ndarray:
    """
    Extract CodeT5+ embeddings and align them to PROMISE CSV row order.

    PROMISE CSVs are per-class (one row per Java class).
    This function maps each CSV row to its corresponding .java file
    using the class name, then extracts embeddings in the same order.

    Parameters
    ----------
    project_name     : e.g. 'ant'
    src_dir          : path to Java source root
    promise_csv_path : path to the PROMISE CSV (must have a 'name' column)
    cache_dir        : directory to cache .npy files

    Returns
    -------
    np.ndarray of shape (n_rows, 256)  — aligned to CSV row order
    """
    import pandas as pd

    df = pd.read_csv(promise_csv_path)

    # PROMISE uses 'name' column with fully-qualified class names
    # e.g. 'org.apache.tools.ant.Main'
    if 'name' not in df.columns:
        print("  [WARN] No 'name' column in CSV — using file order instead.")
        extractor = CodeT5Extractor()
        cache_path = os.path.join(cache_dir, f"{project_name}_codet5.npy")
        return extractor.extract_project(src_dir, cache_path)

    cache_path = os.path.join(cache_dir, f"{project_name}_codet5.npy")
    if os.path.exists(cache_path):
        print(f"  [CodeT5+] Loading cached: {cache_path}")
        return np.load(cache_path)

    # Build a map: class_name → java file path
    java_files = glob.glob(
        os.path.join(src_dir, "**", "*.java"), recursive=True)
    file_map = {}
    for path in java_files:
        # Convert path to class-name style
        rel = os.path.relpath(path, src_dir)
        class_key = rel.replace(os.sep, '.').replace('.java', '')
        file_map[class_key] = path
        # Also store just the filename stem for fuzzy matching
        file_map[os.path.basename(path).replace('.java', '')] = path

    extractor = CodeT5Extractor()
    snippets = []
    missing  = 0
    for class_name in df['name']:
        # Try exact match first, then stem match
        path = file_map.get(class_name) or \
               file_map.get(class_name.split('.')[-1])
        if path:
            snippets.append(extractor._read_file(path))
        else:
            snippets.append("")   # missing file → zero embedding
            missing += 1

    if missing:
        print(f"  [WARN] {missing}/{len(df)} classes had no matching .java file "
              f"(will produce zero embeddings for those rows).")

    embeddings = extractor.extract_strings(snippets)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache_path, embeddings)
    print(f"  [CodeT5+] Saved to {cache_path}")

    return embeddings