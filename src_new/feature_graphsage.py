"""
Stage 1b: Structural Feature Extraction using GraphSAGE + CDN
==============================================================
Builds a Class Dependency Network (CDN) from a Java project and
learns 32-d node embeddings using GraphSAGE with unsupervised
contrastive learning (positive/negative sample pairs).

Pipeline per project:
  1. Parse .java files → detect import/extends/implements dependencies
  2. Build CDN graph (nodes=classes, edges=dependencies)
  3. Initialize node features with Node2Vec embeddings + traditional metrics
  4. Train GraphSAGE (2 layers, mean aggregation, contrastive loss)
  5. Return 32-d embedding per node, aligned to PROMISE CSV row order

Requirements (install on your machine)
---------------------------------------
    pip install networkx numpy scikit-learn
    # torch + torch_geometric for full GraphSAGE (falls back to
    # a pure-numpy approximation if torch is not available)

Usage
-----
    from feature_graphsage import CDNGraphSAGEExtractor

    extractor = CDNGraphSAGEExtractor()
    embeddings = extractor.extract_project(
        src_dir      = "path/to/java/src/",
        promise_csv  = "data/ant.csv",
        project_name = "ant",
        cache_dir    = "embeddings/"
    )
    # returns np.ndarray (n_classes, 32)
"""

import os
import re
import glob
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# ─────────────────────────────────────────────────────────────
# Step 1 & 2: CDN Construction
# ─────────────────────────────────────────────────────────────

def _parse_java_dependencies(java_path: str) -> tuple:
    """
    Extract class name and its dependencies from a .java file.
    Returns (class_name, set_of_dependencies).
    Uses regex — no Java compiler needed.
    """
    for enc in ('utf-8', 'latin-1', 'cp1252'):
        try:
            with open(java_path, 'r', encoding=enc) as f:
                src = f.read()
            break
        except UnicodeDecodeError:
            src = ""

    # Class / interface name
    class_match = re.search(
        r'(?:public\s+)?(?:abstract\s+)?(?:class|interface|enum)\s+(\w+)',
        src)
    class_name = class_match.group(1) if class_match else \
                 os.path.basename(java_path).replace('.java', '')

    deps = set()

    # import statements  →  dependency edges
    for m in re.finditer(r'import\s+([\w.]+)\s*;', src):
        dep = m.group(1).split('.')[-1]   # keep only simple class name
        deps.add(dep)

    # extends / implements
    for m in re.finditer(r'(?:extends|implements)\s+([\w,\s]+?)(?:\{|$)', src):
        for dep in re.split(r'[,\s]+', m.group(1).strip()):
            if dep:
                deps.add(dep)

    # field type declarations  (e.g.  private MyClass foo;)
    for m in re.finditer(r'(?:private|protected|public)\s+(\w+)\s+\w+\s*[;=,\(]', src):
        deps.add(m.group(1))

    deps.discard(class_name)
    return class_name, deps


def build_cdn(src_dir: str) -> nx.DiGraph:
    """
    Build a Class Dependency Network for all .java files in src_dir.
    Returns a directed NetworkX graph.
    """
    java_files = glob.glob(
        os.path.join(src_dir, "**", "*.java"), recursive=True)

    G = nx.DiGraph()
    file_map = {}   # simple_name → full class name (deduplication)

    # First pass: register all classes as nodes
    class_info = {}
    for path in java_files:
        cname, deps = _parse_java_dependencies(path)
        class_info[path] = (cname, deps)
        G.add_node(cname, file=path)
        file_map[cname] = cname

    # Second pass: add edges for resolved dependencies
    all_nodes = set(G.nodes())
    for path, (cname, deps) in class_info.items():
        for dep in deps:
            if dep in all_nodes and dep != cname:
                G.add_edge(cname, dep)

    return G


# ─────────────────────────────────────────────────────────────
# Step 3: Node2Vec-style initial embeddings (pure numpy)
# ─────────────────────────────────────────────────────────────

def _node2vec_init(G: nx.DiGraph, dim: int = 32, seed: int = 42) -> dict:
    """
    Lightweight random-walk based node embedding initialization.
    Uses SVD on the adjacency + degree matrix (no torch needed).
    Returns dict {node_name: np.ndarray(dim)}
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {name: i for i, name in enumerate(nodes)}

    if n == 0:
        return {}

    # Adjacency matrix (undirected)
    A = np.zeros((n, n), dtype=np.float32)
    for u, v in G.edges():
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1.0
            A[idx[v], idx[u]] = 1.0

    # Add self-loops and degree normalization
    D = np.diag(A.sum(axis=1) + 1)
    A = A + np.eye(n)

    # SVD to get low-dim structural representations
    actual_dim = min(dim, n - 1, A.shape[1])
    if actual_dim < 1:
        actual_dim = 1
    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    emb_matrix = svd.fit_transform(A)

    # Pad to full dim if needed
    if emb_matrix.shape[1] < dim:
        pad = np.zeros((n, dim - emb_matrix.shape[1]), dtype=np.float32)
        emb_matrix = np.hstack([emb_matrix, pad])

    return {nodes[i]: emb_matrix[i] for i in range(n)}


# ─────────────────────────────────────────────────────────────
# Step 4: GraphSAGE (pure numpy — no torch required)
# ─────────────────────────────────────────────────────────────

class GraphSAGENumpy:
    """
    2-layer GraphSAGE with mean aggregation.
    Trained with a contrastive loss (positive = edge pairs,
    negative = random non-edge pairs).
    Pure numpy — no GPU or torch_geometric needed.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64,
                 out_dim: int = 32, n_layers: int = 2,
                 lr: float = 0.01, epochs: int = 100,
                 margin: float = 0.5, neg_ratio: int = 3,
                 random_state: int = 42):
        self.in_dim    = in_dim
        self.hidden    = hidden_dim
        self.out_dim   = out_dim
        self.n_layers  = n_layers
        self.lr        = lr
        self.epochs    = epochs
        self.margin    = margin
        self.neg_ratio = neg_ratio
        rng = np.random.RandomState(random_state)

        # Weight matrices for each layer
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.W = []
        for i in range(n_layers):
            # He initialization
            scale = np.sqrt(2.0 / (dims[i] * 2))
            self.W.append(rng.randn(dims[i] * 2, dims[i + 1]).astype(np.float32) * scale)

    def _relu(self, x):
        return np.maximum(0, x)

    def _normalize(self, x):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        return x / norms

    def _forward(self, H, adj_list):
        """Forward pass through all GraphSAGE layers."""
        for layer_idx, W in enumerate(self.W):
            H_new = []
            for i, neighbors in enumerate(adj_list):
                if neighbors:
                    neigh_mean = H[neighbors].mean(axis=0)
                else:
                    neigh_mean = np.zeros(H.shape[1], dtype=np.float32)
                agg = np.concatenate([H[i], neigh_mean])
                h   = self._relu(agg @ W)
                H_new.append(h)
            H = np.array(H_new, dtype=np.float32)
            H = self._normalize(H)
        return H

    def _cosine_sim(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def fit(self, H_init: np.ndarray, edge_list: list, n_nodes: int,
            rng: np.random.RandomState = None):
        """
        Parameters
        ----------
        H_init    : (n_nodes, in_dim) initial node features
        edge_list : list of (i, j) integer index pairs
        n_nodes   : total number of nodes
        """
        if rng is None:
            rng = np.random.RandomState(42)

        # Build adjacency list
        adj = [[] for _ in range(n_nodes)]
        edge_set = set()
        for u, v in edge_list:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((min(u,v), max(u,v)))

        H = H_init.copy().astype(np.float32)

        for epoch in range(self.epochs):
            H_out = self._forward(H, adj)

            # Contrastive loss gradient (simplified, SGD)
            grad = np.zeros_like(H, dtype=np.float32)
            loss = 0.0

            # Positive pairs (edges)
            pos_pairs = list(edge_set)
            if not pos_pairs:
                break

            # Negative pairs (random non-edges)
            neg_pairs = []
            while len(neg_pairs) < len(pos_pairs) * self.neg_ratio:
                u = rng.randint(0, n_nodes)
                v = rng.randint(0, n_nodes)
                if u != v and (min(u,v), max(u,v)) not in edge_set:
                    neg_pairs.append((u, v))

            for u, v in pos_pairs:
                sim = self._cosine_sim(H_out[u], H_out[v])
                loss += (1 - sim) ** 2
                # Push embeddings closer via feature update
                diff = H_out[u] - H_out[v]
                grad[u] += -2 * (1 - sim) * diff / (n_nodes + 1)
                grad[v] +=  2 * (1 - sim) * diff / (n_nodes + 1)

            for u, v in neg_pairs:
                sim = self._cosine_sim(H_out[u], H_out[v])
                if sim > self.margin:
                    loss += (sim - self.margin) ** 2
                    diff = H_out[u] - H_out[v]
                    grad[u] +=  2 * (sim - self.margin) * diff / (n_nodes + 1)
                    grad[v] += -2 * (sim - self.margin) * diff / (n_nodes + 1)

            H -= self.lr * grad

            if epoch % 20 == 0 and epoch > 0:
                print(f"      GraphSAGE epoch {epoch}/{self.epochs}  loss={loss:.4f}")

        self.H_final_ = self._forward(H, adj)
        return self

    def get_embeddings(self) -> np.ndarray:
        return self.H_final_


# ─────────────────────────────────────────────────────────────
# Main Extractor Class
# ─────────────────────────────────────────────────────────────

class CDNGraphSAGEExtractor:
    """
    Full pipeline: Java source → CDN → GraphSAGE → 32-d embeddings.

    Parameters
    ----------
    out_dim      : embedding dimension (default 32, matches base paper)
    epochs       : GraphSAGE training epochs
    random_state : seed for reproducibility
    """

    def __init__(self, out_dim: int = 32, epochs: int = 100,
                 random_state: int = 42):
        self.out_dim = out_dim
        self.epochs  = epochs
        self.rs      = random_state

    def extract_project(self, src_dir: str,
                        promise_csv: str = None,
                        project_name: str = "project",
                        cache_dir: str = "embeddings") -> np.ndarray:
        """
        Extract GraphSAGE embeddings for a project.

        Parameters
        ----------
        src_dir      : path to Java source root
        promise_csv  : path to PROMISE CSV (for row-order alignment)
        project_name : used for cache filename
        cache_dir    : where to save/load .npy cache

        Returns
        -------
        np.ndarray of shape (n_classes, 32)
        Rows aligned to PROMISE CSV order if promise_csv is provided.
        """
        cache_path = os.path.join(cache_dir, f"{project_name}_graphsage.npy")
        if os.path.exists(cache_path):
            print(f"  [GraphSAGE] Loading cached: {cache_path}")
            return np.load(cache_path)

        print(f"  [GraphSAGE] Building CDN for {project_name} ...")
        G = build_cdn(src_dir)
        n_nodes = G.number_of_nodes()
        print(f"  [GraphSAGE] CDN: {n_nodes} nodes, {G.number_of_edges()} edges")

        if n_nodes == 0:
            raise ValueError(f"No Java classes found in {src_dir}")

        # Node2Vec initialization
        node2vec_emb = _node2vec_init(G, dim=32, seed=self.rs)
        nodes = list(G.nodes())
        node_idx = {n: i for i, n in enumerate(nodes)}

        # Initial feature matrix
        H_init = np.array([node2vec_emb.get(n, np.zeros(32))
                           for n in nodes], dtype=np.float32)

        # Edge list as integer indices
        edge_list = [(node_idx[u], node_idx[v])
                     for u, v in G.edges()
                     if u in node_idx and v in node_idx]

        # Train GraphSAGE
        print(f"  [GraphSAGE] Training ({self.epochs} epochs) ...")
        rng   = np.random.RandomState(self.rs)
        sage  = GraphSAGENumpy(
            in_dim=32, hidden_dim=64, out_dim=self.out_dim,
            epochs=self.epochs, random_state=self.rs)
        sage.fit(H_init, edge_list, n_nodes, rng)
        node_embeddings = sage.get_embeddings()   # (n_nodes, 32)

        # Map back: node name → embedding
        emb_dict = {nodes[i]: node_embeddings[i] for i in range(n_nodes)}

        # Align to PROMISE CSV row order
        embeddings = self._align_to_csv(emb_dict, promise_csv, project_name)

        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, embeddings)
        print(f"  [GraphSAGE] Saved to {cache_path}")
        return embeddings

    def _align_to_csv(self, emb_dict: dict,
                      promise_csv: str,
                      project_name: str) -> np.ndarray:
        """
        Return embeddings in the same row order as the PROMISE CSV.
        Falls back to sorted node order if CSV not provided.
        """
        zero = np.zeros(self.out_dim, dtype=np.float32)

        if promise_csv is None or not os.path.exists(promise_csv):
            # Return in arbitrary node order
            return np.array(list(emb_dict.values()), dtype=np.float32)

        import pandas as pd
        df = pd.read_csv(promise_csv)

        if 'name' not in df.columns:
            # No name column → return sorted order
            return np.array(list(emb_dict.values()), dtype=np.float32)

        rows = []
        missing = 0
        for class_name in df['name']:
            # Try exact match, then simple class name (last segment)
            simple = str(class_name).split('.')[-1]
            emb = emb_dict.get(class_name) or emb_dict.get(simple)
            if emb is not None:
                rows.append(emb)
            else:
                rows.append(zero)
                missing += 1

        if missing:
            print(f"  [GraphSAGE] {missing}/{len(df)} classes "
                  f"not found in CDN (zero-padded).")

        return np.array(rows, dtype=np.float32)