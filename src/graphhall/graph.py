from __future__ import annotations

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.csgraph import laplacian as normalized_laplacian
from scipy.sparse.linalg import eigsh


def build_knn_graph(
    embeddings: np.ndarray,
    *,
    k: int,
    threshold: float = 0.8,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Build directed incoming-neighbor -> query cosine k-NN edges in bounded chunks."""
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise ValueError("embeddings must have shape (N>=2, D>=1)")
    if not np.isfinite(x).all():
        raise ValueError("embeddings must contain only finite values")
    if np.any(np.linalg.norm(x, axis=1) == 0):
        raise ValueError("embeddings must not contain zero-norm rows")
    if k < 1:
        raise ValueError("k must be positive")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    n_nodes = x.shape[0]
    n_neighbors = min(k, n_nodes - 1)
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []

    for start in range(0, n_nodes, chunk_size):
        stop = min(start + chunk_size, n_nodes)
        similarities = x[start:stop] @ x.T
        for local_row, query in enumerate(range(start, stop)):
            similarities[local_row, query] = -np.inf
            row = similarities[local_row]
            candidates = np.argpartition(-row, n_neighbors - 1)[:n_neighbors]
            candidates = sorted(candidates.tolist(), key=lambda index: (-float(row[index]), index))
            for neighbor in candidates:
                similarity = float(row[neighbor])
                if similarity >= threshold:
                    sources.append(neighbor)
                    targets.append(query)
                    weights.append(similarity)

    edge_index = np.asarray([sources, targets], dtype=np.int64).reshape(2, -1)
    return edge_index, np.asarray(weights, dtype=np.float32)


def laplacian_positional_encoding(
    edge_index: np.ndarray,
    *,
    num_nodes: int,
    dimensions: int = 10,
    edge_weight: np.ndarray | None = None,
    dense_limit: int = 512,
) -> np.ndarray:
    """Return the first nonzero normalized-Laplacian eigenvectors, sign-fixed and padded."""
    edges = np.asarray(edge_index, dtype=np.int64)
    if num_nodes < 1 or dimensions < 1:
        raise ValueError("num_nodes and dimensions must be positive")
    if edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    if edges.size and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("edge_index contains an invalid node index")
    if edge_weight is None:
        weights = np.ones(edges.shape[1], dtype=np.float64)
    else:
        weights = np.asarray(edge_weight, dtype=np.float64)
        if weights.shape != (edges.shape[1],) or not np.isfinite(weights).all():
            raise ValueError("edge_weight must be finite and have one value per edge")

    adjacency = sparse.coo_matrix(
        (weights, (edges[0], edges[1])), shape=(num_nodes, num_nodes), dtype=np.float64
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    lap = normalized_laplacian(adjacency, normed=True).tocsr()

    if num_nodes <= dense_limit:
        eigenvalues, eigenvectors = linalg.eigh(lap.toarray(), check_finite=True)
    else:
        max_pairs = min(128, num_nodes - 1)
        pairs = min(max(dimensions + 1, 32), max_pairs)
        while True:
            eigenvalues, eigenvectors = eigsh(
                lap,
                k=pairs,
                which="SM",
                tol=1e-5,
                v0=np.random.default_rng(42).standard_normal(num_nodes),
            )
            order = np.argsort(eigenvalues)
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            if np.count_nonzero(eigenvalues > 1e-8) >= dimensions or pairs >= max_pairs:
                break
            pairs = min(max_pairs, pairs * 2)

    selected = eigenvectors[:, eigenvalues > 1e-8][:, :dimensions].copy()
    for column in range(selected.shape[1]):
        pivot = int(np.argmax(np.abs(selected[:, column])))
        if selected[pivot, column] < 0:
            selected[:, column] *= -1
    result = np.zeros((num_nodes, dimensions), dtype=np.float32)
    result[:, : selected.shape[1]] = selected.astype(np.float32)
    return result
