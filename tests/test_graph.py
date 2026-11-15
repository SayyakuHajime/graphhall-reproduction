import numpy as np
import pytest

from graphhall.graph import build_knn_graph, laplacian_positional_encoding


def test_knn_edges_are_directed_neighbor_to_query_and_thresholded() -> None:
    x = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)

    edge_index, edge_weight = build_knn_graph(x, k=1, threshold=0.8)

    assert edge_index.shape == (2, 2)
    assert set(map(tuple, edge_index.T.tolist())) == {(1, 0), (0, 1)}
    assert np.all(edge_weight > 0.8)


def test_knn_handles_empty_edges_k_at_least_n_and_invalid_features() -> None:
    x = np.eye(3, dtype=np.float32)
    edge_index, weights = build_knn_graph(x, k=20, threshold=1.01)
    assert edge_index.shape == (2, 0)
    assert weights.shape == (0,)

    edge_index, weights = build_knn_graph(x, k=20, threshold=-1.0)
    assert edge_index.shape == (2, 6)
    assert weights.shape == (6,)

    with pytest.raises(ValueError, match="zero-norm"):
        build_knn_graph(np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32), k=1)
    with pytest.raises(ValueError, match="finite"):
        build_knn_graph(np.asarray([[np.nan, 1.0], [1.0, 0.0]], dtype=np.float32), k=1)


def test_lpe_uses_nontrivial_spectrum_and_pads_isolated_graph_deterministically() -> None:
    edge_index = np.asarray([[1, 0], [0, 1]], dtype=np.int64)

    first = laplacian_positional_encoding(edge_index, num_nodes=3, dimensions=4)
    second = laplacian_positional_encoding(edge_index, num_nodes=3, dimensions=4)

    assert first.shape == (3, 4)
    np.testing.assert_allclose(first, second)
    assert np.all(np.isfinite(first))
    # The single nontrivial eigenvector is sign-fixed; the remaining columns pad with zero.
    np.testing.assert_allclose(first[:, 1:], 0.0, atol=1e-8)
