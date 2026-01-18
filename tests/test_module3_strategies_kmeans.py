import numpy as np
import pytest

from beie.module3.strategies import KMeansClustering


def test_kmeans_fit_predict_shapes_and_types():
    X = np.array(
        [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1],
        ],
        dtype=float,
    )

    strat = KMeansClustering(n_clusters=2, random_state=42)
    strat.fit(X)
    labels = strat.predict(X)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)
    assert labels.dtype.kind in ("i",)  # integer labels


def test_kmeans_is_deterministic_with_fixed_seed():
    X = np.array(
        [
            [10.0, 0.0],
            [10.1, 0.1],
            [0.0, 10.0],
            [0.1, 10.1],
        ],
        dtype=float,
    )

    s1 = KMeansClustering(n_clusters=2, random_state=123)
    s2 = KMeansClustering(n_clusters=2, random_state=123)

    s1.fit(X)
    s2.fit(X)

    l1 = s1.predict(X)
    l2 = s2.predict(X)

    assert np.array_equal(l1, l2)


def test_kmeans_predict_without_fit_raises():
    X = np.array([[1.0, 2.0]], dtype=float)
    strat = KMeansClustering(n_clusters=1)

    with pytest.raises(RuntimeError):
        strat.predict(X)
