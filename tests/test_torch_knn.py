"""
Tests for TorchKNNRegressor (predict_molecules.py, KN_BACKEND='torch', ex_27): the exact brute-force search must agree
with sklearn's KNeighborsRegressor wherever the nearest neighbors are unique, and break exact ties by training order.
"""
import os
import sys
import importlib.util

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors

FINGERPRINTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'graph_hdc', 'experiments', 'fingerprints')


@pytest.fixture(scope='module')
def knn_class():
    # the experiment modules are scripts, not a package; pycomex needs the module registered in sys.modules
    spec = importlib.util.spec_from_file_location('predict_molecules',
                                                  os.path.join(FINGERPRINTS, 'predict_molecules.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TorchKNNRegressor


@pytest.mark.parametrize('metric', ['minkowski', 'cosine'])
def test_continuous_metrics_match_sklearn(knn_class, metric):
    rng = np.random.default_rng(0)
    X_train, X_test = rng.normal(size=(300, 32)), rng.normal(size=(50, 32))
    y_train = rng.normal(size=(300, 1))
    expected = KNeighborsRegressor(n_neighbors=5, metric=metric, algorithm='brute').fit(X_train, y_train).predict(X_test)
    actual = knn_class(n_neighbors=5, metric=metric, batch_size=16).fit(X_train, y_train).predict(X_test)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_jaccard_distances_match_sklearn(knn_class):
    """The Tanimoto similarities equal one minus sklearn's Jaccard distances (bit vectors with many ties)."""
    import torch
    rng = np.random.default_rng(1)
    X_train, X_test = rng.random((200, 64)) < 0.1, rng.random((40, 64)) < 0.1
    distances, _ = NearestNeighbors(n_neighbors=7, metric='jaccard', algorithm='brute').fit(X_train).kneighbors(X_test)
    model = knn_class(n_neighbors=7, metric='jaccard').fit(X_train.astype(float), np.zeros(200))
    similarity = model.similarity(torch.tensor(X_test, dtype=torch.float64, device=model.device))
    np.testing.assert_allclose(np.sort(1 - similarity.topk(7, dim=1).values.cpu().numpy(), axis=1),
                               np.sort(distances, axis=1), rtol=0, atol=1e-12)


def test_ties_choose_the_first_training_samples(knn_class):
    """Five identical training fingerprints at the same distance: the first k of them are the neighbors."""
    query = np.array([[1, 1, 0, 0]], dtype=float)
    X_train = np.array([[1, 0, 0, 0]] * 5 + [[0, 0, 1, 1]], dtype=float)   # first five: Tanimoto 0.5, last: 0
    y_train = np.arange(6, dtype=float)[:, None]
    prediction = knn_class(n_neighbors=3, metric='jaccard').fit(X_train, y_train).predict(query)
    assert prediction[0, 0] == pytest.approx(np.mean([0, 1, 2]))


def test_distinct_near_ties_are_not_reordered(knn_class):
    """Only exact ties are broken by the training order: a neighbor that is closer by 1e-11 wins at any position."""
    query = np.array([[1.0, 0.0]])
    X_train = np.array([[1.0, np.sqrt(2e-11)]] + [[0.0, 1.0]] * 999 + [[1.0, 0.0]])  # cosine 1 - 1e-11, 0, ..., 1
    y_train = np.arange(1001, dtype=float)[:, None]
    prediction = knn_class(n_neighbors=1, metric='cosine', batch_size=1).fit(X_train, y_train).predict(query)
    assert prediction[0, 0] == 1000.0


def test_empty_fingerprints_are_identical(knn_class):
    X_train = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    prediction = knn_class(n_neighbors=1, metric='jaccard').fit(X_train, np.array([[3.0], [7.0]])).predict(
        np.zeros((1, 3)))
    assert prediction[0, 0] == 3.0
