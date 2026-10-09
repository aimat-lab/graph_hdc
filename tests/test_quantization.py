import numpy as np
import pytest

from graph_hdc.utils import quantize_equal_frequency


@pytest.mark.parametrize('bits', [1, 2, 4])
def test_quantize_equal_frequency_levels_and_balance(bits):
    """Each column has at most 2**bits levels, and the training values are split into equally full bins."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(1000, 8))
    train_rows = np.arange(800)
    quantized = quantize_equal_frequency(features, train_rows, bits)
    assert quantized.shape == features.shape and quantized.dtype == np.float32
    for col in range(features.shape[1]):
        levels, counts = np.unique(quantized[train_rows, col], return_counts=True)
        assert len(levels) == 2 ** bits
        assert counts.min() >= 0.9 * len(train_rows) / 2 ** bits
        # the order of the values is preserved (quantization is monotonic)
        order = np.argsort(features[:, col])
        assert np.all(np.diff(quantized[order, col]) >= 0)


def test_quantize_equal_frequency_one_bit_is_median_split():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(500, 3))
    train_rows = np.arange(500)
    quantized = quantize_equal_frequency(features, train_rows, 1)
    for col in range(3):
        median = np.median(features[:, col])
        low = features[:, col] <= median
        assert np.allclose(quantized[low, col], features[low, col].mean(), atol=1e-6)
        assert np.allclose(quantized[~low, col], features[~low, col].mean(), atol=1e-6)


def test_quantize_equal_frequency_fitted_on_training_rows_only():
    """Test rows do not influence the bins: changing them leaves the training rows unchanged."""
    rng = np.random.default_rng(2)
    features = rng.normal(size=(300, 4))
    train_rows = np.arange(200)
    first = quantize_equal_frequency(features, train_rows, 2)
    changed = features.copy()
    changed[200:] *= 100
    second = quantize_equal_frequency(changed, train_rows, 2)
    assert np.array_equal(first[train_rows], second[train_rows])


def test_quantize_equal_frequency_more_levels_than_training_rows():
    """With 2**bits >= training rows, training values are kept and other values snap to the nearest one."""
    train = np.array([[0.0], [1.0], [3.0]])
    features = np.vstack([train, [[0.4], [2.1], [10.0], [-5.0]]])
    quantized = quantize_equal_frequency(features, np.arange(3), 16)
    assert np.allclose(quantized[:, 0], [0.0, 1.0, 3.0, 0.0, 3.0, 3.0, 0.0])


def test_quantize_equal_frequency_error_decreases_with_bits():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(2000, 16))
    train_rows = np.arange(1600)
    errors = [np.abs(quantize_equal_frequency(features, train_rows, b) - features).mean() for b in (1, 2, 4, 8)]
    assert all(a > b for a, b in zip(errors, errors[1:]))


def test_quantize_equal_frequency_rejects_non_finite_values():
    features = np.array([[0.0], [1.0], [np.nan]])
    with pytest.raises(ValueError):
        quantize_equal_frequency(features, np.arange(3), 1)
