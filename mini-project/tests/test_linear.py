from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from layers.linear import Linear


def test_initialization_shapes() -> None:
    layer = Linear(in_features=3, out_features=2)

    assert layer.weight.shape == (3, 2)
    assert layer.bias.shape == (2,)
    assert layer.grad_weight.shape == (3, 2)
    assert layer.grad_bias.shape == (2,)
    assert layer.input is None

def test_forward_computes_affine_transform() -> None:
    layer = Linear(in_features=2, out_features=3)
    layer.weight = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    layer.bias = np.array([0.5, -0.5, 1.0], dtype=np.float64)

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    output = layer.forward(x)

    expected = np.array(
        [[9.5, 11.5, 16.0], [19.5, 25.5, 34.0]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(output, expected)
    np.testing.assert_array_equal(layer.input, x)

def test_backward_computes_gradients_and_input_grad() -> None:
    layer = Linear(in_features=2, out_features=2)
    layer.weight = np.array(
        [[1.0, 2.0], [3.0, 4.0]],
        dtype=np.float64,
    )
    layer.bias = np.array([0.0, 0.0], dtype=np.float64)

    x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    grad_output = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

    layer.forward(x)
    grad_input = layer.backward(grad_output)

    expected_grad_weight = np.array(
        [[26.0, 30.0], [38.0, 44.0]],
        dtype=np.float64,
    )
    expected_grad_bias = np.array([12.0, 14.0], dtype=np.float64)
    expected_grad_input = np.array(
        [[17.0, 39.0], [23.0, 53.0]],
        dtype=np.float64,
    )

    np.testing.assert_allclose(layer.grad_weight, expected_grad_weight)
    np.testing.assert_allclose(layer.grad_bias, expected_grad_bias)
    np.testing.assert_allclose(grad_input, expected_grad_input)

def test_backward_requires_forward_first() -> None:
    layer = Linear(in_features=2, out_features=2)

    with pytest.raises(RuntimeError, match="forward must be called before backward"):
        layer.backward(np.array([[1.0, 2.0]], dtype=np.float64))

def test_zero_grad_clears_cached_gradients() -> None:
    layer = Linear(in_features=2, out_features=2)
    layer.grad_weight = np.ones((2, 2), dtype=np.float64)
    layer.grad_bias = np.ones((2,), dtype=np.float64)

    layer.zero_grad()

    np.testing.assert_array_equal(layer.grad_weight, np.zeros((2, 2)))
    np.testing.assert_array_equal(layer.grad_bias, np.zeros((2,)))
