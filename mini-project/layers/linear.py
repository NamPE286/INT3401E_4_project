import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Linear:
    weight: FloatArray
    bias: FloatArray
    input: FloatArray | None
    grad_weight: FloatArray
    grad_bias: FloatArray

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        limit = np.sqrt(6.0 / (in_features + out_features))

        self.weight = np.random.uniform(
            low=-limit,
            high=limit,
            size=(in_features, out_features),
        ).astype(np.float64)

        self.bias = np.zeros(
            shape=(out_features,),
            dtype=np.float64,
        )

        self.input = None

        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x: FloatArray) -> FloatArray:
        self.input = x

        return x @ self.weight + self.bias

    def backward(self, grad_output: FloatArray) -> FloatArray:
        if self.input is None:
            raise RuntimeError("forward must be called before backward")

        x: FloatArray = self.input

        x_2d: FloatArray = x.reshape(-1, x.shape[-1])
        grad_2d: FloatArray = grad_output.reshape(
            -1,
            grad_output.shape[-1],
        )

        self.grad_weight = x_2d.T @ grad_2d
        self.grad_bias = grad_2d.sum(axis=0)

        grad_input: FloatArray = grad_output @ self.weight.T

        return grad_input

    def zero_grad(self) -> None:
        self.grad_weight.fill(0.0)
        self.grad_bias.fill(0.0)
