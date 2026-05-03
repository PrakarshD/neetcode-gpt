import numpy as np
from numpy.typing import NDArray


class Solution:
    def sigmoid(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1/(1 + np.exp(-1 * x))

    def relu(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return max(0.0, x)

    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z = w.T @ x + b
        if activation == 'relu':
            return np.round(self.relu(z),5)
        elif activation == 'sigmoid':
            return np.round(self.sigmoid(z),5)
        else:
            print("invalid activation")
            return

        
