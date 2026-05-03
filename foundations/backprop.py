import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z: NDArray[np.float64]):
        return 1/(1 + np.exp(-1*z))

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = w.T @ x + b 
        y_pred = self.sigmoid(z)

        err = y_pred - y_true # scalar here
        loss = (err.T * err) * 0.5

        # dL_dw = dL_dz  * dz_dw ; dz_dw = x vector as z = w.x + b; dL_dz = err x y_pred(1-y_pred)
        dL_dz = err * y_pred * (1 - y_pred)
        dz_dw = x

        dL_dw = np.round(dL_dz * dz_dw,5)
        dL_db = np.round(dL_dz * 1,5)
        return dL_dw, dL_db
