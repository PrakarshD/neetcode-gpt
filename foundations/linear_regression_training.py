import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64],
     N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights_ = initial_weights
        assert(len(weights_) == X.shape[1])

        for idx_x in range(num_iterations):

            model_pred = self.get_model_prediction(X,weights_)

            # Grad descent for each weight across each sample 
            for idx_y in range(X.shape[1]):
                grad_idx_y = self.get_derivative(model_pred, Y, X.shape[0], X, idx_y)
                weights_[idx_y] -= self.learning_rate * grad_idx_y

        return np.round(weights_,5)



