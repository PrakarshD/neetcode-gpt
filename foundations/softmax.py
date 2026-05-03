import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        
        z_norm = z - np.max(z)
        softmax_z_num = np.exp(z_norm)
        softmax_z_den = np.sum(softmax_z_num)
        return np.round(softmax_z_num / softmax_z_den,4)
