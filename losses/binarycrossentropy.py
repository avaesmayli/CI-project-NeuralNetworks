import numpy as np

class BinaryCrossEntropy:
    def __init__(self):
        pass

    def compute(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        batch_size = y.shape[1]

        # Compute the binary cross entropy loss
        cost = -(1 / batch_size) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

        return np.squeeze(cost)

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Compute the derivative of the binary cross entropy loss
        d_loss = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

        return d_loss
