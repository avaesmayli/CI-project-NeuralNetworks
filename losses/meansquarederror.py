import numpy as np

class MeanSquaredError:
    def __init__(self):
        pass

    def compute(self, y_pred, y_true):
        batch_size = y_pred.shape[1]

        # Compute the mean squared error loss
        cost = (1 / (2 * batch_size)) * np.sum(np.square(y_pred - y_true))

        return np.squeeze(cost)

    def backward(self, y_pred, y_true):
        # Compute the derivative of the mean squared error loss
        d_loss = (1 / y_pred.shape[1]) * (y_pred - y_true)

        return d_loss
