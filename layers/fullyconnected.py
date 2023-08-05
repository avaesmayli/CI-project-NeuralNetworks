import numpy as np

class FC:
    def __init__(self, input_size: int, output_size: int, name: str, initialize_method: str = "random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None

    def initialize_weights(self):
        if self.initialize_method == "random":
            return np.random.randn(self.input_size, self.output_size) * 0.01
        elif self.initialize_method == "xavier":
            xavier_stddev = np.sqrt(1 / self.input_size)
            return np.random.randn(self.input_size, self.output_size) * xavier_stddev
        elif self.initialize_method == "he":
            he_stddev = np.sqrt(2 / self.input_size)
            return np.random.randn(self.input_size, self.output_size) * he_stddev
        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        return np.zeros((1, self.output_size))

    def forward(self, A_prev):
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)

        if len(self.input_shape) > 2:
            batch_size = self.input_shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
        self.reshaped_shape = A_prev_tmp.shape

        W, b = self.parameters
        Z = np.dot(A_prev_tmp.T, W) + b
        return Z

    def backward(self, dZ, A_prev):
        A_prev_tmp = np.copy(A_prev)
        if len(self.input_shape) > 2:
            batch_size = self.input_shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T

        W, b = self.parameters
        dW = np.dot(A_prev_tmp, dZ) / batch_size
        db = np.sum(dZ, axis=0, keepdims=True) / batch_size
        dA_prev = np.dot(dZ, W.T)
        grads = [dW, db]

        if len(self.input_shape) > 2:
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads

    def update_parameters(self, optimizer, grads):
        self.parameters = optimizer.update(grads, self.name)
