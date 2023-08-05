import numpy as np
from abc import ABC, abstractmethod

class Activation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, Z: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        pass

class Sigmoid(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = 1 / (1 + np.exp(-Z))
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        A = self.forward(Z)
        dZ = dA * A * (1 - A)
        return dZ

class ReLU(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = np.maximum(0, Z)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = np.where(Z > 0, dA, 0)
        return dZ

class Tanh(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = np.tanh(Z)
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        A = self.forward(Z)
        dZ = dA * (1 - np.square(A))
        return dZ

class LinearActivation(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        A = Z
        return A

    def backward(self, dA: np.ndarray, Z: np.ndarray) -> np.ndarray:
        dZ = dA
        return dZ

def get_activation(activation: str) -> tuple:
    if activation == 'sigmoid':
        return Sigmoid(), Sigmoid()
    elif activation == 'relu':
        return ReLU(), ReLU()
    elif activation == 'tanh':
        return Tanh(), Tanh()
    elif activation == 'linear':
        return LinearActivation(), LinearActivation()
    else:
        raise ValueError('Activation function not supported')
