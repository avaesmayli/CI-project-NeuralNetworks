from layers.convolution2d import Conv2D
from layers.maxpooling2d import MaxPool2D
from layers.fullyconnected import FC

from activations import Activation, get_activation

import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle


class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        if name is None:
            self.model = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)

    def is_layer(self, layer):
        return isinstance(layer, (Conv2D, FC))

    def is_activation(self, layer):
        return isinstance(layer, Activation)

    def forward(self, x):
        tmp = []
        A = x
        for l in range(len(self.layers_names)):
            layer = self.model[self.layers_names[l]]
            if self.is_layer(layer):
                Z = layer.forward(A)
                tmp.append(Z.copy())
                A = layer.activation.forward(Z)
                tmp.append(A.copy())
            elif self.is_activation(layer):
                Z = tmp[-2]
                A = layer.forward(Z)
                tmp.append(Z.copy())
                tmp.append(A.copy())
        return tmp

    def backward(self, dAL, tmp, x):
        dA = dAL
        grads = {}
        for l in range(len(self.layers_names), 0, -1):
            layer = self.model[self.layers_names[l - 1]]
            if self.is_layer(layer):
                if l > 2:
                    Z, A = tmp[l - 1], tmp[l - 2]
                else:
                    Z, A = tmp[l - 1], x
                dZ = layer.activation.backward(dA, Z)
                dA, grad = layer.backward(dZ, A)
                grads[self.layers_names[l - 1]] = grad
        return grads

    def update(self, grads):
        for layer_name, layer in self.model.items():
            if self.is_layer(layer) and not isinstance(layer, MaxPool2D):
                layer.update(grads[layer_name])

    def one_epoch(self, x, y):
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion.forward(AL, y)
        dAL = self.criterion.backward()
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss

    def save(self, name):
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)

    def load_model(self, name):
        with open(name, 'rb') as f:
            return pickle.load(f)

    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        last_index = min(index + batch_size, len(order))
        batch = order[index:last_index]
        if len(X.shape) == 4:
            bx = X[batch]
            by = y[batch]
            return bx, by
        else:
            bx = X[:, batch]
            by = y[:, batch]
            return bx, by

    def compute_loss(self, X, y, batch_size):
        m = X.shape[0] if len(X.shape) == 4 else X.shape[1]
        order = self.shuffle(m, True)
        cost = 0
        for b in range(m // batch_size):
            bx, by = self.batch(X, y, batch_size, b * batch_size, order)
            tmp = self.forward(bx)
            AL = tmp[-1]
            cost += self.criterion.forward(AL, by)
        return cost

    def train(self, X, y, epochs, val=None, batch_size=32, shuffling=False, verbose=1, save_after=None):
        train_cost = []
        val_cost = []
        m = X.shape[0] if len(X.shape) == 4 else X.shape[1]
        for e in tqdm.tqdm(range(epochs), desc='Epochs'):
            order = self.shuffle(m, shuffling)
            cost = 0
            for b in range(m // batch_size):
                bx, by = self.batch(X, y, batch_size, b * batch_size, order)
                cost += self.one_epoch(bx, by)
            train_cost.append(cost)
            if val is not None:
                val_cost.append(self.compute_loss(val[0], val[1], batch_size))
            if verbose != 0 and (e + 1) % verbose == 0:
                print(f"Epoch {e + 1}: train cost = {cost}")
                if val is not None:
                    print(f"Epoch {e + 1}: val cost = {val_cost[-1]}")
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost

    def predict(self, X):
        tmp = self.forward(X)
        return tmp[-1]
