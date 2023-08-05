class GD:
    def __init__(self, layers_list: dict, learning_rate: float):
        self.learning_rate = learning_rate
        self.layers = layers_list

    def update(self, grads, name):
        layer = self.layers[name]
        params = []

        for i in range(len(grads)):
            param = layer.parameters[i] - (self.learning_rate * grads[i])
            params.append(param)

        return params
