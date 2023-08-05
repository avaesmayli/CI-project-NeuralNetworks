import numpy as np

class MaxPool2D:
    def __init__(self, kernel_size=(3, 3), stride=(1, 1), mode="max"):
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.mode = mode

    def target_shape(self, input_shape):
        H_prev, W_prev, _ = input_shape
        f_h, f_w = self.kernel_size
        stride_h, stride_w = self.stride

        H = int((H_prev - f_h) / stride_h) + 1
        W = int((W_prev - f_w) / stride_w) + 1

        return H, W

    def forward(self, A_prev):
        batch_size, H_prev, W_prev, C_prev = A_prev.shape
        f_h, f_w = self.kernel_size
        stride_h, stride_w = self.stride

        H, W = self.target_shape(A_prev.shape)

        A = np.zeros((batch_size, H, W, C_prev))

        for i in range(batch_size):
            for h in range(H):
                h_start = h * stride_h
                h_end = h_start + f_h

                for w in range(W):
                    w_start = w * stride_w
                    w_end = w_start + f_w

                    for c in range(C_prev):
                        a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]

                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)
                        else:
                            raise ValueError("Invalid mode")

        return A

    def create_mask_from_window(self, x):
        mask = x == np.max(x)
        return mask

    def distribute_value(self, dz, shape):
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones(shape) * average
        return a

    def backward(self, dA, A_prev):
        f_h, f_w = self.kernel_size
        stride_h, stride_w = self.stride

        batch_size, H_prev, W_prev, C_prev = A_prev.shape
        batch_size, H, W, C = dA.shape

        dA_prev = np.zeros((batch_size, H_prev, W_prev, C_prev))

        for i in range(batch_size):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        h_start = h * stride_h
                        h_end = h_start + f_h
                        w_start = w * stride_w
                        w_end = w_start + f_w

                        if self.mode == "max":
                            a_prev_slice = A_prev[i, h_start:h_end, w_start:w_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += np.multiply(mask, dA[i, h, w, c])
                        elif self.mode == "average":
                            dz = dA[i, h, w, c]
                            shape = (f_h, f_w)
                            dA_prev[i, h_start:h_end, w_start:w_end, c] += self.distribute_value(dz, shape)
                        else:
                            raise ValueError("Invalid mode")

        return dA_prev, None
