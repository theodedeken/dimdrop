import numpy as np


class Transform:
    def __init__(self, scale=True, log=False):
        self.scale = scale
        self.log = log

    def __call__(self, data):
        if self.log:
            data = np.log2(data + 1)
        if self.scale:
            for i in range(data.shape[0]):
                data[i, :] = data[i, :] / np.max(data[i, :])
        return data
