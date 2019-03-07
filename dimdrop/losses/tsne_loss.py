import numpy as np

from keras import backend as K


class TSNELoss:
    """
    Custom keras loss function implementing the cost function of t-SNE

    Parameters
    ----------
    dim : int
        The dimension of the output of the network
    batch_size : int
        The batch size of the network
    """
    __name__ = 'tsne_loss'

    def __init__(self, dim, batch_size):
        self.dim = dim
        self.batch_size = batch_size

    def __call__(self, y_true, y_pred):
        d = self.dim
        n = self.batch_size
        v = d - 1.

        eps = K.variable(10e-15)
        sum_act = K.sum(K.square(y_pred), axis=1)
        Q = K.reshape(sum_act, [-1, 1]) + -2 * \
            K.dot(y_pred, K.transpose(y_pred))
        Q = (sum_act + Q) / v
        Q = K.pow(1 + Q, -(v + 1) / 2)
        Q *= K.variable(1 - np.eye(n))
        Q /= K.sum(Q)
        Q = K.maximum(Q, eps)
        C = K.log((y_true + eps) / (Q + eps))
        C = K.sum(y_true * C)
        return C
