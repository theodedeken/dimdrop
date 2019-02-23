from ..losses import TSNELoss
from keras.callbacks import Callback


class TSNERegularizer(Callback):
    __name__ = 'tsne_regularizer'

    def __init__(self, dim, joint_prob, batch_size=100):
        self.dim = dim
        self.joint_prob = joint_prob
        self.batch_size = batch_size
        self.batch = 0
        self.loss = TSNELoss(self.dim, self.batch_size)

    def on_batch_begin(self, batch, logs={}):
        self.batch = batch

    def __call__(self, activations):
        return self.loss(self.joint_prob[self.batch], activations)
