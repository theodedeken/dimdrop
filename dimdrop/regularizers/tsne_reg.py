from ..losses import TSNELoss
from keras.callbacks import Callback


class TSNERegularizer(Callback):
    """
    Regularizer using the tsne loss function.

    Parameters
    ----------
    dim : int,
        the dimension of the layer on which the regularizer is working
    joint_prob : array
        The joint probabilities of the input data
    batch_size : int
        The batch size of the training, default `100`.

    Attributes
    ----------
    batch : int
        The current batch
    loss : `dimdrop.losses.TSNELoss`
        The t-SNE loss function.
    """
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
