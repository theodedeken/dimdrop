import math
from keras.utils import Sequence


class DECSequence(Sequence):
    """
    Sequence generator for the DEC network

    Parameters
    ----------
    data : array
        The data to generate the sequence for
    model : `dimdrop.models.DEC`
        The model to train
    batch_size : int
        The batch size

    Attributes
    ----------
    target : array
        The current target distribution
    """

    def __init__(self, data, model, batch_size):
        self.data = data
        self.model = model
        self.batch_size = batch_size
        self.target = target_distribution(self.model.predict(self.data))

    def __len__(self):
        return math.ceil(self.data.shape[0] / self.batch_size)

    def __getitem__(self, index):
        idx = slice(
            index * self.batch_size,
            min((index+1) * self.batch_size, self.target.shape[0])
        )
        return (self.data[idx], self.target[idx])

    def on_epoch_end(self):
        """
        After each epoch update the target distribution.
        """
        self.target = target_distribution(self.model.predict(self.data))


def target_distribution(q):
    """
    Calculate the target distribution

    Attributes
    ----------
    q : array
        the input data

    Returns
    -------
    target distribution
    """
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
