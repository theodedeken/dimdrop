from sklearn.neural_network import BernoulliRBM
import numpy as np


class EarlyStoppingRBM:
    """
    Adaptation on the `BernoulliRBM` class of sklearn to add the ability to
    stop early when training does not improve.

    Parameters
    ----------
    n_components : int, optional
        The size of the output, default `256`
    batch_size : int, optional
        The batch size of the rbm, default `100`
    lr : float, optional
        The learning rate of the rbm, default `0.01`
    patience : int, optional
        The amount of epochs without improvement before training stops,
        default `3`
    epochs : int, optional
        The maximum amount of epochs, default `1000`
    verbose : int, optional
        The verbosity of the rbm, default `0`

    Attributes
    ----------
    rbm : BernoulliRBM
        the rbm to be trained
    """

    def __init__(
        self,
        n_components=256,
        batch_size=100,
        lr=0.01,
        patience=3,
        epochs=1000,
        verbose=0
    ):
        self.rbm = BernoulliRBM(
            n_components=n_components,
            n_iter=1,
            batch_size=batch_size,
            learning_rate=lr,
            verbose=verbose
        )
        self.patience = patience
        self.epochs = epochs
        self.verbose = verbose

    def fit(self, data):
        """
        Fit the rbm to the given data

        Parameters
        ----------
        data : array
            Data to be fitted
        """
        self.rbm.fit(data)
        min_likelyhood = np.mean(
            [np.mean(self.rbm.score_samples(data)) for _ in range(5)])
        last_likelyhood = min_likelyhood
        min_index = 0
        for i in range(1, self.epochs):
            if min_index + self.patience > i:
                if self.verbose:
                    print('Epoch {}/{}'.format(i + 1, self.epochs))
                self.rbm.fit(data)
                last_likelyhood = np.mean(
                    [np.mean(self.rbm.score_samples(data)) for _ in range(5)])
                if last_likelyhood < min_likelyhood:
                    min_likelyhood = last_likelyhood
                    min_index = i
            else:
                break
