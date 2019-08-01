import numpy as np

from .autoencoder import Autoencoder
from ..regularizers import KMeansRegularizer


class DeepCluster(Autoencoder):
    def __init__(
            self,
            in_dim,
            out_dim,
            k,
            layer_sizes=[2000, 1000, 500],
            lr=0.01,
            scale=True,
            log=False,
            batch_size=100,
            patience=3,
            epochs=1000,
            regularizer='kmeans',
            regularizer_weight=0.5,
            decay=True,
            verbose=0
    ):
        if regularizer == 'kmeans':
            regularizer_obj = KMeansRegularizer(
                np.random.rand(k, out_dim), weight=regularizer_weight)
        elif regularizer == 'gmm':
            # TODO add gmm cluster regularizer
            regularizer_obj = None
        else:
            raise AttributeError
        super().__init__(
            in_dim,
            out_dim,
            layer_sizes=layer_sizes,
            lr=lr,
            scale=scale,
            log=log,
            batch_size=batch_size,
            patience=patience,
            epochs=epochs,
            regularizer=regularizer_obj,
            pretrain_method=None,
            decay=decay
            verbose=verbose
        )

    def get_cluster_centers(self):
        return self.regularizer.cluster_centers

    def get_cluster_assignments(self):
        return self.regularizer.cluster_assignments
