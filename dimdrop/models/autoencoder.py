from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping
from sklearn.neural_network import BernoulliRBM
import numpy as np


class Autoencoder:
    """
    A deep autoencoder model as baseline for other autoencoder based dimensionality reduction methods.

    The defaults are set to the parameters explained in the paper of Geoffrey Hinton.

    References
    ----------
    - TODO
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        layer_sizes=[2000, 1000, 500],
        lr=0.01,
        batch_size=100,
        patience=3,
        epochs=1000,
        regularizer=None,
        pretrain_method='rbm',
        verbose=0
    ):
        """
        Initialize the autoencoder.

        Parameters
        ----------
        in_dim : int 
            The input dimension
        out_dim : int
            The output dimension
        layer_sizes : array of int, optional
            The sizes of the layers of the network, is mirrored over encoder en decoder parts, default `[2000, 1000, 500]`
        lr : float, optional 
            The learning rate of the network, default `0.01`
        batch_size : int, optional
            The batch size of the network, default `100`
        patience : int, optional
            The amount of epochs without improvement before the network stops training, default `3`
        epochs : int, optional
            The maximum amount of epochs, default `1000`
        regularizer : keras regularizer, optional
            A regularizer to use for the middle layer of the autoencoder. None or instance of KMeansRegularizer, GMMRegularizer, TSNERegularizer.
        pretrain_method : string, optional
            The pretrain method to use. None, `'rbm'` or `'stacked'`
        verbose : int, optional
            The verbosity of the network
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.epochs = epochs
        self.regularizer = regularizer
        self.pretrain_method = pretrain_method
        self.verbose = verbose
        self.__pretrainers = {'rbm': self.__pretrain_rbm}
        self.__init_network()

    def __init_network(self):
        activation = 'sigmoid' if self.pretrain_method == 'rbm' else 'relu'
        self.layers = [
            Dense(self.layer_sizes[0], activation=activation, input_shape=(self.in_dim,))]
        self.layers += [Dense(size, activation=activation)
                        for size in self.layer_sizes[1:]]
        self.layers += [Dense(self.out_dim, activity_regularizer=self.regularizer)
                        ] if self.regularizer else [Dense(self.out_dim)]
        self.layers += [Dense(size, activation=activation)
                        for size in self.layer_sizes[::-1]]
        self.layers += [Dense(self.in_dim)]

        self.model = Sequential(
            self.layers
        )
        self.model.compile(
            loss='mse',
            optimizer=SGD(lr=self.lr)
        )
        self.encoder = Sequential(
            self.layers[:len(self.layers) // 2]
        )
        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=self.lr)
        )

    def __pretrain_rbm(self, data):
        rbms = [BernoulliRBM(batch_size=self.batch_size, learning_rate=self.lr,
                             n_components=num, n_iter=20, verbose=self.verbose) for num in self.layer_sizes + [self.out_dim]]
        current = data
        for i, rbm in enumerate(rbms):
            if self.verbose:
                print('Training RBM {}/{}'.format(i + 1, len(rbms)))
            rbm.fit(current)
            current = rbm.transform(current)
            dec_layer = self.layers[len(self.layers) - 1 - i]
            enc_layer = self.layers[i]
            dec_layer.set_weights(
                [rbm.components_, dec_layer.get_weights()[1]])
            enc_layer.set_weights(
                [np.transpose(rbm.components_), enc_layer.get_weights()[1]])

    def fit(self, data):
        """
        Fit the given data to the model.

        Parameters
        ----------
        data : array
            Array of training samples where each sample is of size `in_dim`
        """
        if self.pretrain_method:
            self.__pretrainers[self.pretrain_method](data)

        early_stopping = EarlyStopping(monitor='loss', patience=self.patience)
        callbacks = [early_stopping]
        if self.regularizer:
            self.regularizer.init_fit(self.encoder, data)
            callbacks.append(self.regularizer)
        self.model.fit(data, data, epochs=self.epochs, callbacks=callbacks,
                       batch_size=self.batch_size, verbose=self.verbose)

    def transform(self, data):
        """
        Transform the given data

        Parameters
        ----------
        data : array
            Array of samples to be transformed, where each sample is of size `in_dim`

        Returns
        -------
        array
            Transformed samples, where each sample is of size `out_dim`
        """
        return self.encoder.predict(data, verbose=self.verbose)

    def fit_transform(self, data):
        """
        Fit the given data to the model and return its transformation

        Parameters
        ----------
        data : array
            Array of training samples where each sample is of size `in_dim`

        Returns
        -------
        array
            Transformed samples, where each sample is of size `out_dim`
        """
        self.fit(data)
        return self.transform(data)