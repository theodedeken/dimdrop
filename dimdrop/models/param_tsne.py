from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.neural_network import BernoulliRBM
import numpy as np

from ..losses import TSNELoss
from ..util.tsne import compute_joint_probabilities


class ParametricTSNE:
    def __init__(self, in_dim, out_dim, layer_sizes=[500, 500, 2000], lr=0.01, batch_size=100, pretrain=False, perplexity=30, tol=1e-5, patience=3, epochs=1000, verbose=0):
        """
        Parametric t-SNE.

        Implementation of the parametric variant of t-distributed neighborhood embedding.

        Parameters
        ----------
        in_dim : int
            The input dimension
        out_dim : int
            The output dimension
        layer_sizes : array, optional
            sizes of each layer in the neural network, default is the structure proposed in the original paper, namely: `[500, 2000, 2000]`
        lr : float, optional
            The learning rate of the network, default `0.01`
        batch_size : int, optional
            The batch size of the network, default `100`
        pretrain : int, optional
            Whether to perform pretraining using Restricted Boltzmann Machines, default `False`
        perplexity : int, optional
            Perplexity parameter in the t-SNE formula, controls how many neighbors are considered in the local neighborhood, default `30`
        tol : float, optional
            Tolerance of the perplexity, default `1e-5`
        patience : int, optional
            The amount of epochs without improvement before fitting will stop early, default `3`
        epochs : int, optional
            Maximum amount of epochs, default `1000`
        verbose : int, optional
            Controls the verbosity of the model, default `0`


        References
        ----------
        - Laurens van der Maaten. Learning a parametric embedding by preserving local structure. 
        In David van Dyk and Max Welling, editors, *Proceedings of the Twelth International 
        Conference on Artificial Intelligence and Statistics*, volume 5 of *Proceedings of
        Machine Learning Research*, pages 384–391, Hilton Clearwater Beach Resort, Clearwater 
        Beach, Florida USA, 16–18 Apr 2009. PMLR.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.pretrain = pretrain
        self.perplexity = perplexity
        self.tol = tol
        self.patience = patience
        self.epochs = epochs
        self.verbose = verbose
        self.__init_network()

    def __init_network(self):
        if self.pretrain:
            self.rbms = [BernoulliRBM(batch_size=self.batch_size, learning_rate=self.lr,
                                      n_components=num, n_iter=20, verbose=self.verbose) for num in self.layer_sizes]
            activation = 'sigmoid'
        else:
            activation = 'relu'
        self.layers = []
        for i, num in enumerate(self.layer_sizes):
            if i == 0:
                self.layers.append(
                    Dense(num, activation=activation, input_shape=(self.in_dim,)))
            else:
                self.layers.append(Dense(num, activation=activation))
        self.layers.append(Dense(self.out_dim))

        self.model = Sequential(self.layers)

        optimizer = SGD(lr=self.lr)
        loss = TSNELoss(self.in_dim, self.batch_size)

        self.model.compile(
            optimizer=optimizer,
            loss=loss
        )

        if self.verbose:
            self.model.summary()

    def __pretrain(self, data):
        current = data
        for i, rbm in enumerate(self.rbms):
            if self.verbose:
                print('Training RBM {}/{}'.format(i + 1, len(self.rbms)))
            rbm.fit(current)
            current = rbm.transform(current)

    def fit(self, data):
        """
        Fit the given data to the model.

        Parameters
        ----------
        data : array
            Array of training samples where each sample is of size `in_dim`
        """
        if self.pretrain:
            if self.verbose:
                print('Pretraining network')
            self.__pretrain(data)

        if self.pretrain:
            for i, rbm in enumerate(self.rbms):
                self.layers[i].set_weights(
                    [np.transpose(rbm.components_), rbm.intercept_hidden_])

        early_stopping = EarlyStopping(monitor='loss', patience=self.patience)

        P = compute_joint_probabilities(
            data, batch_size=self.batch_size, d=self.layer_sizes[-1], perplexity=self.perplexity, tol=self.tol, verbose=self.verbose)
        y_train = P.reshape(data.shape[0], -1)

        self.model.fit(data, y_train, epochs=self.epochs, callbacks=[
                       early_stopping], batch_size=self.batch_size, shuffle=False, verbose=self.verbose)

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
        return self.model.predict(data, verbose=self.verbose)

    def fit_transform(self, data):
        """
        Fit the given data to the model en return its transformation

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
