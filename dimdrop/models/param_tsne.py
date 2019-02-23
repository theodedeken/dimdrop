from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from sklearn.neural_network import BernoulliRBM
import numpy as np

from ..losses import TSNELoss
from ..util.tsne import compute_joint_probabilities


class ParametricTSNE:
    def __init__(self, in_dim, layer_sizes=[500, 500, 2000, 2], lr=0.01, batch_size=5000, pretrain=False, perplexity=30, tol=1e-5, verbose=0, patience=3, epochs=1000):
        self.in_dim = in_dim
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
            elif i == len(self.layer_sizes) - 1:
                self.layers.append(Dense(num))
            else:
                self.layers.append(Dense(num, activation=activation))

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
                       early_stopping], batch_size=self.batch_size, shuffle=False)
