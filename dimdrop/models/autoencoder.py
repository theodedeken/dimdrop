from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


class Autoencoder:
    def __init__(self, in_dim, layer_sizes=[2000, 1000, 500, 2], lr=0.01, batch_size=100, patience=3, epochs=1000, regularizer=None, pretrain_method=None, verbose=0):
        self.in_dim = in_dim
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.epochs = epochs
        self.regularizer = regularizer
        self.pretrain_method = pretrain_method
        self.verbose = verbose
        self.__init_network()

    def __init_network(self):
        activation = 'sigmoid' if self.pretrain_method == 'rbm' else 'relu'
        self.layers = [Input(shape=(self.in_dim, ))]
        self.layers += [Dense(size, activation=activation)
                        for size in self.layer_sizes[:-1]]
        self.layers += [Dense(self.layer_sizes[-1])]
        self.layers += [Dense(size, activation=activation)
                        for size in self.layer_sizes[::-1][1:]]
        self.layers += [Dense(self.in_dim)]

        self.model = Sequential([
            self.layers
        ])
        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=self.lr)
        )
        self.encoder = Sequential([
            self.layers[:len(self.layers) // 2 + 1]
        ])
        self.model.compile(
            loss='mse',
            optimizer=Adam(lr=self.lr)
        )

    def fit(self, data):
        early_stopping = EarlyStopping(monitor='loss', patience=self.patience)
        self.model.fit(data, data, epochs=self.epochs, callbacks=[
                       early_stopping], batch_size=self.batch_size, verbose=self.verbose)

    def transform(self, data):
        return self.encoder.predict(data, verbose=self.verbose)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
