"""
Code adapted from https://github.com/wang-research/VASC
"""
# FIXME if possible reimplement annealing for gumbel approximation in more
# idiomatic keras code

# -*- coding: utf-8 -*-
import h5py
from keras import metrics
from keras.optimizers import RMSprop, Adagrad, Adam
import numpy as np
from keras.utils.layer_utils import print_summary
from keras import regularizers
from keras.models import Model
import keras.backend as K
from keras.layers.merge import concatenate, multiply
from keras.layers import (
    Input,
    Dense,
    Activation,
    Lambda,
    RepeatVector,
    Reshape,
    Layer,
    Dropout,
    BatchNormalization,
    Permute
)
from keras.callbacks import EarlyStopping
from ..losses import VAELoss
from ..util import Transform


class VASC:
    """
    VASC: variational autoencoder for scRNA-seq datasets

    Parameters
    ----------
    in_dim : int
        The input dimension
    out_dim : int
        The output dimension
    layer_sizes : array
        sizes of each layer in the neural network, default is the structure
        proposed in the original paper, namely: `[512, 128, 32]`
    epochs: int, optional
        Maximum number of epochs, default `5000`
    patience: int, optional
        The amount of epochs without improvement before the network stops
        training, default `3`
    batch_size: int, optional
        The batch size for stochastic optimization, default `100`
    lr : float, optional
        The learning rate of the network, default `0.01`
    var: boolean, optional
        Whether to estimate the variance parameters, default `False`
    log: boolean, optional
        Whether log-transformation should be performed, default `False`
    scale: boolean, optional
        Whether scaling (making values within [0,1]) should be performed,
        default `True`
    decay : bool, optional
        Whether to decay the learning rate during training, default `True`.
    verbose : int, optional
        Controls the verbosity of the model, default `0`

    Attributes
    ----------
    ae : keras model
        Partial model used for generating the embbeddings
    vae : keras model
        The variational autoencoder

    References
    ----------
    * Dongfang Wang and Jin Gu. Vasc: dimension reduction and visualization of
      single cell rna sequencing data by deep variational autoencoder.
      *bioRxiv*, 2017.
    """

    def __init__(
            self,
            in_dim,
            out_dim,
            layer_sizes=[512, 128, 32],
            epochs=1000,
            patience=3,
            batch_size=100,
            lr=0.01,
            var=False,
            log=False,
            scale=True,
            decay=True,
            verbose=0):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.var = var
        self.data_transform = Transform(scale, log)
        self.verbose = verbose
        self.decay = decay
        self.__init_network()

    def __init_network(self):
        in_dim = self.in_dim
        expr_in = Input(shape=(self.in_dim,))

        # The first part of model to recover the expr.
        h0 = Dropout(0.5)(expr_in)
        # Encoder layers
        last_layer = h0
        for i, layer_size in enumerate(self.layer_sizes):
            if i == 0:
                last_layer = Dense(
                    units=layer_size,
                    name='encoder_{}'.format(i + 1),
                    kernel_regularizer=regularizers.l1(0.01)
                )(last_layer)
            else:
                last_layer = Dense(
                    units=layer_size,
                    name='encoder_{}'.format(i + 1)
                )(last_layer)
            last_layer = Activation('relu')(last_layer)

        z_mean = Dense(units=self.out_dim, name='z_mean')(last_layer)
        z_log_var = None
        if self.var:
            z_log_var = Dense(units=2, name='z_log_var')(last_layer)
            z_log_var = Activation('softplus')(z_log_var)

        # sampling new samples
            z = Lambda(sampling, output_shape=(
                self.out_dim,))([z_mean, z_log_var])
        else:
            z = Lambda(sampling, output_shape=(self.out_dim,))([z_mean])

        # Decoder layers
        last_layer = z
        for i, layer_size in enumerate(self.layer_sizes[::-1]):
            last_layer = Dense(
                units=layer_size, name='decoder_{}'.format(i + 1))(last_layer)
            last_layer = Activation('relu')(last_layer)

        expr_x = Dense(units=self.in_dim, activation='sigmoid')(
            last_layer)

        expr_x_drop = Lambda(lambda x: -x ** 2)(expr_x)
        expr_x_drop_p = Lambda(lambda x: K.exp(x))(expr_x_drop)
        expr_x_nondrop_p = Lambda(lambda x: 1-x)(expr_x_drop_p)
        expr_x_nondrop_log = Lambda(lambda x: K.log(x+1e-20))(expr_x_nondrop_p)
        expr_x_drop_log = Lambda(lambda x: K.log(x+1e-20))(expr_x_drop_p)
        expr_x_drop_log = Reshape(target_shape=(
            self.in_dim, 1))(expr_x_drop_log)
        expr_x_nondrop_log = Reshape(target_shape=(
            self.in_dim, 1))(expr_x_nondrop_log)
        logits = concatenate(
            [expr_x_drop_log, expr_x_nondrop_log], axis=-1)

        temp_in = Input(shape=(self.in_dim,))
        temp_ = RepeatVector(2)(temp_in)

        temp_ = Permute((2, 1))(temp_)
        samples = Lambda(gumbel_softmax, output_shape=(
            self.in_dim, 2,))([logits, temp_])
        samples = Lambda(lambda x: x[:, :, 1])(samples)
        samples = Reshape(target_shape=(self.in_dim,))(samples)

        out = multiply([expr_x, samples])

        vae = Model(inputs=[expr_in, temp_in], outputs=out)

        loss_func = VAELoss(in_dim, z_log_var, z_mean)

        opt = RMSprop(lr=self.lr, decay=self.lr /
                      self.epochs if self.decay else 0.0)
        vae.compile(optimizer=opt, loss=loss_func)

        ae = Model(inputs=[expr_in, temp_in], outputs=[
                   z_mean, z, samples, out])

        self.vae = vae
        self.ae = ae

        if self.verbose:
            self.vae.summary()

    def fit(self, data):
        """
        Fit the given data to the model.

        Parameters
        ----------
        data : array
            Array of training samples where each sample is of size `in_dim`
        """
        data = self.data_transform(data)

        tau_in = np.ones(data.shape, dtype='float32')

        early_stopping = EarlyStopping(monitor='loss', patience=self.patience)

        self.vae.fit(
            [data, tau_in],
            data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=self.verbose,
            callbacks=[early_stopping]
        )

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

    def transform(self, data):
        """
        Transform the given data

        Parameters
        ----------
        data : array
            Array of samples to be transformed, where each sample is of size
            `in_dim`

        Returns
        -------
        array
            Transformed samples, where each sample is of size `out_dim`
        """
        data = self.data_transform(data)
        return self.ae.predict([data, np.ones(data.shape, dtype='float32')])[0]


def sampling(args):
    epsilon_std = 1.0

    if len(args) == 2:
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean),
                                  mean=0.,
                                  stddev=epsilon_std)

        return z_mean + K.exp(z_log_var / 2) * epsilon
    else:
        z_mean = args[0]
        epsilon = K.random_normal(shape=K.shape(z_mean),
                                  mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(1.0 / 2) * epsilon


def sampling_gumbel(shape, eps=1e-8):
    u = K.random_uniform(shape)
    return -K.log(-K.log(u+eps)+eps)


def compute_softmax(logits, temp):
    z = logits + sampling_gumbel(K.shape(logits))
    return K.softmax(z / temp)


def gumbel_softmax(args):
    logits, temp = args
    return compute_softmax(logits, temp)
