from keras import backend as K
from keras.metrics import binary_crossentropy


class VAELoss:
    """Custom loss function for a variational autoencoder.

    Parameters
    ----------
    in_dim : int
        The dimension of the input
    z_log_var : tensor
        The variance tensor
    z_mean : tensor
        The mean tensor
    """
    __name__ = 'vae_loss'

    def __init__(self, in_dim, z_log_var, z_mean):

        self.in_dim = in_dim
        self.z_log_var = z_log_var
        self.z_mean = z_mean

    def __call__(self, x, x_decoded_mean):
        xent_loss = self.in_dim * \
            binary_crossentropy(x, x_decoded_mean)
        if self.z_log_var:
            kl_loss = - 0.5 * \
                K.sum(1 + self.z_log_var - K.square(self.z_mean) -
                      K.exp(self.z_log_var), axis=-1)
        else:
            kl_loss = - 0.5 * \
                K.sum(1 + 1 - K.square(self.z_mean) - K.exp(1.0), axis=-1)
        return K.mean(xent_loss + kl_loss)
