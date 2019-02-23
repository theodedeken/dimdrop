from keras.callbacks import Callback


class KMeansRegularizer(Callback):
    __name__ = 'kmeans_regularizer'

    def __init__(self, input_data, k):
        self.input_data = input_data
        self.k = k
