from keras.callbacks import Callback
import numpy as np
from keras import backend as K


class KMeansRegularizer(Callback):
    __name__ = 'kmeans_regularizer'

    def __init__(self, cluster_centers, weight=0.5):
        self.cluster_centers = cluster_centers
        self.weight = K.variable(weight)
        self.cluster_assignments = []

    def on_epoch_end(self, epoch, logs=None):
        # update cluster centers
        grouped = [[] for _ in range(len(self.cluster_centers))]
        for el in self.cluster_assignments:
            grouped[el[0]].append(el[1])
        for i, group in enumerate(grouped):
            self.cluster_centers[i] = np.sum(np.array(group)) / len(group)

    def __call__(self, activations):
        dists = K.zeros(shape=(len(activations),))
        for i, activation in enumerate(activations):
            dist_2 = np.sum((self.cluster_centers - activation)**2, axis=1)
            min_dist = np.argmin(dist_2)
            dists[i] = dist_2[min_dist]
            self.cluster_assignments.append((min_dist, activation))

        squared = K.square(dists)
        return self.weight * K.sum(squared)
