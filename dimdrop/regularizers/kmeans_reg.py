from keras.callbacks import Callback
import numpy as np
from keras import backend as K

import random


class KMeansRegularizer(Callback):
    __name__ = 'kmeans_regularizer'

    def __init__(self, cluster_centers, batch_size=100, weight=0.5):
        self.cluster_centers = cluster_centers
        self.batch_size = batch_size
        self.weight = K.variable(weight)
        self.cluster_assignments = []

    def init_fit(self, encoder, input_data):
        self.encoder = encoder
        self.input_data = input_data

    def on_epoch_end(self, epoch, logs=None):
        # update cluster centers
        encoding = self.encoder.predict(self.input_data)
        new_centers = np.zeros(self.cluster_centers.shape)
        counters = np.zeros(self.cluster_centers.shape[0])
        for point in encoding:
            dist_2 = np.sum((self.cluster_centers - point)**2, axis=1)
            min_dist = np.argmin(dist_2)
            new_centers[min_dist] += point
            counters[min_dist] += 1

        self.cluster_centers = np.array(
            [new_centers[i] / counters[i] for i in range(len(counters))])
        print()
        print(self.cluster_centers)
        self.__fix_centers()

        print(self.cluster_centers)

        #grouped = [[] for _ in range(len(self.cluster_centers))]
        # for el in self.cluster_assignments:
        #    grouped[el[0]].append(K.get_value(el[1]))
        #new_centers = []

        # for i, group in enumerate(grouped):
        #    new_centers.append(np.sum(np.array(group)) / len(group))
        #self.cluster_centers = new_centers
        #self.cluster_assignments = []

    def __call__(self, activations):
        #dists = K.zeros(shape=(self.batch_size,))
        dists = K.map_fn(self.__cluster_dist, activations)

        squared = K.square(dists)
        return self.weight * K.sum(squared)

    def __fix_centers(self):
        false_centers = [i for i in range(
            len(self.cluster_centers)) if np.isnan(self.cluster_centers[i][0])]
        true_centers = [i for i in range(
            len(self.cluster_centers)) if not np.isnan(self.cluster_centers[i][0])]
        for index in false_centers:
            sample = random.sample(true_centers, 3)
            new_center = np.zeros(2)
            for el in sample:
                new_center += self.cluster_centers[el]

            self.cluster_centers[index] = new_center / 3

    def __calc_center(self, nodes):
        print(nodes)
        return np.sum(nodes) / len(nodes)

    def __cluster_dist(self, activation):
        dist_2 = K.sum((self.cluster_centers - activation)**2, axis=1)
        min_dist = np.argmin(dist_2)

        #self.cluster_assignments.append((min_dist, activation))
        return dist_2[min_dist]
