import numpy as np
from sklearn.model_selection import train_test_split
import os


class SignalsHandler:
    NO_PULSATION_NAME = "no_pulsation.csv"
    PULSATION_NAME = "pulsation.csv"

    def __init__(self, preprocessing=None):
        self.preprocessing = preprocessing
        self.no_pulsation = None
        self.pulsation = None
        self.data = None
        self.learning_data = None
        self.learning_labels = None
        self.testing_data = None
        self.testing_labels = None

    def load_signals(self, folder_name):
        self.no_pulsation = np.genfromtxt(os.path.join(folder_name, self.NO_PULSATION_NAME), delimiter=",")
        self.pulsation = np.genfromtxt(os.path.join(folder_name, self.PULSATION_NAME), delimiter=",")

    def load_no_pulsation(self, filename):
        self.no_pulsation = np.genfromtxt(filename, delimiter=",")

    def load_pulsation(self, filename):
        self.pulsation = np.genfromtxt(filename, delimiter=",")

    def cut_signal(self, start, end):
        self.no_pulsation = self.no_pulsation[:, start:end]
        self.pulsation = self.pulsation[:, start:end]

    def initialize_data_sets(self):
        self._add_labels()
        self._compose_into_one()
        self._shuffle_data()
        self._preprocess()
        self._split_into_train_test()

    def cross_validation(self, parts_number):
        self._add_labels()
        self._compose_into_one()
        self._preprocess()
        self._shuffle_data()
        parts = self._split_into_n_parts(parts_number)

        for i in range(parts_number):
            testing_data = parts[i][:, 1:]
            testing_labels = parts[i][:, 0]
            learning = np.concatenate(parts[:i] + parts[i + 1:])
            learning_data = learning[:, 1:]
            learning_labels = learning[:, 0]
            yield learning_data, testing_data, learning_labels, testing_labels

    def _add_labels(self):
        self.no_pulsation[:, 0] = 0
        self.pulsation[:, 0] = 1

    def _compose_into_one(self):
        self.data = np.append(self.no_pulsation, self.pulsation, axis=0)

    def _preprocess(self):
        if self.preprocessing:
            self.data = np.array(list(map(self.preprocessing, self.data)))

    def _shuffle_data(self):
        np.random.shuffle(self.data)

    def _split_into_n_parts(self, n):
        return np.array_split(self.data, n, axis=0)

    def _split_into_train_test(self):
        self.learning_data, self.testing_data, self.learning_labels, self.testing_labels = \
            train_test_split(self.data[:, 1:], self.data[:, 0], test_size=0.2)
