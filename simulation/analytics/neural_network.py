from sklearn.neural_network import MLPClassifier
import pickle


class NeuralNetwork(MLPClassifier):

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))
