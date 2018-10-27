from simulation.analytics.neural_network import NeuralNetwork
from simulation.analytics.preprocessing import PreProcessing


class Monitor:

    def __init__(self):
        self.model = NeuralNetwork.load('models/mlp_classif.model')
        self.preprocessing = PreProcessing()

    def get_status(self, signal):
        pass
