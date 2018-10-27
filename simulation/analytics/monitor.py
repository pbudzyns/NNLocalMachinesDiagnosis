from simulation.analytics.neural_network import NeuralNetwork
from simulation.analytics.preprocessing import PreProcessing


class Monitor:

    def __init__(self):
        self._model = None
        self._preprocessing = PreProcessing()

    def get_status(self, signal):
        data = self._prepare_input(signal)
        return self._model.predict([data])

    def get_damage_proba(self, signal):
        data = self._prepare_input(signal)
        return self._model.predict_proba([data])

    def _prepare_input(self, signal):
        return self._preprocessing.transform(signal)[1:]

    def load_model(self, path):
        self._model = NeuralNetwork.load(path)