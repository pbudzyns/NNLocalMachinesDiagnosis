from sklearn.neural_network import MLPClassifier
from simulation.analytics.train_model.signals_handler import SignalsHandler
from simulation.analytics.preprocessing import PreProcessing
import pickle


def get_preprocessing():
    prec = PreProcessing()
    return prec.transform


def prepare_signal_handler(path):
    handler = SignalsHandler(preprocessing=get_preprocessing())
    handler.load_signals(path)
    handler.initialize_data_sets()
    return handler


def train_and_save_model(model, data_handler, path):
    model.fit(data_handler.learning_data, data_handler.learning_labels)
    pickle.dump(model, open(path, "wb"))


if __name__ == '__main__':
    # handler = prepare_signal_handler(".\inputs\symulacje_400_50_srednie")
    handler = SignalsHandler(preprocessing=get_preprocessing())
    handler.load_pulsation(r".\inputs\snr_100_100\pulsation_1.5_100_-6.274.csv")
    handler.load_no_pulsation(r".\inputs\snr_100_100\nopulsation_100.csv")
    handler.initialize_data_sets()

    clf = MLPClassifier(solver='lbfgs',
                        activation='logistic',
                        max_iter=10000,
                        hidden_layer_sizes=(5,),
                        learning_rate_init=0.001
                        )



    train_and_save_model(clf, handler, "..\models\mlp_classifier_one_big_5.model")
    print(f"Done. Model successfully saved.\n Mean accuracy: {clf.score(handler.testing_data, handler.testing_labels)}")
