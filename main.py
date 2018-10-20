from signals_handler import SignalsHandler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt


def print_scores(y_true, y_pred):
    print("ACCURACY: ", accuracy_score(y_true, y_pred))
    print("F1_score: ", f1_score(y_true, y_pred))
    print("RECALL: ", recall_score(y_true, y_pred))
    print("AVG_PRECISION: ", average_precision_score(y_true, y_pred))


def get_estimators(signal):
    maximum = np.max(signal)
    minimum = np.min(signal)
    skewnes = scipy.stats.skew(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    variance = np.var(signal)
    # TODO: add root mean square, crest factor, waveform factor
    return maximum, minimum, variance, skewnes, kurtosis


def extract_from_vector(label, data_vec):
    features = get_estimators(data_vec)
    return label, features


if __name__ == '__main__':
    handler = SignalsHandler()
    handler.load_signals('inputs/symulacje_400_50_srednie')
    handler.initialize_data_sets()
    # TODO: odszumianie
    # TODO: ekstrakcja sygnalu informacyjnego
    handler.learning_data = np.array(list(map(get_estimators, handler.learning_data)))
    handler.testing_data = np.array(list(map(get_estimators, handler.testing_data)))
    # print(training_data)

    # handler.cut_signal(0, 8)

    # print(handler.learning_labels[:2])
    # print(handler.learning_data[:2, :])
    # print(handler.testing_labels.shape)
    # print(handler.testing_data.shape)

    scaler = StandardScaler()
    scaler.fit(handler.learning_data)
    handler.learning_data = scaler.transform(handler.learning_data)
    handler.testing_data = scaler.transform(handler.testing_data)

    clf = MLPClassifier(solver='lbfgs',
                        activation='logistic',
                        max_iter=10000,
                        hidden_layer_sizes=(7, 8, 6),
                        learning_rate_init=0.001
                        )
    #
    #
    clf.fit(handler.learning_data, handler.learning_labels)
    predictions = clf.predict(handler.testing_data)
    print_scores(handler.testing_labels, predictions)

    plt.plot(handler.pulsation[0])
    plt.show()
#