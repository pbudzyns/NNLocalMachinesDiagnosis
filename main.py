from signals_handler import SignalsHandler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning)


def print_scores(y_true, y_pred):
    print("ACCURACY: ", accuracy_score(y_true, y_pred))
    print("F1_score: ", f1_score(y_true, y_pred))
    print("RECALL: ", recall_score(y_true, y_pred))
    print("AVG_PRECISION: ", average_precision_score(y_true, y_pred))


def print_stats(acc, recall, f1, prec):
    print(f"ACCURACY: {acc}")
    print(f"F1_score: {f1}")
    print(f"RECALL: {recall}")
    print(f"AVG_PRECISION: {prec}")


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

def cross_validate_model(model, data_handler, parts_number):

    acc, recall, f1, prec = [], [], [], []
    for learning_data, testing_data, learning_labels, testing_labels in data_handler.cross_validation(parts_number):
        learning_data = np.array(list(map(get_estimators, learning_data)))
        testing_data = np.array(list(map(get_estimators, testing_data)))

        model.fit(learning_data, learning_labels)
        predictions = clf.predict(testing_data)

        acc.append(accuracy_score(testing_labels, predictions))
        recall.append(recall_score(testing_labels, predictions))
        f1.append(f1_score(testing_labels, predictions))
        prec.append(average_precision_score(testing_labels, predictions))

    return (np.mean(stat) for stat in (acc, recall, f1, prec))



if __name__ == '__main__':
    handler = SignalsHandler()
    handler.load_signals('inputs/symulacje_400_50_male')
    handler.initialize_data_sets()
    # TODO: odszumianie
    # TODO: ekstrakcja sygnalu informacyjnego
    # handler.learning_data = np.array(list(map(get_estimators, handler.learning_data)))
    # handler.testing_data = np.array(list(map(get_estimators, handler.testing_data)))

    architectures = [(3,), (4, 3), (7, 5, 3)]
    for arch in architectures:
        clf = MLPClassifier(solver='lbfgs',
                        activation='logistic',
                        max_iter=10000,
                        hidden_layer_sizes=arch,
                        learning_rate_init=0.001
                        )
        print(f"TESTING HIDDEN LAYERS: {arch}")
        print_stats(*cross_validate_model(clf, handler, 10))

    # plt.plot(handler.pulsation[0])
    # plt.show()
