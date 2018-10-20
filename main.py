from signals_handler import SignalsHandler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import scipy.stats
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


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
    label, signal = signal[0], signal[1:]
    maximum = np.max(signal)
    minimum = np.min(signal)
    skewnes = scipy.stats.skew(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    variance = np.var(signal)
    # TODO: add root mean square, crest factor, waveform factor
    return label, maximum, minimum, variance, skewnes, kurtosis


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

    return [round(np.mean(stat), 3) for stat in (acc, recall, f1, prec)]


def prepare_snr_to_acc_table(model, data_handler, foldername):
    # files = glob.glob(os.path.join(foldername, "*.csv"))
    table = pd.DataFrame()
    pulsation_files = []
    no_pulsation_file = ''

    for filename in os.listdir(foldername):
        print(filename)
        if filename.startswith("nopulsation"):
            no_pulsation_file = filename
        else:
            pulsation_files.append(filename)

    SNRs = []
    AMPs = []
    ACCs = []
    RECALLs = []
    F1_SCOREs = []
    data_handler.load_no_pulsation(os.path.join(foldername, no_pulsation_file))
    for pulsation_file in pulsation_files:
        print(f'Computing with {pulsation_file}...')
        data_handler.load_pulsation(os.path.join(foldername, pulsation_file))
        scores = cross_validate_model(model, data_handler, parts_number=7)

        SNRs.append(pulsation_file.replace(".csv", '').split("_")[-1])
        AMPs.append(pulsation_file.replace(".csv", '').split("_")[1])
        ACCs.append(scores[0])
        RECALLs.append(scores[1])
        F1_SCOREs.append(scores[2])

    table['SNR'] = SNRs
    table['IMP_AMP'] = AMPs
    table['ACC'] = ACCs
    table['RECALL'] = RECALLs
    table['F1_SCORE'] = F1_SCOREs

    return table.sort_values(by=['IMP_AMP'])


if __name__ == '__main__':
    # handler = SignalsHandler(preprocessing=get_estimators)
    # handler.load_signals('inputs/symulacje_400_50_srednie')
    # handler.initialize_data_sets()
    # print(handler.learning_data[0])
    # print(handler.learning_labels[0])
    # print(handler.testing_labels[0])
    # TODO: odszumianie
    # TODO: ekstrakcja sygnalu informacyjnego
    # handler.learning_data = np.array(list(map(get_estimators, handler.learning_data)))
    # handler.testing_data = np.array(list(map(get_estimators, handler.testing_data)))

    # architectures = [(3,), (4, 3), (7, 5, 3)]
    # for arch in architectures:
    #     clf = MLPClassifier(solver='lbfgs',
    #                     activation='logistic',
    #                     max_iter=10000,
    #                     hidden_layer_sizes=arch,
    #                     learning_rate_init=0.001
    #                     )
    #     print(f"TESTING HIDDEN LAYERS: {arch}")
    #     print_stats(*cross_validate_model(clf, handler, 10))

    handler = SignalsHandler(preprocessing=get_estimators)
    clf = MLPClassifier(solver='lbfgs',
                        activation='logistic',
                        max_iter=10000,
                        # hidden_layer_sizes=(7, 6, 3),
                        hidden_layer_sizes=(6, 3),
                        learning_rate_init=0.001
                        )
    table = prepare_snr_to_acc_table(clf, handler, "inputs\snr_100_100")
    print(table)
    table.to_csv("snr_to_acc.csv")

    # plt.plot(handler.pulsation[0])
    # plt.show()
