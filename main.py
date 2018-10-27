from signals_handler import SignalsHandler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import scipy.stats
import scipy.signal
import numpy as np
import os
import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd


import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

skew_tmp = []
kurt_tmp = []
var_tmp = []

def get_and_print_scores(y_true, y_pred) -> None:
    print("ACCURACY: ", accuracy_score(y_true, y_pred))
    print("F1_score: ", f1_score(y_true, y_pred))
    print("RECALL: ", recall_score(y_true, y_pred))
    print("AVG_PRECISION: ", average_precision_score(y_true, y_pred))


def print_stats(acc, recall, f1, prec) -> None:
    print(f"ACCURACY: {acc}")
    print(f"F1_score: {f1}")
    print(f"RECALL: {recall}")
    print(f"AVG_PRECISION: {prec}")


def get_estimators(signal):
    global skew_tmp, kurt_tmp, var_tmp
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


def cross_validate_model(model, data_handler: SignalsHandler, parts_number: int) -> list:
    acc, recall, f1, prec = [], [], [], []
    for learning_data, testing_data, learning_labels, testing_labels in data_handler.cross_validation(parts_number):
        learning_data = np.array(list(map(get_estimators, learning_data)))
        testing_data = np.array(list(map(get_estimators, testing_data)))

        model.fit(learning_data, learning_labels)
        predictions = model.predict(testing_data)

        acc.append(accuracy_score(testing_labels, predictions))
        recall.append(recall_score(testing_labels, predictions))
        f1.append(f1_score(testing_labels, predictions))
        prec.append(average_precision_score(testing_labels, predictions))

    print(acc, recall, f1, prec)
    return [round(np.mean(stat), 3) for stat in (acc, recall, f1, prec)]


def prepare_snr_to_acc_table(model, data_handler: SignalsHandler, foldername: str) -> pd.DataFrame:
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
    PRECISIONs = []
    F1_SCOREs = []
    data_handler.load_no_pulsation(os.path.join(foldername, no_pulsation_file))
    for pulsation_file in pulsation_files:
        print(f'Computing with {pulsation_file}...')
        data_handler.load_pulsation(os.path.join(foldername, pulsation_file))
        scores = cross_validate_model(model, data_handler, parts_number=10)

        SNRs.append(pulsation_file.replace(".csv", '').split("_")[-1])
        AMPs.append(pulsation_file.replace(".csv", '').split("_")[1])
        ACCs.append(scores[0])
        RECALLs.append(scores[1])
        F1_SCOREs.append(scores[2])
        PRECISIONs.append(scores[3])

    table['SNR'] = SNRs
    table['IMP_AMP'] = AMPs
    table['ACC'] = ACCs
    table['RECALL'] = RECALLs
    table['PRECISION'] = PRECISIONs
    table['F1_SCORE'] = F1_SCOREs

    return table.sort_values(by=['IMP_AMP'])


def test_architectures(data_handler: SignalsHandler, architectures: list):
    tables = []
    for architecture in architectures:
        clf = MLPClassifier(solver='lbfgs',
                            activation='logistic',
                            max_iter=10000,
                            hidden_layer_sizes=architecture,
                            # hidden_layer_sizes=(6, 2),
                            # hidden_layer_sizes=(5, 2),
                            learning_rate_init=0.001
                            )
        table = prepare_snr_to_acc_table(clf, data_handler, "inputs\snr_100_100")
        tables.append(table)

    for architecture, table in zip(architectures, tables):
        print(f"\n\nRESULT FOR ARCHITECTURE {architecture}")
        print(table)
    return architectures, tables


def visualise_clusters(model):
    # _, maximum, minimum, variance, _, kurtosis = get_estimators(handler.learning_data[0])

    # skewness = np.linspace(start=-1.15, stop=0.2, num=20)
    # kurtosis = np.linspace(start=-0.90, stop=2.20, num=20)
    # points = [[(x1, x2) for x2 in kurtosis] for x1 in skewness]
    # labels = [[model.predict([[maximum, minimum, variance, x[0], x[1]]]) for x in pts] for pts in points]
    handler = SignalsHandler(preprocessing=get_estimators)
    # handler.load_signals('inputs/symulacje_400_50_srednie')
    handler.load_no_pulsation('inputs/snr_100_100/nopulsation_100.csv')
    handler.load_pulsation('inputs/snr_100_100/pulsation_2_100_-5.733.csv')
    # handler.load_pulsation('inputs/snr_100_100/pulsation_3.5_100_-4.061.csv')
    # handler.load_pulsation('inputs/snr_100_100/pulsation_0.1_100_-6.912.csv')
    handler.initialize_data_sets()
    model.fit(handler.learning_data, handler.learning_labels)
    # var, skew, kurt
    points = [(x[3], x[4]) for x in handler.learning_data]
    labels = model.predict(handler.learning_data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for point, label in zip(points, labels):
        if label == 1:
            symbol = '+'
            color = 'r'
        else:
            symbol = 'o'
            color = 'g'
        ax.scatter(*point, marker=symbol, c=color)
            # print('ploting point')
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Kurtosis')
    # ax.set_zlabel('Kurtosis')
    plt.show()

    # for i in range(len(points)):
    #     for j in range(len(points)):
    #         if labels[i][j] == 1:
    #             symbol = '+'
    #             color = 'r'
    #         else:
    #             symbol = 'o'
    #             color = 'g'
    #         plt.scatter(*points[i][j], marker=symbol, c=color)
    #         # print('ploting point')
    # plt.show()


def plot_spectrogram(signal, title=''):
    # f, t, Sxx = scipy.signal.spectrogram(signal,
    #                                      fs=10000,
    #                                      nfft=2048)
    # plt.pcolormesh(t, f, Sxx)
    plt.specgram(signal, Fs=10e4, NFFT=256)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [s]')
    # plt.title(title)
    plt.show()


if __name__ == '__main__':

    # handler = SignalsHandler(preprocessing=get_estimators)
    # handler = SignalsHandler()
    # handler.load_signals('inputs/symulacje_400_50_duze')
    # handler.initialize_data_sets()
    # plot_spectrogram(handler.pulsation[0])
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


    # architectures = [(5,), (8,), (1,), (3, 1)]
    # architectures = [(5, ), (1, ), (3, 1), (5, 1)]
    # architectures = [(5, 3, 2), (8, 5), (5, ), (2, )]
    # handler = SignalsHandler(preprocessing=get_estimators)
    # arch, tables = test_architectures(handler, architectures)

    # print("SKEWNESS ", min(skew_tmp), max(skew_tmp))
    # print("KURTOSIS ", min(kurt_tmp), max(kurt_tmp))
    # print("VARIANCE ", min(var_tmp), max(var_tmp))
    handler = SignalsHandler(preprocessing=get_estimators)
    handler.load_signals('inputs/symulacje_400_50_duze')
    handler.initialize_data_sets()
    plot_spectrogram(handler.pulsation[0], title='Pulsation')
    # plot_spectrogram(handler.no_pulsation[0], title='No pulsation')

    # clf = SVC()
    # clf.fit(handler.learning_data, handler.learning_labels)
    # print(clf.score(handler.testing_data, handler.testing_labels))
    # print(clf.predict_proba(handler.testing_data))
    # res = cross_validate_model(clf, handler, 5)
    # print(res)

    clf = MLPClassifier(solver='lbfgs',
                        activation='logistic',
                        max_iter=10000,
                        hidden_layer_sizes=(5, ),
                        learning_rate_init=0.001
                        )

    clf.fit(handler.learning_data, handler.learning_labels)
    # visualise_clusters(clf)

    # visualise_clusters(clf)
    # print(clf.coefs_)
    # table = prepare_snr_to_acc_table(clf, handler, "inputs\snr_100_100")
    # print(table)
    # table.to_csv("snr_to_acc.csv")

    # architectures = [(6, 2), (6, 3), (7, 6, 3), (6, 5, 4), (7, 8, 6)]
    # architectures = [(6, 3), (6, 4), (6, 5), (6, )]
    # architectures = [(3, ), (4, ), (6, ), (8, )]
    # architectures = [(3, ), (2, ), (1, )]
    # architectures = [(5, 3), (6, 3), (3, ), (1, )]



    # plt.plot(handler.pulsation[0])
    # plt.show()
