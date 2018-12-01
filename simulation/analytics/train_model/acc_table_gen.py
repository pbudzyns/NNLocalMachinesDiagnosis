from simulation.analytics.train_model.signals_handler import SignalsHandler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score
import scipy.stats
import scipy.signal
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.base


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
        model = sklearn.base.clone(model, safe=True)
        learning_data = np.array(list(map(get_estimators, learning_data)))
        testing_data = np.array(list(map(get_estimators, testing_data)))
        print(f"learning model...")
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
    no_pulsation_file = ""

    for filename in os.listdir(foldername):
        print(filename)
        if "one_big" in filename: continue
        if filename.startswith("nopulsation"):
            if "100" in filename:
                no_pulsation_file = filename
            else:
                continue
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
        print(f"Pulsation file {pulsation_file} loaded")
        scores = cross_validate_model(model, data_handler, parts_number=10)
        print(f"cross valdation done")
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
        table = prepare_snr_to_acc_table(clf, data_handler, "snr_100_100")
        tables.append(table)

    for architecture, table in zip(architectures, tables):
        print(f"\n\nRESULT FOR ARCHITECTURE {architecture}")
        print(table)
    return architectures, tables



if __name__ == '__main__':


    # architectures = [(5,), (2, ), (8, 5,), (5,3,2)]
    architectures = [(1000,500,200), (500,200)]
    for architecture in architectures:
        clf = MLPClassifier(solver='lbfgs',
                            activation='logistic',
                            max_iter=10000,
                            hidden_layer_sizes=architecture,
                            learning_rate_init=0.001
                            )


        table = prepare_snr_to_acc_table(clf,
                                         SignalsHandler(preprocessing=None),
                                         "./inputs/snr_100_100")
        table.to_csv("../../interface/outputs/snr_to_acc{}.csv".format(architecture))

