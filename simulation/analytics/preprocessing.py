import numpy as np
import scipy.stats


class PreProcessing:

    def __init__(self):
        super(PreProcessing, self).__init__()

    def transform(self, data):
        return self._get_estimators(data)

    def _get_estimators(self, signal):
        # global skew_tmp, kurt_tmp, var_tmp
        label, signal = signal[0], signal[1:]
        maximum = np.max(signal)
        minimum = np.min(signal)
        skewnes = scipy.stats.skew(signal)
        kurtosis = scipy.stats.kurtosis(signal)
        variance = np.var(signal)
        # TODO: add root mean square, crest factor, waveform factor

        return label, maximum, minimum, variance, skewnes, kurtosis
