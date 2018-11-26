import numpy as np
import scipy.signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def signal_generator(fs=2**13, T=1.0, mu_imp1=0, f1_low=1500, f1_high=2500, s_noise=0.1, s_add=0.7):
    ff1 = 5
    nx = int(fs * T)
    # print(nx)
    t = np.arange(0, nx) / fs
    rob1 = np.random.randn(nx)
    noise = rob1 * s_noise
    impacts1 = np.zeros((nx))
    impact_ind1 = np.arange(100, nx, round(fs / ff1))
    impacts1[impact_ind1] = mu_imp1
    soi1 = impacts1 + noise
    dumping = 200
    imp = np.exp(-t * dumping)
    df = fs / nx
    win1 = np.hanning(round(f1_high - f1_low) / df)

    fft_mask1 = np.zeros((nx))
    rang = np.arange(round(f1_low / df), round(f1_high / df))
    fft_mask1[rang] = win1
    fft_mask1 = fft_mask1.reshape((1, len(fft_mask1)))
    fft_mask1 = fft_mask1 + np.rot90(fft_mask1, k=2)

    carrier1 = np.real(np.fft.ifft(np.fft.fft(noise) * fft_mask1))
    carrier1 = carrier1 / np.abs(scipy.signal.hilbert(carrier1))

    response1 = carrier1 * imp.reshape((1, len(imp)))
    y = np.convolve(soi1, np.ndarray.flatten(response1))
    y = y[1:nx]

    rob2 = np.random.randn(nx - 1)
    n_add = rob2 * s_add
    y = y + n_add
    return y, t[:-1]


if __name__ == '__main__':
    y, t = signal_generator(fs=2 ** 13, T=0.5, mu_imp1=7,
                            f1_low=1500, f1_high=2500,
                            s_noise=0.1, s_add=0.7)

