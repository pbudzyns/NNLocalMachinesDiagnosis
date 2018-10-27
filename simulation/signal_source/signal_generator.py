import numpy as np

def signal_generator(fs, T, mu_imp1, f1_low, f1_high, s_noise, s_add):
    ff1 = 5
    nx = fs*T
    t = np.arange(1, nx)/fs
    rob1 = np.random.randn(nx)
    noise = rob1*s_noise
    impacts1 = np.zeros((nx, 1))
    impact_ind1 = np.arange(100, nx, round(fs/ff1))
    impacts1[impact_ind1] = mu_imp1
    soi1 = impacts1+noise
    dumping = 200
    imp = np.exp(-t*dumping)
    print(impact_ind1)
    pass

if __name__ == '__main__':
    signal_generator(2**13, 2, 0.5, 1500, 2500, 0.1, 0.7)