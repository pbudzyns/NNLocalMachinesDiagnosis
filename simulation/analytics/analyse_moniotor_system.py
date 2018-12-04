from simulation.signal_source.signal_generator import signal_generator
from simulation.analytics.monitor import Monitor
import numpy as np
import pandas as pd

mu_imps = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.50, 1.75, 2.0, 2.25, 2.50]
monitor = Monitor()
monitor.load_model("../analytics/models/mlp_classifier_one_big_5.model")

snrs, hits = [], []
probas = []
n = 300
for mu_imp in mu_imps:
    hits_number = 0
    snr_sum = 0
    proba = []
    for i in range(n):
        y, t, soi1, response1, n_add = signal_generator(mu_imp1=mu_imp)
        hits_number += int(monitor.get_status(y))
        proba.append(monitor.get_damage_proba(y)[1])
        snr_sum += np.sqrt(np.var(soi1, 0)/np.var(n_add, 0))
    snrs.append(snr_sum/n)
    hits.append((hits_number/n)*100)
    probas.append(proba)
    print("Done for {}".format(mu_imp))

df = pd.DataFrame()
df["IMP"] = mu_imps
df["SNR"] = snrs
df["Hits"] = hits
df["Proba"] = probas
df.to_csv("outputs/monitor_system_stats300_with_probas2.csv")
