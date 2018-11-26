from simulation.signal_source.signal_generator import signal_generator
import numpy as np
import csv
import os

if __name__ == '__main__':
    n = 200
    output_folder = "outputs"
    filename = f"nopulsation_{n}.csv"

    with open(os.path.join(output_folder, filename), "w", newline='') as f:
        csvwriter = csv.writer(f, delimiter=",")
        for i in range(n):
            y, t = signal_generator(T=0.5, mu_imp1=0)
            csvwriter.writerow(np.append(y, 0))

