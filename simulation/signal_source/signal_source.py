from simulation.signal_source.signal_generator import signal_generator
import numpy as np
import time
import threading
from threading import Thread

class SignalSource:
    def __init__(self, imp_amp, interval):
        self._imp_amp = imp_amp
        self._interval = interval
        self._max_time = 2
        self._single_signal_len = self._get_single_signal_len()
        self._total_signal_len = int((self._max_time/self._interval)*self._single_signal_len)
        self._signal = None
        self._active = threading.Event()

    def get_signal(self, time):
        n = int((time / self._max_time) * len(self._signal))
        return self._signal[-n:]

    def start_thread(self):
        self._signal, _ = self._get_signal_for_interval()
        thread = Thread(target=self._generating_signal)
        thread.start()

    def stop_thread(self):
        self._active.set()

    def _generating_signal(self):
        while self._active:
            signal, _ = self._get_signal_for_interval()
            self._signal = np.append(self._signal, signal)
            if len(self._signal) > self._total_signal_len:
                self._signal = self._signal[-self._total_signal_len:]
            time.sleep(self._interval)

    def _get_signal_for_interval(self):
        return signal_generator(T=self._interval, mu_imp1=self._imp_amp)

    def _get_single_signal_len(self):
        y, t = signal_generator(T=self._interval)
        return len(y)

if __name__ == '__main__':
    ss = SignalSource(imp_amp=4, interval=0.5)
    import matplotlib.pyplot as plt
    ss.start_thread()
    time.sleep(3)
    for i in range(3):
        plt.clf()
        plt.plot(ss.get_signal(1))
        plt.show()
        time.sleep(1)
    ss.stop_thread()
    # plt.plot(ss.get_signal(1))
    # plt.show()