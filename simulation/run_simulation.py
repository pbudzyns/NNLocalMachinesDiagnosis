from simulation.analytics.monitor import Monitor
from simulation.signal_source.signal_source import SignalSource
import time


def get_monitor(model_path):
    monitor = Monitor()
    monitor.load_model(model_path)
    return monitor


def get_signal_source():
    signal_source = SignalSource()
    return signal_source


if __name__ == '__main__':

    monitor = get_monitor("./analytics/models/mlp_classifier.model")
    signal_source = get_signal_source()

    signal_source.start_thread()
    print("Waiting for signal source...")
    time.sleep(3)
    for signal, t in signal_source.iterate(duration=1, loop_time=5):
        status = monitor.get_status(signal)
        proba = monitor.get_damage_proba(signal)
        print(f"Predicted {status} with proba {proba}, imp {signal_source.get_pulsation()}")
        signal_source.increase_pulsation(0.5)
        time.sleep(0.5)

    signal_source.stop_thread()

