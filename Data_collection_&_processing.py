import pyedflib
import os
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import welch
import matplotlib.pyplot as plt
data_dir = "D:/Sleep_EDF_data"

psg_files = [f for f in os.listdir(data_dir) if '-PSG.edf' in f]

def load_edf_signals(file_path):
    with pyedflib.EdfReader(file_path) as f:
        signal_labels = f.getSignalLabels()
        sampling_rates = [f.getSampleFrequency(i) for i in range(len(signal_labels))]
        signals = {}
        for label in ['Resp oro-nasal', 'EEG Fpz-Cz', 'Event marker']:
            if label in signal_labels:
                index = signal_labels.index(label)
                signals[label] = {
                    'data': f.readSignal(index),
                    'sampling_rate': sampling_rates[index]
                }
    return signals

def remove_artifacts(data, lower_bound=-2000, upper_bound=2000):
    return np.clip(data, lower_bound, upper_bound)

def calculate_baseline(resp_flow, percentile=50):
    # Check if resp_flow contains NaN or Inf values
    if np.isnan(resp_flow).any():
        resp_flow = np.nan_to_num(resp_flow, nan=0.0)  # Replace NaNs with 0

    if np.isinf(resp_flow).any():
        resp_flow = np.nan_to_num(resp_flow, posinf=0.0, neginf=0.0)  # Replace Inf with 0

    # Ensure the data is not empty
    if len(resp_flow) == 0:
        raise ValueError("resp_flow is empty")

    # Calculate and return the percentile as baseline
    return np.percentile(np.abs(resp_flow), percentile)

def smooth_signal(data, window_size):
    return pd.Series(data).rolling(window=window_size, center=True).mean()

def detect_events(resp_flow, sampling_rate):
    baseline = calculate_baseline(resp_flow, 50)
    apnea_threshold = -0.1 * baseline  # 90% reduction for apnea
    hypopnea_threshold = -0.7 * baseline  # 30% reduction for hypopnea

    print(f"Baseline: {baseline}")
    print(f"Apnea Threshold: {apnea_threshold}")
    print(f"Hypopnea Threshold: {hypopnea_threshold}")
    min_duration = int(sampling_rate * 9.5)  # 9.5 seconds
    max_duration = int(sampling_rate * 120)  # 2 minutes

    events = []
    event_start = None
    event_type = None

    for i in range(len(resp_flow)):
        if resp_flow[i] <= apnea_threshold:
            if event_start is None:
                event_start = i
                event_type = 'apnea'
        elif resp_flow[i] <= hypopnea_threshold:
            if event_start is None:
                event_start = i
                event_type = 'hypopnea'
        else:
            if event_start is not None:
                duration = i - event_start
                if min_duration <= duration <= max_duration:
                    events.append({
                        'start': event_start,
                        'end': i,
                        'type': event_type
                    })
                event_start = None
                event_type = None

    return events

# Estimate sleep time using delta band (0.5 - 4 Hz) power
def estimate_sleep_time(eeg_data, sampling_rate, epoch_duration=30, delta_band=(0.5, 4.0)):
    epoch_samples = int(epoch_duration * sampling_rate)  # Samples per epoch
    epochs = len(eeg_data) // epoch_samples  # Number of epochs

    sleep_epochs = 0
    delta_powers = []

    for i in range(epochs):
        # Extract the current epoch
        epoch = eeg_data[i * epoch_samples:(i + 1) * epoch_samples]

        # Calculate power spectral density using Welch's method
        freqs, psd = welch(epoch, fs=sampling_rate, nperseg=epoch_samples)

        # Find the power in the delta band (0.5 - 4 Hz for NREM sleep detection)
        delta_power = np.trapz(psd[(freqs >= delta_band[0]) & (freqs <= delta_band[1])])
        delta_powers.append(delta_power)

    # Calculate a threshold based on the median delta power across epochs
    delta_power_threshold = np.median(delta_powers) * 1.5  

    # Count the number of epochs where delta power exceeds the threshold
    for delta_power in delta_powers:
        if delta_power > delta_power_threshold:
            sleep_epochs += 1

    # Calculate total sleep time in hours
    total_sleep_time = (sleep_epochs * epoch_duration) / 3600  # Convert to hours
    return total_sleep_time

def calculate_ahi(events, total_sleep_time_hours):
    total_events = len(events)
    return total_events / total_sleep_time_hours
columns = ["Baseline", "Apnea Threshold", "Hypopnea Threshold", 
           "Total events detected", "Estimated sleep time", "AHI", "Apnea Status"]

df = pd.DataFrame(columns=columns)
# Main code to run the analysis
sleep_disorders = {}
for file in psg_files:
    edf_file = "D:/Sleep_EDF_data/"+file 
    signals = load_edf_signals(edf_file)

    resp_flow = remove_artifacts(signals['Resp oro-nasal']['data'])
    eeg_data = signals['EEG Fpz-Cz']['data']

    sampling_rate = signals['Resp oro-nasal']['sampling_rate']

    window_size = int(sampling_rate * 5)  # 5-second window
    resp_flow_smooth = smooth_signal(resp_flow, window_size)

    events = detect_events(resp_flow_smooth, sampling_rate)

    # Improved sleep time estimation based on delta power in EEG data
    total_sleep_time_hours = estimate_sleep_time(eeg_data, signals['EEG Fpz-Cz']['sampling_rate'])

    ahi = calculate_ahi(events, total_sleep_time_hours)

    # Print results
    print(f"Total events detected: {len(events)}")
    print(f"Estimated sleep time: {total_sleep_time_hours:.2f} hours")
    print(f"AHI: {ahi:.2f}")

    # Classify apnea severity
    if ahi < 5:
        apnea_status = 0
    elif 5 <= ahi < 15:
        apnea_status = 1
    elif 15 <= ahi < 30:
        apnea_status = 2
    else:
        apnea_status = 3

    if apnea_status in sleep_disorders:
        sleep_disorders[apnea_status] += 1
    else:
        sleep_disorders[apnea_status] = 1
    row = {
        "Baseline": calculate_baseline(resp_flow_smooth),
        "Apnea Threshold": -0.1 * calculate_baseline(resp_flow_smooth),
        "Hypopnea Threshold": -0.7 * calculate_baseline(resp_flow_smooth),
        "Total events detected": len(events),
        "Estimated sleep time": total_sleep_time_hours,
        "AHI": ahi,
        "Apnea Status": apnea_status
    }
    
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    print(f"Apnea Status: {apnea_status}")
    print()
print(sleep_disorders)

df.to_csv('sleep_apnea_data_numeric.csv', index=False)

