import streamlit as st
import pandas as pd
import joblib
import pyedflib
import numpy as np
from scipy.signal import welch
import os

# Load EDF signals
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

# Remove artifacts from the data
def remove_artifacts(data, lower_bound=-2000, upper_bound=2000):
    return np.clip(data, lower_bound, upper_bound)

# Calculate baseline
def calculate_baseline(resp_flow, percentile=50):
    if np.isnan(resp_flow).any():
        resp_flow = np.nan_to_num(resp_flow, nan=0.0)

    if np.isinf(resp_flow).any():
        resp_flow = np.nan_to_num(resp_flow, posinf=0.0, neginf=0.0)

    if len(resp_flow) == 0:
        raise ValueError("resp_flow is empty")

    return np.percentile(np.abs(resp_flow), percentile)

# Smooth the signal
def smooth_signal(data, window_size):
    return pd.Series(data).rolling(window=window_size, center=True).mean()

# Detect apnea and hypopnea events
def detect_events(resp_flow, sampling_rate):
    baseline = calculate_baseline(resp_flow, 50)
    apnea_threshold = -0.1 * baseline
    hypopnea_threshold = -0.7 * baseline
    min_duration = int(sampling_rate * 9.5)
    max_duration = int(sampling_rate * 120)

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
                    events.append({'start': event_start, 'end': i, 'type': event_type})
                event_start = None
                event_type = None

    return events

# Estimate sleep time using delta band (0.5 - 4 Hz) power
def estimate_sleep_time(eeg_data, sampling_rate, epoch_duration=30, delta_band=(0.5, 4.0)):
    epoch_samples = int(epoch_duration * sampling_rate)
    epochs = len(eeg_data) // epoch_samples
    sleep_epochs = 0
    delta_powers = []

    for i in range(epochs):
        epoch = eeg_data[i * epoch_samples:(i + 1) * epoch_samples]
        freqs, psd = welch(epoch, fs=sampling_rate, nperseg=epoch_samples)
        delta_power = np.trapz(psd[(freqs >= delta_band[0]) & (freqs <= delta_band[1])])
        delta_powers.append(delta_power)

    delta_power_threshold = np.median(delta_powers) * 1.5

    for delta_power in delta_powers:
        if delta_power > delta_power_threshold:
            sleep_epochs += 1

    total_sleep_time = (sleep_epochs * epoch_duration) / 3600
    return total_sleep_time

# Calculate the Apnea-Hypopnea Index (AHI)
def calculate_ahi(events, total_sleep_time_hours):
    total_events = len(events)
    if total_sleep_time_hours == 0:
        return 0
    return total_events / total_sleep_time_hours

# Process the PSG file and extract features
def process_psg_file(file_path):
    signals = load_edf_signals(file_path)
    resp_flow = remove_artifacts(signals['Resp oro-nasal']['data'])
    eeg_data = signals['EEG Fpz-Cz']['data']
    sampling_rate = signals['Resp oro-nasal']['sampling_rate']

    window_size = int(sampling_rate * 5)
    resp_flow_smooth = smooth_signal(resp_flow, window_size)
    events = detect_events(resp_flow_smooth, sampling_rate)

    total_sleep_time_hours = estimate_sleep_time(eeg_data, signals['EEG Fpz-Cz']['sampling_rate'])
    ahi = calculate_ahi(events, total_sleep_time_hours)

    data_to_predict = {
        "Baseline": [calculate_baseline(resp_flow)],
        "Apnea Threshold": [-0.1 * calculate_baseline(resp_flow)],
        "Hypopnea Threshold": [-0.7 * calculate_baseline(resp_flow)],
        "Total events detected": [len(events)],
        "Estimated sleep time": [total_sleep_time_hours],
    }

    return pd.DataFrame(data_to_predict)

# Make predictions using both models
def make_predictions(df):
    ahi_model = joblib.load('rf_regressor.pkl')
    predicted_ahi = ahi_model.predict(df)

    severity_model = joblib.load('best_svm.pkl')
    predicted_sleep_apnea_label = severity_model.predict(df)
    apnea_status = ["No Sleep Apnea", "Mild Sleep Apnea", "Moderate Sleep Apnea", "Severe Sleep Apnea"][predicted_sleep_apnea_label[0]]

    return predicted_ahi[0], predicted_sleep_apnea_label[0], apnea_status

# Streamlit app
def main():
    st.title("Sleep Apnea Prediction Tool")

    uploaded_file = st.file_uploader("Upload PSG file (EDF format)", type=["edf"])
    
    if uploaded_file is not None:
        try:
            # Save the uploaded file to a temporary location
            temp_file_path = os.path.join("temp.edf")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the PSG file
            input_df = process_psg_file(temp_file_path)

            # Make predictions
            predicted_ahi, predicted_sleep_apnea_label, apnea_status = make_predictions(input_df)

            # Display results
            st.subheader("Prediction Results")
            st.write(f"Predicted AHI: {predicted_ahi}")
            st.write(f"Predicted Sleep Apnea Label: {apnea_status}")

            # Remove the temp file after processing
            os.remove(temp_file_path)

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
