# Sleep Apnea Prediction Tool  
## Link- https://sleep-disorder-ml-project.streamlit.app/

This project is a Machine Learning-based tool designed to predict sleep apnea and classify its severity using Polysomnography (PSG) data in EDF format. The model processes signals related to respiration and EEG to estimate sleep time, detect apnea/hypopnea events, and predict the Apnea-Hypopnea Index (AHI) and sleep apnea severity.

### Features

AHI Prediction: Predicts the Apnea-Hypopnea Index based on PSG data.
Sleep Apnea Classification: Classifies sleep apnea severity into No Sleep Apnea, Mild Sleep Apnea, Moderate Sleep Apnea, or Severe Sleep Apnea.
Event Detection: Detects respiratory events like apnea and hypopnea from the PSG signals.
Sleep Time Estimation: Estimates total sleep time using EEG delta band power.

### Models Used

Random Forest Regressor for AHI prediction (rf_regressor.pkl).
Support Vector Machine (SVM) for classifying sleep apnea severity (best_svm.pkl).

### Data Inputs

The tool accepts Polysomnography (PSG) files in EDF format and processes the following signals:
- Respiratory (oro-nasal airflow): For detecting apnea and hypopnea events.
- EEG (Fpz-Cz): For estimating sleep time based on delta band power.

### Data Source
https://physionet.org/content/sleep-edfx/1.0.0/

### Model performance

These are the metrics we have achieved on the SVM model:

Cross-validation accuracy scores: [0.94736842 0.78947368 0.94594595 0.94594595 0.91891892]
Mean accuracy: 0.9095305832147937
