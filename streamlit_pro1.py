import pyedflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("sleep_apnea_data_numeric.csv")

# Assume the last column is the label (adjust based on your dataset)
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Labels

# Use SMOTE to generate synthetic data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine the resampled features and labels back into a dataframe
df_resampled = pd.DataFrame(X_resampled, columns=df.columns[:-1])
df_resampled['label'] = y_resampled

# Sample 400 rows from the resampled dataset
df_resampled = df_resampled.sample(n=400, random_state=42,replace=True)

# Save the new dataset
df_resampled.to_csv("sleep_apnea_data_resampled.csv", index=False)

print(f"New dataset shape: {df_resampled.shape}")


# Assuming 'df_resampled' is your DataFrame and 'AHI' is the column name for AHI values
df_filtered = df_resampled[df_resampled['AHI'] <= 40]

# Optional: Save the filtered DataFrame to a new CSV file
df_filtered.to_csv("sleep_apnea_data_filtered.csv", index=False)

print(f"Filtered dataset shape: {df_filtered.shape}")


df_resampled=df_filtered

import matplotlib.pyplot as plt

# Load the DataFrame from the CSV

# Create a scatter plot for the AHI column
plt.figure(figsize=(10, 6))
plt.scatter(df_resampled.index, df_resampled['AHI'], color='b', alpha=0.7)
plt.title('Scatter Plot of AHI (Apnea-Hypopnea Index)')
plt.xlabel('Index')
plt.ylabel('AHI')
plt.grid(True)

# Show the plot
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming 'data' is the DataFrame containing the dataset
X = df_resampled[['Baseline', 'Apnea Threshold', 'Hypopnea Threshold', 'Total events detected', 'Estimated sleep time']]
y = df_resampled['label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)


X_train.shape, X_test.shape, y_train.shape, y_test.shape

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load your dataset
# df = pd.read_csv("your_dataset.csv")  # Uncomment and load your dataset

# Assuming X and y are defined as features and labels
# X = df.drop(columns=['label'])  # Adjust according to your dataset
# y = df['label']

models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "Support Vector Machine": SVC(),
}

results = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5)  # Replace X and y with your feature and label variables
    results[model_name] = {
        "Mean Accuracy": scores.mean(),
        "Standard Deviation": scores.std(),
    }

# Convert results to DataFrame for easier sorting
results_df = pd.DataFrame(results).T
print("Model Performance:\n", results_df)

# Sort by Mean Accuracy
sorted_results = results_df.sort_values(by='Mean Accuracy', ascending=False)
print("\nSorted Model Performance:\n", sorted_results)


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Initialize SVM model
svm = SVC(random_state=42)

# Grid Search for tuning hyperparameters
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],  # Different kernel types
    'C': [0.1, 1, 10, 100],               # Regularization parameter
    'gamma': ['scale', 'auto']            # Kernel coefficient
}

# Cross-validation with Grid Search
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_svm = grid_search.best_estimator_

print("Best hyperparameters:", grid_search.best_params_)


from sklearn.model_selection import cross_val_score, StratifiedKFold

# Assuming you have already split your data into X_train, y_train, X_test, y_test

# Initialize classifier (e.g., Random Forest)
model = SVC()

# Use StratifiedKFold for cross-validation to maintain class distribution in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation on the training data
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

# Print the cross-validation results
print("Cross-validation accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
print("Standard deviation:", scores.std())


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset
df = pd.read_csv("sleep_apnea_data_filtered.csv")

# Select features and target variable (assuming 'AHI' is your target variable)
X = df.drop(columns=['AHI','label'])  # Drop the target variable
y = df['AHI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_regressor.fit(X_train, y_train)

# Make predictions
y_pred = rf_regressor.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


import joblib

# Assuming 'best_model' is the model you want to save
joblib.dump(best_svm, 'best_svm.pkl')

import joblib

# Assuming 'best_model' is the model you want to save
joblib.dump(rf_regressor, 'rf_regressor.pkl')

import joblib
import pandas as pd

# Load the trained model
model_path = 'best_svm.pkl'  # Update with your model path
model = joblib.load(model_path)
ahi_predictor = joblib.load("rf_regressor.pkl")

# Prepare the data for prediction
# Assuming you have the calculated values from your previous code
data_to_predict = {
    "Baseline":[93],  # Replace with your actual function call if needed
    "Apnea Threshold": [-9.3],
    "Hypopnea Threshold": [-65],
    "Total events detected": [167],  # Total events detected
    "Estimated sleep time": [9],  # Total sleep time in hours
}

# Create DataFrame
input_df = pd.DataFrame(data_to_predict)

# Make predictions
predicted_ahi = ahi_predictor.predict(input_df)  # Predict AHI
predicted_sleep_apnea_label = model.predict(input_df)  # Replace with the specific prediction call for labels if different

# Display predictions
print(f"Predicted AHI: {predicted_ahi}")
if predicted_sleep_apnea_label == 0:
        apnea_status = "No Sleep Apnea"
elif predicted_sleep_apnea_label == 1:
    apnea_status = "Mild Sleep Apnea"
elif predicted_sleep_apnea_label == 2:
    apnea_status = "Moderate Sleep Apnea"
else:
    apnea_status = "Severe Sleep Apnea"

print(f"Predicted Sleep Apnea Label: {predicted_sleep_apnea_label}")
print(f"Apnea Status: {apnea_status}")




