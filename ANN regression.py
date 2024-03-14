#deep learnign regression.

import pandas as pd

import os
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Configuration
features_file_path = "reverse_nobg/Features.csv"
features_obverse_file_path = "obverse_nobg/Features.csv"
average_hsv_file_path = "HSV/AverageHsvPerCoinV2.csv"
average_hsv_reverse_file_path = "HSV/AverageHsvPerCoinV2Reverse.csv"

# Paths to the text files and column names
obverse_brightness = pd.read_csv("Brightness Data/obverse_brightness.csv")
reverse_brightness = pd.read_csv("Brightness Data/reverse_brightness.csv")

# Merge datasets
df_merged_brightness = pd.merge(obverse_brightness, reverse_brightness, on='Filename',suffixes=('_obverse', '_reverse'))

# Function Definitions
def extract_grade(filename):
    """Extracts the grade from the filename using regex."""
    match = re.search(r'GRADE(\d+)', filename)
    return float(match.group(1)) if match else None

def load_and_preprocess(features_path, obverse=False):
    """Loads and preprocesses feature data."""
    df = pd.read_csv(features_path)
    df.rename(columns={'Unnamed: 0': 'Filename'}, inplace=True)
    df['Filename'] = df['Filename'].astype(str)
    if obverse:
        # Append '_obverse' to column names, excluding 'Filename'
        df.columns = ['Filename' if col == 'Filename' else col + '_obverse' for col in df.columns]
    return df

# Load and preprocess datasets
df_coins = load_and_preprocess(features_file_path)
df_coins_obv = load_and_preprocess(features_obverse_file_path, obverse=True)

# Merge datasets
df_coins = pd.merge(df_coins, df_coins_obv, on='Filename')
df_coins['GRADE'] = df_coins['Filename'].apply(extract_grade)

# Load HSV average data and merge
df_obv_avg = pd.read_csv(average_hsv_file_path)
df_rev_avg = pd.read_csv(average_hsv_reverse_file_path)
df_coins = pd.merge(df_coins, df_obv_avg, on='Filename')
df_coins = pd.merge(df_coins, df_rev_avg, on='Filename')
df_coins = pd.merge(df_coins,df_merged_brightness, on='Filename')
df_coins['pcgs'] = df_coins['Filename'].apply(lambda x: 1 if 'pcgs' or 'PCGS' in x else 0)

#Load the thresholded feature extraction edge data
df_features = pd.read_csv('dataframe_ML_model.csv')
df_features['Filename'] = df_coins['Filename'].astype(str)
df_features['Filename'] = df_coins['Filename'].str.replace('features_edges_', '', regex=False)
df_coins = pd.merge(df_coins,df_features, on='Filename',suffixes=('','_x'))
df_coins = df_coins.drop('GRADE_x',axis=1)
MAX_GRADE = 68
MIN_GRADE = 50
df_coins = df_coins[(df_coins['GRADE'] < MAX_GRADE) & (df_coins['GRADE'] > MIN_GRADE)]

#if Filename column contains '+' then increase GRADE by 0.5
df_coins.loc[df_coins['Filename'].str.contains('\+', regex=True), 'GRADE'] += 0.5

def check_and_update(row):
    if '+' in row['Filename']:
            row['GRADE'] += 0.9
    return row

df_coins.apply(check_and_update,axis=1)

X = df_coins.drop(['GRADE', 'Filename'], axis=1)
y = df_coins['GRADE']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Assuming X_scaled is your standardized feature set
np.random.seed(101)  # For reproducibility

# Adding Gaussian noise to your features (ensure this makes sense for your dataset)
noise = np.random.normal(0, 0.01, X_scaled.shape)
X_noisy = X_scaled + noise

X_train_dl, X_test_dl, y_train_dl, y_test_dl = train_test_split(X_noisy, y, test_size=0.2, random_state=101)



# Define the deep learning model architecture
model = Sequential()
model.add(Dense(128, input_dim=X_train_dl.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))  # Output layer



# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error', 'mean_squared_error'])


# Train the model 282
model.fit(X_train_dl, y_train_dl, epochs=2500, batch_size=32, validation_split=0.2)

# Evaluate the model on the test data
#loss, accuracy = model.evaluate(X_test_dl, y_test_dl)
#print(f'Test accuracy: {accuracy*100:.2f}%')
loss, mae, mse = model.evaluate(X_test_dl, y_test_dl)
print(f'Test Mean Absolute Error: {mae:.2f}')
print(f'Test Mean Squared Error: {mse:.2f}')


# Prediction (optional step to see how the model performs on new data)
y_pred_dl = model.predict(X_test_dl)
# Convert predictions from one-hot encoded back to labels
mae = mean_absolute_error(y_test_dl, y_pred_dl)  # Ensure y_test_real is the original, unscaled test labels
mse = mean_squared_error(y_test_dl, y_pred_dl)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
"""
print(classification_report(encoder.inverse_transform(np.argmax(y_test_dl, axis=1)), y_pred_grades, zero_division=0))
print(confusion_matrix(encoder.inverse_transform(np.argmax(y_test_dl, axis=1)), y_pred_grades))
"""
def accuracy_with_tolerance(y_true, y_pred,tolerance=1):
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if abs(true - pred) <= tolerance:
            correct += 1
    return correct / len(y_true)

val_true = y_test_dl.tolist()
val_pred = y_pred_dl.tolist()

print(f'The accuracy is {accuracy_with_tolerance(y_test_dl, y_pred_dl)}')
#make a dataframe with true and predicted values
df = pd.DataFrame({'true': val_true, 'predicted': val_pred})
df.to_csv('results.csv', index=False)

#svc_accuracy_with_tolerance = accuracy_with_tolerance(encoder.inverse_transform(np.argmax(y_test_dl, axis=1)), y_pred_grades)
#print(f'Deep Learning Accuracy with Tolerance: {svc_accuracy_with_tolerance*100:.2f}%')
svc_accuracy_with_tolerance = accuracy_with_tolerance(y_test_dl, y_pred_dl)
svc_accuracy_with_tolerance2 = accuracy_with_tolerance(y_test_dl, y_pred_dl,2)
svc_accuracy_with_tolerance3 = accuracy_with_tolerance(y_test_dl, y_pred_dl,3)
print(f'Deep Learning Accuracy with Tolerance of 1 : {round(svc_accuracy_with_tolerance*100)}%')
print(f'Deep Learning Accuracy with Tolerance of 2 : {round(svc_accuracy_with_tolerance2*100)}%')
print(f'Deep Learning Accuracy with Tolerance of 3 : {round(svc_accuracy_with_tolerance3*100)}%')