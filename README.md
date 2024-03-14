# Deep Learning Regression for Coin Grading

This README outlines a deep learning regression model designed to grade coins based on various image-derived features. The script leverages Python libraries such as Pandas, OpenCV, NumPy, Matplotlib, Scikit-learn, Imbalanced-learn, and TensorFlow to preprocess data, extract features, and construct a regression model capable of predicting the grade of a coin with remarkable accuracy.

## Overview

The model processes coin images to extract and combine features related to the coin's brightness, color (HSV values), and edge characteristics. These features are then used to train a deep learning model that predicts the coin's grade, following the standards set by professional coin grading services (PCGS or NGC).

## Features

- **Data Integration**: Combines obverse and reverse features of coins, including brightness and average HSV values.
- **Preprocessing**: Standardizes features and applies Gaussian noise to improve model robustness.
- **Deep Learning Model**: Utilizes a Sequential model with Dense and Dropout layers for regression tasks.
- **Performance Evaluation**: Employs mean absolute error and mean squared error for model evaluation.

## Files and Data

- **Feature Files**: CSV files containing features extracted from coin images (`Features.csv`, `AverageHsvPerCoinV2.csv`, etc.).
- **Brightness Data**: `obverse_brightness.csv` and `reverse_brightness.csv` for obverse and reverse brightness data.
- **Edge Features**: `dataframe_ML_model.csv` for thresholded feature extraction related to coin edges.

## Model Details

- **Input**: Image-derived features (brightness, HSV values, edge features).
- **Output**: Predicted coin grade.
- **Architecture**: Sequential model with layers configured for regression, including ReLU activation for hidden layers and linear activation for the output layer.

## How to Run

1. **Prepare the Data**: Ensure all CSV files with features and data are in the specified directories.
2. **Execute the Script**: Run the script to preprocess data, train the model, and evaluate its performance.
3. **Evaluate Results**: The script outputs mean absolute error, mean squared error, and accuracy within specified tolerances.

## Performance

- The model demonstrates the capability to predict coin grades with high accuracy.
- An accuracy of 94% is achieved with a tolerance of 2 grades, indicating the model's effectiveness in grading coins closely to professional standards.

## Conclusion

This deep learning regression model provides a powerful tool for automating the coin grading process, leveraging image-derived features to predict grades with high accuracy. By processing and combining multiple aspects of coin imagery, the model mimics the comprehensive analysis performed by human experts, offering scalability and consistency for coin grading tasks.
