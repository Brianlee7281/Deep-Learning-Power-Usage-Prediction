# Deep-Learning-Power-Usage-Prediction

Objective 

The primary objective of this project is to accurately forecast electricity consumption to ensure a stable, efficient, and cost-effective energy supply. By leveraging temporal data and meteorological features, this machine learning pipeline is designed to discover power demand algorithms that can be actively deployed in industrial settings. This project validates the practical applicability of regression models in real-world energy management.

Key Feature

To prepare the dataset for the prediction, couple features were utilized 
Data Preprocessing: Handles missing values and extracts temporal features (month, day, time) from timestamp data
K-Means Clustering: Utilizes KMeans clustering to generate a new categorical feature based on other variables
DNN(Deep Neural Network) and Optimization: Implemented Multilayer Perceptron for regression and prediction. Used AdamW optimizer, StepLr scheduler, and early stopping to prevent overfitting and improve model performance

Modeling Strategy

A multi-stage approach was utilized, beginning with unsupervised learning for feature engineering, followed by DNN and optimization.

1. Feature Engineering (Unsupervised Learning)

K-Means Clustering: Before training regression models, K-Means clustering was applied to discover hidden groupings and operational regimes within the historical data.
Elbow Method: Utilized the Elbow Method to determine the optimal number of clusters, appending the resulting cluster labels as an additional predictive feature to help the models differentiate between distinct consumption patterns.
Data Splitting: The data was split into training and validation sets. 
Feature Scaling: Standard Scaler was used to scale numerical features

2. Deep Neural Network

DNN Model was defined in 'DNNRegression' class with 8 layers, BatchNorm1d, ReLU, and Dropout for regularization.

3. Model Training

MSE(Mean Squared Error) was used as the loss function, and AdamW was used as the optimizer
'StepLR' was used as a learning rate scheduler
early stopping based on validation loss was implemented to save the best performing model

4. Prediction

The best saved model was loaded on the test dataset and made predictions on hte preprocessed test dataset

Performance

