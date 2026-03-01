# ⚡ Deep-Learning-Power-Usage-Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

## 🎯 Objective
The primary objective of this project is to accurately forecast electricity consumption to ensure a stable, efficient, and cost-effective energy supply. By leveraging temporal data and meteorological features, this machine learning pipeline is designed to discover power demand patterns that can be actively deployed in industrial settings. This project validates the practical applicability of regression models in real-world energy management.

## ✨ Key Features
To prepare the dataset for prediction, a robust preprocessing and feature engineering pipeline was utilized:
* **Data Preprocessing:** Handles missing weather values (filling rainfall with 0, and wind speed/humidity with column means). Extracts critical temporal features (month, day, time) from raw timestamp data.
* **K-Means Clustering:** Generates a new categorical feature based on spatial/meteorological variables to group similar consumption behaviors.
* **Deep Neural Network (DNN):** Implements a deep, fully connected Multilayer Perceptron for continuous regression.
* **Optimization & Regularization:** Utilizes the AdamW optimizer, a `StepLR` learning rate scheduler, Dropout layers, Batch Normalization, and Early Stopping to prevent overfitting and ensure optimal model performance.

---

## 🧠 Modeling Strategy
A multi-stage approach was utilized, beginning with unsupervised learning for feature engineering, followed by supervised deep learning.

### 1. Feature Engineering (Unsupervised Learning)

Before training the regression model, K-Means clustering was applied to discover hidden groupings and operational regimes within the historical data. 
* **Elbow Method & Silhouette Scores:** Utilized to determine the optimal number of clusters (k=2). The resulting cluster labels were appended as an additional predictive feature to help the model differentiate between distinct consumption patterns.
* **Feature Scaling:** `StandardScaler` was applied to all numerical features to normalize the input space for the neural network, taking care to exclude the categorical cluster label from standard scaling.

### 2. Deep Neural Network Architecture


[Image of a Deep Neural Network architecture]

The core predictive model is defined in the `DNNRegression` PyTorch class. It features an 8-layer architecture utilizing Xavier weight initialization to ensure stable gradient flow.

| Layer | Input Size | Output Size | Regularization / Activation |
| :--- | :--- | :--- | :--- |
| **Hidden 1** | `Input Features` | 128 | BatchNorm, Dropout (0.5), ReLU |
| **Hidden 2** | 128 | 256 | BatchNorm, Dropout (0.5), ReLU |
| **Hidden 3** | 256 | 512 | BatchNorm, Dropout (0.5), ReLU |
| **Hidden 4** | 512 | 1024 | BatchNorm, Dropout (0.5), ReLU |
| **Hidden 5** | 1024 | 1024 | BatchNorm, Dropout (0.5), ReLU |
| **Hidden 6** | 1024 | 512 | BatchNorm, Dropout (0.5), ReLU |
| **Hidden 7** | 512 | 256 | BatchNorm, Dropout (0.5), ReLU |
| **Hidden 8** | 256 | 128 | BatchNorm, Dropout (0.5), ReLU |
| **Output Layer**| 128 | 1 | None (Raw continuous output) |

### 3. Model Training
* **Loss Function:** Mean Squared Error (MSE) was used to measure regression performance.
* **Optimizer:** AdamW with an initial learning rate of 0.0001.
* **Scheduler:** `StepLR` (step size = 2, gamma = 0.1) dynamically decays the learning rate to fine-tune weights as training progresses.
* **Early Stopping:** Monitors validation loss with a patience of 5 epochs. The best-performing model state (`best_model.pth`) is automatically saved.

### 4. Prediction & Inference
The best-saved model is loaded to process the unseen test dataset. The test data runs through the exact same preprocessing, clustering, and scaling pipeline as the training data before being passed into the model to generate the final `submit3.csv` predictions.

---

## 📈 Performance & Results

> 📸 **Insert Training/Validation Loss Curve Here:**
![Loss Curve](docs/images/loss_curve.png)

*(Note: Replace the values below with your final metrics from the console output once training completes!)*
* **Best Validation Loss (MSE):** `XXXX.XXXX`
* **Early Stopping Triggered At Epoch:** `XX`

## ⚙️ How to Run
1. Clone this repository.
2. Ensure `train.csv`, `test.csv`, and `sample_submission.csv` are in the root directory.
3. Run the Jupyter Notebook or Python script to train the model.
4. The script will automatically output `submit3.csv` containing the final power usage predictions.
