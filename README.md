# fifa21-ml-prediction
Machine Learning project to predict FIFA 21 player ratings using Random Forest (CRISP-DM methodology)

# FIFA 21 Player Rating Prediction (Machine Learning)

## Project Overview
This project aims to predict FIFA 21 player overall rating (OVA) using Machine Learning techniques following the CRISP-DM methodology.

The objective is to analyze player attributes and build a model capable of accurately estimating player ratings.

---

## Dataset
The dataset used in this project comes from Kaggle:

https://www.kaggle.com/code/ekrembayar/fifa-21-data-analysis-visualization?scriptVersionId=44960423&cellId=6

Note: The dataset is not included in this repository due to its size.

---

## Methodology (CRISP-DM)

### 1. Business Understanding
Predict player overall rating (OVA) based on attributes.

### 2. Data Understanding
- Explored dataset structure and feature types
- Analyzed distributions and correlations

### 3. Data Preparation
- Removed irrelevant columns (ID, Name, URLs, etc.)
- Handled missing values using SimpleImputer (median strategy)
- Converted "Hits" feature to numeric
- Selected relevant features
- Applied feature scaling using StandardScaler

### 4. Modeling
Three models were trained and compared:
- Linear Regression
- K-Nearest Neighbors (KNN)
- Random Forest Regressor

### 5. Evaluation
Performance metrics used:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

Results:
- Linear Regression: R² ≈ 0.92
- KNN: R² ≈ 0.95
- Random Forest: R² ≈ 0.96 (Best)

### 6. Model Improvement
- KNN tuned (best k = 11)
- Random Forest tuned using GridSearchCV
- Best parameters:
  - n_estimators = 200
  - max_depth = None

Final model performance:
- R² ≈ 0.962
- RMSE ≈ 1.33

---

## Feature Importance
Feature importance analysis using Random Forest showed:

- Reactions is the most important feature
- Base Stats and Potential also contribute significantly
- Other features provide complementary information

A feature reduction experiment showed that removing less important features decreases performance, highlighting the importance of feature interactions.

---

## Final Model
The final selected model is a tuned Random Forest Regressor.

The model and scaler are saved using joblib:
- fifa_model.pkl
- scaler.pkl

---

## How to Run the Project

1. Install dependencies: pip install pandas numpy matplotlib seaborn scikit-learn joblib
2. Run the script: FirstFile.py


---

## Example Prediction

python
sample_player = [80, 400, 75, 70, 78, 85, 1800, 75, 70, 80, 72, 25]
Predicted OVA: ~80 


---


## Model Files
The trained model (.pkl) is not included due to size limitations.

To generate the model, run the training script.
