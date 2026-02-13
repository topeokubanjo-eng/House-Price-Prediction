# House Price Prediction – End-to-End Machine Learning Pipeline

## Overview
This project builds a complete regression pipeline to predict residential property sale prices using the Kaggle  
**House Prices: Advanced Regression Techniques** dataset.

The emphasis of this project is not just model accuracy, but disciplined data exploration, feature engineering, and reproducible preprocessing. The objective was to understand which features truly drive housing prices and evaluate how different modeling strategies impact predictive performance.

---

## Project Objective
- Perform deep exploratory data analysis (EDA) to identify high-impact predictors
- Transform raw housing attributes into informative engineered features
- Build a reproducible preprocessing pipeline to prevent data leakage
- Compare linear and ensemble-based regression models
- Optimize model performance using cross-validation and hyperparameter tuning
- Evaluate whether increased model complexity produces meaningful gains

---

## Dataset
**Source:** Kaggle – *House Prices: Advanced Regression Techniques*

The dataset contains 79 explanatory variables describing residential homes in Ames, Iowa, including:

- Structural features (square footage, number of rooms, lot size)
- Construction quality and overall condition ratings
- Neighborhood and zoning classifications
- Garage, basement, and remodeling attributes
- Final sale price (target variable)

The target variable (`SalePrice`) was log-transformed to reduce right-skewness and stabilize variance for regression modeling.

---

## Exploratory Data Analysis (EDA)
EDA was conducted to guide modeling decisions and uncover structural relationships in the data.

Key analysis included:
- Distribution analysis of `SalePrice` and numeric predictors
- Correlation analysis between features and target
- Visualization of price vs. living area, overall quality, and property age
- Categorical price comparisons across neighborhoods and zoning classes
- Missing value analysis and outlier detection

Key insight: Overall quality and total usable living space were significantly stronger predictors than many isolated structural counts.

EDA findings directly informed preprocessing and feature engineering strategy.

---

## Data Preparation & Feature Engineering
A structured preprocessing pipeline was built using `ColumnTransformer` and `Pipeline` to ensure reproducibility and eliminate data leakage.

### Numerical Features
- Mean imputation
- Standard scaling

### Categorical Features
- Constant imputation ("missing")
- One-hot encoding via `OneHotEncoder`

### Engineered Features
- `TotalSF` (combined living space across levels)
- `TotalBath` (weighted full and half bathrooms)
- `PropertyAge`
- Binary indicators (HasGarage, HasRemodeled, Has2ndFloor)

Feature engineering improved model generalization and reduced noise from fragmented structural variables.

---

## Modeling & Evaluation
The following regression models were trained and evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

Model evaluation was performed using cross-validation with **Root Mean Squared Error (RMSE)** as the primary metric.

Hyperparameter tuning was conducted using `GridSearchCV` to systematically optimize model performance.

Among the evaluated models, **XGBoost achieved the strongest predictive performance** and was selected as the final model.

---

## Results & Key Observations
- Overall quality, total square footage, and neighborhood-related variables were the dominant predictors of sale price.
- Structured preprocessing and feature engineering had a larger impact on performance than increasing model complexity alone.
- Ensemble tree-based methods outperformed linear regression after proper feature preparation.
- Predictive modeling performance is highly dependent on data representation quality.

This project reinforces a central machine learning principle:
Model performance scales with data understanding.

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Plotly
- Jupyter Notebook

---

## Project Files
- `House Price Prediction.ipynb` – EDA, preprocessing pipeline, modeling, and evaluation
- `data_description.txt` – Official dataset documentation
