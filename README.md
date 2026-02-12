# House Price Prediction – Data Science & Machine Learning Project

## Overview
This project explores the factors that influence residential house prices and applies
machine learning techniques to predict sale prices using the Kaggle
**House Prices: Advanced Regression Techniques** dataset.

The primary focus is on **exploratory data analysis (EDA)** and **feature understanding**,
with machine learning used as a supporting tool to evaluate predictive performance.

---

## Project Objective
- Analyze how property characteristics such as size, quality, and location impact sale price
- Use visual exploration to uncover patterns, distributions, and relationships in the data
- Prepare a clean, structured dataset suitable for regression modeling
- Compare multiple models and evaluate their effectiveness

---

## Dataset
Source: Kaggle – *House Prices: Advanced Regression Techniques*

The dataset includes detailed residential property attributes such as:
- Structural features (square footage, rooms, lot size)
- Quality and condition ratings
- Neighborhood and location variables
- Final sale price (target variable)

---

## Exploratory Data Analysis (EDA)
EDA was conducted to better interpret price behavior and feature relationships, including:
- Distribution analysis of sale prices and numerical predictors
- Visualization of sale price against key features (e.g., living area, quality, lot size)
- Identification of missing values and outliers
- Analysis of categorical variables and their influence on pricing

Insights from EDA directly informed preprocessing and feature engineering decisions.

---

## Data Preparation & Feature Engineering
- Cleaned and standardized numerical variables
- Engineered additional features to better capture property characteristics
- Applied categorical encoding using `OneHotEncoder`
- Prepared model-ready datasets for regression analysis

---

## Modeling & Evaluation
The following models were trained and evaluated:
- Linear Regression
- Random Forest Regression
- XGBoost Regression

Model performance was assessed using cross-validation, and hyperparameter tuning was
performed with `GridSearchCV`. Among the evaluated models, **XGBoost demonstrated the
strongest predictive performance** and was selected as the final model.

---

## Results & Key Observations
- House prices are primarily driven by quality, usable living space, and location
- Feature engineering and thoughtful preprocessing are the most impactful stages of the ML pipeline
- Advanced models are effective only after strong data foundations are established
- Predictive accuracy is as much a data problem as it is a modeling problem

This project reinforced a core data science principle:
Better data understanding consistently outperforms more complex algorithms.

---

## Tools & Technologies
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Plotly
- Jupyter Notebook

---

## Notes
This repository emphasizes the complete data science workflow, including data exploration,
feature engineering, and model evaluation.
---

## Project Files
- `House Price Prediction.ipynb` – Main notebook containing EDA, preprocessing, modeling, and evaluation
- `data_description.txt` – Dataset feature descriptions provided by Kaggle
