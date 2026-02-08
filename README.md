House Price Prediction Using Machine Learning
Project Overview

This project focuses on building and evaluating machine learning models to predict residential housing prices using structured real-estate data. The goal is to explore how different modeling approaches perform on a complex regression problem involving categorical and numerical features, and to identify the most effective model through systematic tuning and evaluation.

Rather than optimizing for leaderboard placement, the emphasis is on the end-to-end machine learning workflow: data preparation, feature engineering, model selection, and performance comparison.

Key Objectives

Understand the factors that influence housing prices

Apply feature engineering techniques to mixed data types

Compare multiple regression models using consistent evaluation methods

Select a best-performing model based on predictive performance and generalization

Methodology

Data cleaning and exploratory data analysis

Categorical feature encoding using OneHotEncoder

Pipeline construction for reproducible preprocessing and modeling

Hyperparameter tuning using GridSearchCV

Model comparison based on validation performance

Final inference using the selected model

Models Implemented

Linear Regression (baseline model)

Random Forest Regressor

XGBoost Regressor (best-performing model)

Results & Insights

Tree-based ensemble models significantly outperformed linear regression, highlighting the importance of capturing non-linear relationships and feature interactions in housing price prediction. XGBoost provided the strongest overall performance after hyperparameter tuning.

Dataset

Source: Kaggle — House Prices: Advanced Regression Techniques

Data includes residential property characteristics such as location, size, condition, and amenities.

Notes

Kaggle submission file generation is environment-specific and not the primary focus of this repository.

The project prioritizes model development, tuning, and evaluation over competition mechanics.

This repository is intended to demonstrate applied machine learning practices on a real-world regression problem.
