📌 Project Overview

This project builds and evaluates machine learning models to predict residential housing prices using structured real-estate data. The focus is on understanding key price drivers and comparing multiple regression models within a complete machine learning pipeline.

The goal is not leaderboard optimization, but demonstrating a clear, reproducible workflow for feature engineering, model tuning, and evaluation.

🎯 Objectives

Analyze factors that influence housing prices

Handle mixed numerical and categorical features

Compare baseline and ensemble regression models

Select a best-performing model using validation metrics

🧠 Workflow

Data cleaning and exploratory data analysis

Categorical encoding using OneHotEncoder

Pipeline construction for preprocessing and modeling

Hyperparameter tuning with GridSearchCV

Model comparison and selection

Final inference using the selected model

🤖 Models Used

Linear Regression (baseline)

Random Forest Regressor

XGBoost Regressor (best-performing model)

📊 Results & Insights

Tree-based ensemble models outperformed linear regression, highlighting the importance of capturing non-linear relationships and feature interactions. After tuning, XGBoost achieved the strongest predictive performance.

📂 Dataset

Source: House Prices: Advanced Regression Techniques (Kaggle)

Contains residential property attributes such as location, size, condition, and amenities

📝 Notes

Kaggle submission file generation is environment-specific and not the primary focus of this project

Emphasis is placed on model development, tuning, and evaluation, rather than competition mechanics

This notebook is intended to demonstrate applied machine learning practices on a real-world regression problem
