# 🏠 House Price Prediction

### End-to-End ML Pipeline · Exploratory Data Analysis · Feature Engineering · Multi-Model Benchmarking · Prediction Evaluation

This project builds a complete machine learning pipeline to predict residential sale prices using the Ames Housing dataset (1,460 homes × 81 features). The work is split across two notebooks:

| Notebook | Purpose |
|---|---|
| `House_Price_Prediction_Model.ipynb` | Full preprocessing pipeline, hyperparameter tuning, grid search, and model evaluation |
| `House_Price_Visualizations.ipynb` | Standalone visual story — EDA, feature analysis, model comparison, and prediction deep-dive |

---

## Tools & Libraries

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-red)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-purple)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Static%20Charts-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-teal)

---

## The Story in Four Acts

---

### Act 1 — Understanding the Data

![EDA Dashboard](images/chart1_eda.png)

The first challenge in any price prediction problem is understanding the shape of the target variable. Raw sale prices in this dataset range from \$34,900 to \$755,000 with a mean of \$181K — but the distribution is heavily right-skewed, meaning a handful of luxury homes distort the signal for the majority.

**The fix is log-transformation.** Applying `log(SalePrice)` produces a near-normal distribution, satisfying linear model assumptions and giving every model a cleaner target to learn from. This decision — visible in the first two panels — is foundational to every result that follows.

Beyond the target, three variables stand out immediately as price drivers:

- **Overall Quality** has a near-perfect monotonic relationship with price — homes rated 10/10 sell for a median \$432K vs \$50K for the lowest tier. This single feature carries enormous signal.
- **Total Square Footage** (engineered as Basement + 1st + 2nd floor) shows a strong linear relationship with price (r = 0.78), with higher-quality homes (yellow dots) clustering at the top of the range.
- **Property Age** shows a clear negative correlation (r = −0.52) — newer homes consistently sell for more, though the relationship is noisier than size or quality.
- **Location matters significantly.** NridgHt commands a median \$315K vs \$146K for NPkVill — a 116% gap purely from postcode.

---

### Act 2 — Feature Engineering

![Feature Engineering](images/chart4_features.png)

Raw features miss composite signals. This act shows how engineering new variables unlocks stronger predictors.

The **correlation heatmap** confirms that OverallQual (r = 0.79) and the engineered TotalSF (r = 0.78) are the two strongest predictors of price — both outperforming any raw individual floor area column. PropertyAge, by contrast, is the strongest *negative* predictor (r = −0.52), reinforcing that age is a liability.

The **feature correlation bar chart** tells the full ranked story: the top positive drivers are OverallQual, TotalSF, GrLivArea, GarageCars, and TotalBath — all size- and quality-related. The negative end is led by PropertyAge, EnclosedPorch, and KitchenAbvGr, suggesting that older homes with enclosed spaces and more above-grade rooms (without corresponding quality) are penalised by the market.

The **TotalSF vs GrLivArea scatter** makes the case for feature engineering directly: the engineered composite (cyan, r = 0.78) sits systematically above the raw living area variable (blue, r = 0.71). Combining all floor areas into a single feature captures total usable space more completely than any one floor alone.

The **boolean feature price premium chart** is particularly clean for storytelling:
- Homes **with a garage** sell for **1.68×** more than those without
- Homes **with a 2nd floor** sell for **1.44×** more
- Homes **with a remodel** sell for **1.18×** more
- Interestingly, homes **with a pool** sell for only **0.91×** — suggesting pools are associated with older or rural properties in this dataset, not luxury homes
- **Fenced homes** (0.82×) follow a similar pattern

These multipliers translate directly into model features that add interpretable, business-relevant signal.

---

### Act 3 — Model Comparison

![Model Comparison Dashboard](images/chart2_model_comparison.png)

Four models were trained and benchmarked across three preprocessing pipelines (Baseline, PCA, and Feature Engineering + PCA):

| Model | Test RMSE | R² | CV RMSE |
|---|---|---|---|
| **Linear Regression** ⭐ | **0.1329** | **0.9054** | 0.1610 |
| Ridge Regression | 0.1360 | 0.9008 | 0.1457 |
| Random Forest | 0.1466 | 0.8848 | 0.1436 |
| Gradient Boosting | 0.1388 | 0.8968 | **0.1311** |

**Linear Regression achieves the lowest test RMSE (0.1329) and highest R² (0.9054)** — meaning it explains over 90% of the variance in log(SalePrice). This is a meaningful result: with proper preprocessing (imputation, scaling, one-hot encoding), a simple linear model outperforms both ensemble methods on this dataset's test split.

The **CV stability chart** adds nuance. While Linear Regression wins on test RMSE, its cross-validation RMSE (0.1610 ± 0.012) is the highest among all models — suggesting slightly more variance across folds. Gradient Boosting shows the best CV performance (0.1311 ± 0.014), hinting that it may generalise more reliably on unseen data at scale. Ridge Regression offers a middle ground with reasonable test performance and tighter CV spread.

The **predicted vs actual scatter** (bottom centre) shows all four models produce well-aligned predictions along the diagonal — a visual confirmation that the log-scale RMSE numbers translate to genuinely close predictions in practice. The **residuals chart** (bottom right) shows residuals scattered symmetrically around zero for all models, with no obvious systematic bias across the price range.

The **normalised leaderboard** (bottom left) scores each model across all three metrics simultaneously. Linear Regression scores highest on RMSE and R², while Gradient Boosting leads on CV stability — making it the more robust choice if deployment generalisability is the priority.

---

### Act 4 — Prediction Deep-Dive

![Prediction Deep-Dive](images/chart3_prediction_deepdive.png)

The final act asks the most important question: **in real dollar terms, how close are the predictions?**

The **predicted vs actual scatter** (top left) shows most homes landing green — within 0–10% of their actual price. Red dots (>20% error) appear primarily in two zones: very low-priced homes (where absolute errors are small but percentage errors amplify) and the upper end of the price range where luxury properties become harder to model.

The **residual distribution** (top centre) is nearly normal and centred tightly at μ = 0.0016 — essentially zero — confirming no systematic over- or under-prediction across the dataset. The near-perfect bell curve is a strong model diagnostic.

The **% error distribution** (top right) is the most portfolio-relevant panel: **70% of test homes are predicted within ±10% of actual price, and 90% are within ±20%**. For a model trained on tabular data without any location-level features like geospatial coordinates or hyperlocal comparables, this is strong real-world accuracy.

The **heteroscedasticity check** (bottom left) shows the smoothed residual mean (green line) staying flat near zero across the full predicted price range — a sign that the model's accuracy is consistent regardless of whether it's predicting a \$100K home or a \$500K home. This is a technically important result: many regression models show fanning residuals at higher price values, but this one holds relatively well.

The **Q-Q plot** (bottom centre) shows residuals tracking the normal line closely through the middle of the distribution, with slight tail deviation — typical for real-world housing data where extreme outliers exist. No major departure from normality.

The **sorted actual vs predicted line** (bottom right) is the most intuitive view: 292 test homes lined up from cheapest to most expensive, with actual prices (green) and predicted prices (purple dashed) nearly overlapping throughout the mid-range. The prediction gap only widens visibly in the top ~20 homes — the ultra-high-end properties where even ensemble models struggle because they represent genuinely unique, hard-to-generalise data points.

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/topeokubanjo-eng/house-price-prediction.git
cd house-price-prediction

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly scipy

# Run the model notebook
jupyter notebook House_Price_Prediction_Model.ipynb

# Run the visualizations notebook (standalone)
jupyter notebook House_Price_Visualizations.ipynb
```

Both notebooks expect `train.csv` and `test.csv` in the same directory.

---

## File Structure

```
📁 House Price Prediction
├── 📓 House_Price_Prediction_Model.ipynb     # Full pipeline: preprocessing, grid search, all models
├── 📓 House_Price_Visualizations.ipynb       # Standalone visual story (self-contained)
├── 📄 train.csv                              # Training data (1,460 homes × 81 features)
├── 📄 test.csv                               # Test data for submission
├── 📄 sample_submission.csv                  # Submission format reference
└── 📁 images/
    ├── chart1_eda.png                        # EDA: distributions & key drivers
    ├── chart2_model_comparison.png           # Model benchmarking dashboard
    ├── chart3_prediction_deepdive.png        # Prediction accuracy analysis
    └── chart4_features.png                   # Feature engineering analysis
```

---

## Key Takeaways

> 🔑 **Log-transforming the target is non-negotiable** — raw SalePrice violates normality assumptions; log(SalePrice) doesn't  
> 🔑 **Engineered TotalSF (r = 0.78) outperforms any single floor area column** — composite features beat raw ones  
> 🔑 **Garages add more value than pools** — homes with garages sell for 1.68× more; pools show a negative premium in this dataset  
> 🔑 **Linear Regression wins on test RMSE (0.1329, R² = 0.905)** — proper preprocessing matters more than model complexity  
> 🔑 **Gradient Boosting is most stable across folds (CV RMSE = 0.131)** — the better production choice if robustness is the goal  
> 🔑 **70% of homes predicted within ±10% of actual price** — strong real-world accuracy for tabular-only data  
> 🔑 **Residuals are normal and unbiased** — the model's assumptions hold, and it doesn't systematically over- or under-predict
