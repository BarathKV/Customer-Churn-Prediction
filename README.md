# Customer Churn Prediction System

This repository contains a comprehensive machine learning pipeline to predict **customer churn** in a telecom dataset. The project explores data preprocessing, feature engineering, exploratory data analysis (EDA), model training, hyperparameter tuning, and ensemble learning using various classification models.

---

## Project Steps

### Data Preprocessing

- Convert `TotalCharges` from object to numeric (float).
- Handle missing values by replacing them with `0`.
- Encode the target variable `Churn` into binary format: `Yes → 1`, `No → 0`.
- Decode `SeniorCitizen` from `0/1` to `No/Yes`.
- Bin `tenure` into equal-width intervals.
- Feature engineering:
  - **Customer Lifetime Value (CLV)** = `tenure × MonthlyCharges`
  - **AvgMonthlyCharges** = `TotalCharges / tenure`

---

### Exploratory Data Analysis (EDA)

- **Target distribution:** Visualize class imbalance.
- **Univariate Analysis:**
  - *Numerical:* `tenure`, `MonthlyCharges`, `TotalCharges`, `CLV`, `AvgMonthlyCharges` (using histplots)
  - *Categorical:* All service-related and demographic fields (using countplots)
- **Bivariate Analysis:**
  - Boxplots for numerical vs target
  - Countplots for categorical vs target
  - Correlation heatmap (only numeric + target)

---

### Encoding

- **Label Encoding** for binary/ordinal features.
- **One-Hot Encoding** for nominal multi-class features.

---

### Dimensionality Reduction

- Applied PCA on both encoding methods.
- Evaluated variance retention using cumulative sum plot.

---

### Class Imbalance Handling

Tested different **sampling techniques**:
- No sampling
- SMOTE
- ENN
- SMOTE-ENN

---

### Model Training (TODO CHECK ALL MODELS)

Trained and evaluated a range of classification models:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Naive Bayes
- K-Nearest Neighbors
- Support Vector Machine

Each model was tested under multiple encoding and sampling combinations.

---

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

### Model Selection & Hyperparameter Tuning

- Top models selected based on ROC-AUC and F1.
- Hyperparameter tuning using GridSearchCV on top 5 models.
- Plotted:
  - ROC-AUC Curves
  - Precision-Recall Curves

---

### Ensemble Models

Tested advanced ensemble strategies:
- **Voting Classifier**
- **Stacking Classifier**

Final ensemble selected based on evaluation metric performance.

---

## Libraries Used

| Library       | Purpose                                                      |
|---------------|--------------------------------------------------------------|
| `pandas`      | Data manipulation and preprocessing                          |
| `numpy`       | Numerical operations and calculations                        |
| `matplotlib`  | Data visualization                                            |
| `seaborn`     | Enhanced statistical visualizations                          |
| `scikit-learn`| ML algorithms, preprocessing, model evaluation, and tuning   |
| `xgboost`     | Extreme Gradient Boosting classifier                         |
| `lightgbm`    | Efficient gradient boosting framework                        |
| `imblearn`    | Sampling techniques like SMOTE, ENN, SMOTE-ENN               |

---

## Contributors

- [Barath K V](https://github.com/BarathKV)
- [Lalith Abhishek G](https://github.com/LalithAbhishekG)
- Nisanth D
