# Pizza Sale Prediction Using Decision Tree Regression

A machine learning project that applies a Decision Tree Regressor to predict pizza sales revenue (`total_price`) from historical transactional data. The pipeline covers data preprocessing, feature engineering, baseline modeling, 5-fold cross-validation, and hyperparameter tuning via GridSearchCV — with detailed interpretation at each stage.

---

## Overview

This project addresses a regression problem in the food service domain: given a set of order-level features such as pizza size, category, quantity, unit price, and order date, the goal is to accurately predict the total sale amount per transaction. The notebook is written to be fully self-explanatory, combining executable code with step-by-step commentary and metric interpretation at each stage.

The dataset is sourced from the Pizza Sales Dataset on Kaggle.

---

## Dataset

The dataset (`pizza_sales.csv`) contains historical pizza order records. The target variable is `total_price` — the total revenue generated per order line.

| Feature | Description |
|---|---|
| `order_id` | Unique identifier for each order |
| `order_date` | Date the order was placed |
| `order_time` | Time the order was placed |
| `pizza_name` | Full name of the pizza |
| `pizza_name_id` | Short identifier for the pizza |
| `pizza_category` | Category of the pizza (e.g., Classic, Veggie, Supreme) |
| `pizza_size` | Size of the pizza (S, M, L, XL, XXL) |
| `pizza_ingredients` | List of toppings |
| `quantity` | Number of units ordered |
| `unit_price` | Price per unit |
| `total_price` | Total revenue for the order line — target variable |

---

## Methodology

### 1. Data Preprocessing

- Dropped rows with missing values
- Parsed `order_date` using `dayfirst=True` with error coercion to handle mixed date formats
- Extracted temporal features from the date column: `year`, `month`, `day`, `weekday`
- Dropped non-informative or high-cardinality text columns: `pizza_ingredients`, `order_date`, `order_time`, `pizza_name`, `pizza_name_id`
- Applied One-Hot Encoding to `pizza_size` and `pizza_category` using `get_dummies` with `drop_first=True`

### 2. Baseline Model

A default `DecisionTreeRegressor` with no depth constraints was trained on an 80/20 train-test split with `random_state=42`.

**Baseline results:**

| Metric | Value |
|---|---|
| MAE | 0.000574 |
| MSE | 0.000171 |
| RMSE | 0.02679 |

While the baseline metrics appear strong, the near-zero errors indicate potential overfitting — a known risk with unconstrained decision trees.

### 3. Cross-Validation

5-fold cross-validation was applied to the baseline model to assess generalization performance across different data splits.

**Cross-validation results:**

| Metric | Value |
|---|---|
| Fold 1 RMSE | 0.0312 |
| Fold 2 RMSE | 0.0240 |
| Fold 3 RMSE | 0.0120 |
| Fold 4 RMSE | 0.4442 |
| Mean RMSE | 0.1076 |
| Std Dev of RMSE | 0.1869 |

The high variance across folds — particularly Fold 4 with an RMSE of 0.444 — confirmed that the default model was overfitting and not generalizing consistently across different data splits.

### 4. Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was applied over the following parameter grid:

| Parameter | Values Searched |
|---|---|
| `max_depth` | 3, 5, 10, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |

**Best parameters identified:**

| Parameter | Optimal Value |
|---|---|
| `max_depth` | 10 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 1 |
| Best RMSE (CV) | 0.1424 |

### 5. Optimized Model Evaluation

The model was retrained using the best parameters and evaluated on the held-out test set.

**Optimized model results:**

| Metric | Value |
|---|---|
| MAE | 0.001904 |
| MSE | 0.001732 |
| RMSE | 0.04162 |

The tuned model generalizes significantly better than the default configuration. Constraining `max_depth` to 10 and requiring a minimum of 10 samples per split reduced variance and overfitting while maintaining low prediction error.

---

## Key Findings

- Temporal features (`weekday`, `month`, `day`) extracted from the order date contribute meaningfully to predicting total sales revenue.
- Pizza size and category are strong structural predictors of total price, captured effectively through one-hot encoding.
- The unconstrained baseline model showed clear signs of overfitting, evidenced by near-zero training errors and high cross-validation variance.
- Hyperparameter tuning via GridSearchCV produced a more robust and generalizable model with consistent performance across folds.
- Ensemble methods such as Random Forest or Gradient Boosting are recommended as the next step for further performance and stability improvements.

---

## Project Structure

## Technologies Used

- Python 3
- Pandas, NumPy — data manipulation and feature engineering
- Scikit-learn — Decision Tree Regressor, GridSearchCV, cross-validation
- Matplotlib — visualization
- Jupyter Notebook — interactive development environment
- Kaggle — dataset source and execution environment

---

## Getting Started

1. Clone the repository:
```bash
   git clone https://github.com/sahandyousefi/Pizza-Sale-Decision-Tree.git
   cd Pizza-Sale-Decision-Tree
```

2. Install dependencies:
```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
```

3. Place the dataset file `pizza_sales.csv` in the expected input path, or update the file path in the notebook to match your local setup.

4. Launch the notebook:
```bash
   jupyter notebook advanced-decision-tree-fully-explained.ipynb
```

---

## Author

Sahand Yousefi
[GitHub](https://github.com/sahandyousefi) | [LinkedIn](https://www.linkedin.com/in/sahand-yousefi/)

