# Pizza-Sale-Decision-Tree
This project predicts pizza sales using a Decision Tree Regressor. It includes cross-validation for robustness and hyperparameter tuning with GridSearchCV to optimize performance, achieving low prediction errors. The pipeline showcases effective modeling for business insights.

# 📌 Project Overview
This project aims to predict pizza sales using a Decision Tree model based on various influencing factors. The dataset includes historical sales data with relevant features such as date, weather conditions, promotions, pizza type, and store location. The goal is to create an accurate predictive model to help businesses forecast sales and optimize inventory management.

# 📊 Dataset
The dataset consists of:

Date: The day of sales.
Store ID: Unique identifier for each store.
Pizza Type: Type of pizza sold (e.g., Margherita, Pepperoni, BBQ Chicken).
Price: Selling price of each pizza.
Promotion: Whether a promotion was applied (Yes/No).
Weather Conditions: Information on weather (Sunny, Rainy, Snowy, etc.).
Sales Volume: Number of pizzas sold (Target Variable).

# 🏗️ Approach
## 1. Data Preprocessing

Handling missing values.
Encoding categorical variables.
Feature scaling (if necessary).
Train-test split.

## 2. Model Selection

Used a Decision Tree Regressor to predict sales volume.
Hyperparameter tuning for optimal depth and leaf nodes.
Evaluated using RMSE, MAE, and R² Score.

## 3. Results & Insights

Identified key factors influencing sales (e.g., promotions, weekends, weather).
Visualized decision tree splits to understand decision boundaries.
Compared performance with baseline models.

# 🚀 Technologies Used
Python
Pandas & NumPy (Data preprocessing)
Scikit-Learn (Decision Tree modeling)
Matplotlib & Seaborn (Data visualization)
Jupyter Notebook (Development environment)
