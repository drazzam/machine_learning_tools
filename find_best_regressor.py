import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Load dataset
df = pd.read_csv('S1Data.csv')

# Set your target variable and input features
target = 'ICP'
features = df.drop(target, axis=1)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

# Define a function for evaluating models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, r2

# List of models
models = [
    ('Linear Regression', LinearRegression()),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Polynomial Regression', Pipeline([('poly', PolynomialFeatures(degree=2)), ('linear', LinearRegression())])),
    ('Ridge Regression', Ridge()),
    ('Lasso Regression', Lasso()),
    ('ElasticNet Regression', ElasticNet()),
    ('Support Vector Regression', SVR()),
    ('Decision Tree Regression', DecisionTreeRegressor()),
    ('Random Forest Regression', RandomForestRegressor()),
    ('Gradient Boosting Regression', GradientBoostingRegressor()),
    ('XGBoost Regression', xgb.XGBRegressor()),
    ('LightGBM Regression', lgb.LGBMRegressor()),
    ('CatBoost Regression', CatBoostRegressor(verbose=0)),
    ('K-Nearest Neighbors Regression', KNeighborsRegressor()),
    ('MLP Neural Networks', MLPRegressor(max_iter=1000))
]

# Train and evaluate models
results = []

for name, model in models:
    model.fit(X_train, y_train)
    mse, rmse, r2 = evaluate_model(model, X_test, y_test)
    results.append((name, mse, rmse, r2))
    print(f"{name}: MSE={mse:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

# Find the best model
best_model = sorted(results, key=lambda x: x[2])[0]
print(f"\nBest model: {best_model[0]}, RMSE={best_model[2]:.4f}")
