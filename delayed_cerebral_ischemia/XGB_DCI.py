# ---------------------------------- Importing The Libraries ---------------------------------- # 

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from xgboost import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
from sklearn.model_selection import KFold
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------- Building and Training The Model Section ---------------------------------- # 

# Load the data
df = pd.read_csv("data.csv")

# Impute missing values and remove rows with invalid values
df.dropna(inplace=True)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# Split the data into features and target
X = df[["sex", "age", "bmi", "hh_score", "mfisher", "htn", "smoke", "size_mm", "tx", "mrs_discharge", "last_mrs", "infarct", "monocytes"]]
y = df["dci"]

# Train the model
model = xgb.XGBRegressor(random_state=0)
model.fit(X, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Get user input for the 12 variables
sex = 0 #Sex: Male= 0, Female= 1
age = 50
bmi = 30
hh_score = 3
mfisher = 3
htn = 1 #Hypertension: Yes= 1, No= 0
smoke = 1 #Smoking: Yes= 1, No= 0
size_mm = 25 #Size of Aneurysm in mm
tx = 1 #Treatment Modality: Microsurgical Clipping= 1, Endovascular Coiling= 2
mrs_discharge = 2
last_mrs = 2
infarct = 1 #Cerebral Infraction: Yes= 1, No= 0
monocytes = 0.5 #Monocyte Count in 10^3/uL

# Make a prediction using the model
input_data = pd.DataFrame([[sex, age, bmi, hh_score, mfisher, htn, smoke, size_mm, tx, mrs_discharge, last_mrs, infarct, monocytes]],
                          columns=["sex", "age", "bmi", "hh_score", "mfisher", "htn", "smoke", "size_mm", "tx", "mrs_discharge", "last_mrs", "infarct", "monocytes"])
prediction = model.predict(input_data)[0]
print(f"Predicted Percentage for Delayed Cerebral Ischemia Is: % {prediction*100}")

# ---------------------------------- Testing and Validating the Model Section ---------------------------------- # 

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2:.2f}")

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")

# Calculate MSE
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")

# Calculate RMSE
rmse = sqrt(mse)
print(f"RMSE: {rmse:.2f}")
    
# Get the feature importance values
importance = model.feature_importances_

# Create a list of feature names
features = X.columns

# Create a dataframe of the feature importance values
importance_df = pd.DataFrame(importance, index=features, columns=["Importance"])

# Sort the dataframe by the importance values
importance_df.sort_values(by="Importance", ascending=False, inplace=True)

# Print the features
print(importance_df)

# Calculate the residuals
residuals = y_test - y_pred

# Create a figure and two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Scatter plot of true values vs. predictions
ax1.scatter(y_test, y_pred)
ax1.set_xlabel('True Values')
ax1.set_ylabel('Predictions')

# Residual plot
ax2.scatter(y_pred, residuals)
ax2.set_xlabel('Predictions')
ax2.set_ylabel('Residuals')

# Show the plot
plt.show()

# Create a SHAP values plot
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
shap.summary_plot(shap_values, X)
