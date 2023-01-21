import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from xgboost import plot_tree

# Load the data
df = pd.read_csv("data.csv")

# Impute missing values and remove rows with invalid values
df.dropna(inplace=True)
df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

# Split the data into features and target
X = df[["sex", "age", "bmi", "hh_score", "mfisher", "htn", "smoke", "size_mm", "tx", "mrs_discharge", "last_mrs", "infarct", "monocytes"]]
y = df["dci"]

# Train the model
model = xgb.XGBRegressor(random_state=0).fit(X, y)

while True:
    # Get user input for the 12 variables
    sex = int(input("Enter the Sex (0= Male, 1= Female): "))
    age = float(input("Enter the Age: "))
    bmi = float(input("Enter the BMI: "))
    hh_score = float(input("Enter The Hunt-Hess Score: "))
    mfisher = float(input("Enter The Modified Fisher Scale: "))
    htn = int(input("Is The Patient Hypertensive or Not? (1= Yes, 0= No): "))
    smoke = int(input("Is The Patient Smoker or Not? (1= Yes, 0= No): "))
    size_mm = float(input("Enter the Aneurysm Size In mm: "))
    tx = int(input("What Is The Treatment Modality (1= Microsurgical Clipping, 2= Endovascular Coiling): "))
    mrs_discharge = float(input("Enter the mRS Score at Discharge: "))
    last_mrs = float(input("Enter the Last mRS Score: "))
    infarct = int(input("Did the Patient Get Infarction? (1= Yes, 0= No): "))
    monocytes = float(input("Enter the Monocyte count (10^3/uL): "))

    # Make a prediction using the model
    input_data = pd.DataFrame([[sex, age, bmi, hh_score, mfisher, htn, smoke, size_mm, tx, mrs_discharge, last_mrs, infarct, monocytes]],
                              columns=["sex", "age", "bmi", "hh_score", "mfisher", "htn", "smoke", "size_mm", "tx", "mrs_discharge", "last_mrs", "infarct", "monocytes"])
    prediction = model.predict(input_data)[0]
    print(f"Predicted Percentage for Delayed Cerebral Ischemia Is: % {prediction*100}")
    
    fig, ax = plt.subplots(figsize=(50, 50), dpi=300)
    plot_tree(model, num_trees=0, rankdir='TB', ax=ax)
    plt.show()
