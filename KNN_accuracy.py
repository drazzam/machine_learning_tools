# ------------------ Testing KNN Accuracy ------------------


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer

# Load the Dataset
data = pd.read_csv('data_e.csv')

# Normalize the features.
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.iloc[:, :-1])

# Convert the target variable into a categorical variable
num_bins = 3  # Adjust this value based on your domain knowledge
labels = [f'bin_{i+1}' for i in range(num_bins)]
data.iloc[:, -1] = pd.cut(data.iloc[:, -1], bins=num_bins, labels=labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_data, data.iloc[:, -1], test_size=0.2, random_state=42)

# Train a k-Nearest Neighbors model on the bootstrapped data and evaluate its accuracy
def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors= 8)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Test different k values and find the one with the highest accuracy
k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
best_k = k_values[0]
best_accuracy = 0

for k in k_values:
    accuracy = train_and_evaluate_knn(X_train, y_train, X_test, y_test, k)
    print(f"Accuracy for k={k}: {accuracy:.2f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

print(f"Best k value: {best_k}, with an accuracy of {best_accuracy:.2f}")
