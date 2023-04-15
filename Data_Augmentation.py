# ------------------- (SMOTE + KNN) and (SMOTE + ENN) ------------------- 

# Load libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler

# Load the Dataset
data = pd.read_csv('data_e.csv')

# Normalize the features.
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data.iloc[:, :-1])

# Convert the target variable into a categorical variable
num_bins = 3  # Adjust this value based on your domain knowledge
labels = [f'bin_{i + 1}' for i in range(num_bins)]
data.iloc[:, -1] = pd.cut(data.iloc[:, -1], bins=num_bins, labels=labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_data, data.iloc[:, -1], test_size=0.2, random_state=42)

# Train a k-Nearest Neighbors model on the bootstrapped data and evaluate its accuracy
def train_and_evaluate_knn(X_train, y_train, X_test, y_test, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    return accuracy_score(y_test, y_pred)


# Test different k values and find the one with the highest accuracy
k_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
best_k = k_values[0]
best_accuracy = 0

for k in k_values:
    accuracy = train_and_evaluate_knn(X_train, y_train, X_test, y_test, k)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k
best_k = 10

# Find the minimum number of samples per class in the training set
min_samples_per_class = y_train.value_counts().min()

# Adjust the k_neighbors parameter for SMOTE if needed
smote_k_neighbors = 10

# Set the desired number of generated data points
num_generated_data = 3200  # Set this value according to your needs

# Calculate the target number of samples per class
target_samples_per_class = (min_samples_per_class + num_generated_data) // num_bins

# Perform SMOTE + KNN
sampling_strategy = {label: target_samples_per_class for label in labels}
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Perform SMOTE + ENN
enn = EditedNearestNeighbours(sampling_strategy='all', n_neighbors=best_k)
X_resampled_enn, y_resampled_enn = enn.fit_resample(X_resampled, y_resampled)

# Denormalize the resampled data and convert it back to a pandas DataFrame
denormalized_data = scaler.inverse_transform(X_resampled_enn)
denormalized_data = pd.DataFrame(denormalized_data, columns=data.columns[:-1])

# Add the target variable back to the denormalized data
denormalized_data['target'] = y_resampled_enn

# Round the categorical/discrete variables to the nearest integer
# Replace 'column_name' with the appropriate column names from your dataset
denormalized_data['age'] = denormalized_data['age'].round().astype(int)
denormalized_data['nimodipine'] = denormalized_data['nimodipine'].round().astype(int)
denormalized_data['hh_score'] = denormalized_data['hh_score'].round().astype(int)
denormalized_data['mfisher'] = denormalized_data['mfisher'].round().astype(int)
denormalized_data['htn'] = denormalized_data['htn'].round().astype(int)
denormalized_data['dbt'] = denormalized_data['dbt'].round().astype(int)
denormalized_data['hypercholestorelemia'] = denormalized_data['hypercholestorelemia'].round().astype(int)
denormalized_data['chf'] = denormalized_data['chf'].round().astype(int)
denormalized_data['cancer'] = denormalized_data['cancer'].round().astype(int)
denormalized_data['smoke'] = denormalized_data['smoke'].round().astype(int)
denormalized_data['alcohol'] = denormalized_data['alcohol'].round().astype(int)
denormalized_data['cocaine'] = denormalized_data['cocaine'].round().astype(int)
denormalized_data['fh_an'] = denormalized_data['fh_an'].round().astype(int)
denormalized_data['location'] = denormalized_data['location'].round().astype(int)
denormalized_data['size'] = denormalized_data['size'].round().astype(int)
denormalized_data['side'] = denormalized_data['side'].round().astype(int)
denormalized_data['tx'] = denormalized_data['tx'].round().astype(int)
denormalized_data['evd'] = denormalized_data['evd'].round().astype(int)
denormalized_data['vp_shunt'] = denormalized_data['vp_shunt'].round().astype(int)
denormalized_data['dci'] = denormalized_data['dci'].round().astype(int)
denormalized_data['tcd_vs'] = denormalized_data['tcd_vs'].round().astype(int)
denormalized_data['angio_vs'] = denormalized_data['angio_vs'].round().astype(int)
denormalized_data['cvs'] = denormalized_data['cvs'].round().astype(int)

# Save the augmented dataset to a CSV file.
denormalized_data.to_csv('augmented_data.csv', index=False)
