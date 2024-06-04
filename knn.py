import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load Data
data = pd.read_csv('Daily.csv')

# Replace commas in numeric columns and convert to float
numeric_columns = ['USD', 'EUR', 'JPY', 'GBP', 'CAD', 'CHF']
for column in numeric_columns:
    data[column] = data[column].str.replace(',', '').astype(float)

# # Extract year from the Date column
# data['Year'] = pd.to_datetime(data['Date'], format='%m/%d/%Y').dt.year

# Extract month and year from the Date column and combine them
data['Month-Year'] = pd.to_datetime(data['Date'], format='%m/%d/%Y').dt.strftime('%Y%m')

# Select Features and Target
# X = data[['Year']]
X = data[['Month-Year']]
Y = data['USD']

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train KNN Model
k = 5
knn = KNeighborsRegressor(n_neighbors=k) 
knn.fit(X_train_scaled, Y_train)
Y_pred = knn.predict(X_test_scaled)

# Sort test data and predicted prices by year
sorted_indices_test = np.argsort(X_test.values.flatten())
X_test_sorted = X_test.values.flatten()[sorted_indices_test]
Y_test_sorted = Y_test.values[sorted_indices_test]
Y_pred_sorted = Y_pred[sorted_indices_test]

# Plot Results
plt.figure(figsize=(10, 6))
# plt.plot(X['Month-Year'], Y, label='Real Prices')
plt.plot(X_test_sorted, Y_test_sorted, color='red', label='Actual Test Prices')
plt.plot(X_test_sorted, Y_pred_sorted, color='blue', label='Predicted Prices')
plt.xlabel('Year')
plt.ylabel('Gold Price (USD)')
plt.title('Gold Price Prediction using KNN')
plt.legend()
plt.show()

