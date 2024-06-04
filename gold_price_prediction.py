import requests
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

url = 'https://example.com'
response = requests.get('https://www.investing.com/commodities/gold-historical-data')
html_content = response.text
soup = BeautifulSoup(html_content, 'html.parser')

table = soup.find('table', {'class': 'freeze-column-w-1'})
rows = table.find_all('tr')

dates = []
prices = []

for row in rows[1:]:
    data = row.find_all('td')
    date = data[0].text
    price = data[1].text
    dates.append(date)
    prices.append(price)
    
# Create a DataFrame from the extracted data
gold_prices_df = pd.DataFrame({'Date': dates, 'Price': prices})
gold_prices_df.to_csv('gold_prices.csv', index=False)
gold_prices_df['Price'] = gold_prices_df['Price'].str.replace(',', '').astype(float)

# Convert dates to datetime objects
gold_prices_df['Date'] = pd.to_datetime(gold_prices_df['Date'], format='%m/%d/%Y').dt.strftime('%Y%m%d')

# Select Features and Target
X = gold_prices_df['Date']
Y = gold_prices_df['Price']

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

X_train_reshaped = X_train.values.reshape(-1, 1)
X_test_reshaped = X_test.values.reshape(-1, 1)

# Normalize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# Train KNN Model
k = 2
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
plt.xlabel('Date')
plt.ylabel('Gold Price (USD)')
plt.title('Gold Price Prediction using KNN')
plt.legend()
plt.show()