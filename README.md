# Gold Price Prediction using KNN

This project fetches historical gold price data from a website, preprocesses it, trains a K-Nearest Neighbors (KNN) regression model, and then plots the actual test prices against the predicted prices.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your_username/gold-price-prediction.git
    ```

2. **Install the required dependencies**:

    ```bash
    pip install numpy pandas scikit-learn matplotlib requests beautifulsoup4
    ```

## Usage

1. **Run the script `gold_price_prediction.py`**:

    ```bash
    python gold_price_prediction.py
    ```

2.  **Run the script `gold_price_prediction.py`**:
    ```bash
    python knn.py
    ```

## Description

- `gold_price_prediction.py`: Python script that extracts real-time price data from https://www.investing.com/commodities/gold-historical-data, preprocesses it, trains a KNN regression model, and plots the results.
- `knn.py`: Python script that takes price data since 1978 to 2023 from https://www.kaggle.com/datasets/rizkykiky/gold-price-dataset?resource=download&select=Daily.csv, preprocesses it, trains a KNN regression model, and plots the results.

## References

- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Real-Time Gold Prices](https://www.investing.com/commodities/gold-historical-data)
- [Gold Prices from 1978 to 2023](https://www.kaggle.com/datasets/rizkykiky/gold-price-dataset?resource=download&select=Daily.csv)
