# this model own by : github.com/hangytongy
# run this to modify and get your point on topicID 11 
# model : XGB

import random
import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime
from xgboost import XGBRegressor
import joblib


def get_data(url):

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()['history']

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Convert 't' (timestamp) to datetime format
        df['t'] = pd.to_datetime(df['t'], unit='s')

        # Rename columns for clarity
        df.columns = ['Timestamp', 'Predict']
        df['Predict'] = df['Predict'].apply(lambda x : x*100)

        # Display the first few rows
        print(df.head())
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        
    return df

def download_data(token):
    print(f"base path : {data_base_path}")
    os.makedirs(data_base_path, exist_ok=True)
    if token == 'R':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=21742633143463906290569050155826241533067272736897614950488156847949938836455&fidelity=720"
        data = get_data(url)
        save_path = os.path.join(data_base_path,'polymarket_R.csv')
        data.to_csv(save_path)

    elif token =='D':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=69236923620077691027083946871148646972011131466059644796654161903044970987404&fidelity=720"
        data = get_data(url)
        save_path = os.path.join(data_base_path,'polymarket_D.csv')
        os.makedirs(os.path.dirname(data_base_path), exist_ok=True)
        data.to_csv(save_path)

def train_model(token):
    # Load the price data
    print("wtf am i running this???")
    if token == 'R':
        training_price_data_path = os.path.join(data_base_path, "polymarket_R.csv")
    elif token == 'D':
        training_price_data_path = os.path.join(data_base_path, "polymarket_D.csv")

    df = pd.read_csv(training_price_data_path)

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour
    
    # Define feature (X) and target (y)
    X = df[['year', 'month', 'day', 'hour']]
    y = df['Predict']

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBRegressor model
    model = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    
    print(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # create the model's parent directory if it doesn't exist
    os.makedirs(model_file_path, exist_ok=True)
    if token == 'R':
        save_path_model = os.path.join(model_file_path,'xgb_model_R.pkl')
    elif token == 'D':
        save_path_model = os.path.join(model_file_path,'xgb_model_D.pkl')

    joblib.dump(model,save_path_model)

    print(f"Trained model saved to {save_path_model}")


def get_inference(token):

    if token == 'R':
        save_path_model = os.path.join(model_file_path,'xgb_model_R.pkl')

    elif token == 'D':
        save_path_model = os.path.join(model_file_path,'xgb_model_D.pkl')

    loaded_model = joblib.load(save_path_model)
    print("loaded model succesfully")

    single_input = pd.DataFrame({
    'year': [2024],
    'month': [10],
    'day': [10],
    'hour': [12]
    })

    predicted_price = loaded_model.predict(single_input)


    return predicted_price[0]
