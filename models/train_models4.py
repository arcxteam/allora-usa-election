# this model own by me : github.com/arcxteam as 0xgan
# run this to modify and get your point on topicID 11 
# model : knn / K-NearestNeighbors
# By me : Oxgan/grey
# Before copy, handling improve tweak this model. you can ask on chatGPT

import pandas as pd
import numpy as np
import os
import requests
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from config import data_base_path, model_file_path, TOKEN

def get_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()['history']
        df = pd.DataFrame(data)
        df['t'] = pd.to_datetime(df['t'], unit='s')
        df.columns = ['Timestamp', 'Predict']
        df['Predict'] = df['Predict'].apply(lambda x: x * 100)
        print(df.head())
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
    return df

def download_data(token):
    os.makedirs(data_base_path, exist_ok=True)
    if token == 'R':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=21742633143463906290569050155826241533067272736897614950488156847949938836455&fidelity=1000"
        data = get_data(url)
        save_path = os.path.join(data_base_path, 'polymarket_R.csv')
        data.to_csv(save_path)
    elif token == 'D':
        url = "https://clob.polymarket.com/prices-history?interval=all&market=69236923620077691027083946871148646972011131466059644796654161903044970987404&fidelity=1000"
        data = get_data(url)
        save_path = os.path.join(data_base_path, 'polymarket_D.csv')
        data.to_csv(save_path)

def train_model(token):
    training_price_data_path = os.path.join(data_base_path, f"polymarket_{token}.csv")
    df = pd.read_csv(training_price_data_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['year'] = df['Timestamp'].dt.year
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.day
    df['hour'] = df['Timestamp'].dt.hour

    X = df[['year', 'month', 'day', 'hour']]
    y = df['Predict']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipeline with StandardScaler and KNN Regressor
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scaling the features
        ('knn', KNeighborsRegressor(n_neighbors=5))  # K-Nearest Neighbors Regressor
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Making predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save the model
    os.makedirs(model_file_path, exist_ok=True)
    save_path_model = os.path.join(model_file_path, f'knn_model_{token}.pkl')
    joblib.dump(pipeline, save_path_model)
    print(f"Trained model saved to {save_path_model}")

def get_inference(token):
    save_path_model = os.path.join(model_file_path, f'knn_model_{token}.pkl')
    loaded_model = joblib.load(save_path_model)
    print("Loaded model successfully")

    single_input = pd.DataFrame({
        'year': [2024],
        'month': [10],
        'day': [10],
        'hour': [12]
    })

    # Scaling single input
    predicted_price = loaded_model.predict(single_input)
    return predicted_price[0]