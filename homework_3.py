#!/usr/bin/env python
# coding: utf-8
import pickle
from pathlib import Path

import pandas as pd
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

def read_dataframe(year, month):
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)

    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    #df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']


    
    return df


def create_X(df, dv=None):
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if dv is None:
        dv = DictVectorizer(sparse=True)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    
    return X, dv


from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train, X_val, y_val, dv):
    with mlflow.start_run() as run:
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        intercept = lr.intercept_

        mlflow.log_metric("rmse", rmse)
        mlflow.log_param("intercept", intercept)

        # Guardar y registrar el preprocesador
        preprocessor_path = "models/preprocessor.b"
        with open(preprocessor_path, "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")

        # Registrar el modelo con MLflow
        mlflow.sklearn.log_model(lr, artifact_path="models")

        return run.info.run_id




def run(year, month):
    df_train = read_dataframe(year=year, month=month)

    next_year = year if month < 12 else year + 1
    next_month = month + 1 if month < 12 else 1

    df_val = read_dataframe(year=next_year, month=next_month)

    X_train, dv = create_X(df_train)
    X_val, _ = create_X(df_val, dv)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val['duration'].values

    #train_model(X_train, y_train, X_val, y_train, dv)
    run_id = train_model(X_train, y_train, X_val, y_val, dv)
    print(f"Mlflow run_id: {run_id}")
    return run_id

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description= 'train a model to predict taxi trip duration')
    parser.add_argument('--year', type=int, required=True, help='year of the data totrain' )
    parser.add_argument('--month', type=int, required=True, help='month of the data to train')
    args = parser.parse_args()

    run_id = run(year=args.year, month=args.month)
    #Save id to a file

    with open('run_id.txt', "w") as f:
        f.write(run_id)


