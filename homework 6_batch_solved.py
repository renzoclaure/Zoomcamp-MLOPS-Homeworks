#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd


# ---------- helpers para rutas dinámicas ---------- #

def get_input_path(year, month):
    default = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    return os.getenv('INPUT_FILE_PATTERN', default).format(year=year, month=month)

def get_output_path(year, month):
    default = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    return os.getenv('OUTPUT_FILE_PATTERN', default).format(year=year, month=month)

# ---------- IO helpers ---------- #

def read_data(path, categorical):
    """Lee parquet desde local o S3 (Localstack)."""
    s3_url = os.getenv('S3_ENDPOINT_URL')
    storage_opts = {'client_kwargs': {'endpoint_url': s3_url}} if path.startswith('s3://') and s3_url else None
    df = pd.read_parquet(path, storage_options=storage_opts)
    return df

def save_data(df, path):
    """Guarda parquet en local o S3 (Localstack)."""
    s3_url = os.getenv("S3_ENDPOINT_URL")
    storage_opts = {'client_kwargs': {'endpoint_url': s3_url}} if path.startswith('s3://') and s3_url else None
    df.to_parquet(
        path,
        engine="pyarrow",
        index=False,
        compression=None,
        storage_options=storage_opts,
    )

# ---------- transformación ---------- #

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype(int).astype(str)
    return df


# ---------- flujo principal ---------- #

def main(year: int, month: int):
    input_path  = get_input_path(year, month)
    output_path = get_output_path(year, month)

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ['PULocationID', 'DOLocationID']

    df_raw = read_data(input_path, categorical)
    df     = prepare_data(df_raw, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    save_data(df_result, output_path)


# ---------- entrypoint ---------- #

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python batch_solved.py YEAR MONTH")
        sys.exit(1)

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)
