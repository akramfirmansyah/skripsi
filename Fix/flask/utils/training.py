import csv
import os
import pickle
from datetime import datetime
from pathlib import Path

import influxdb_client
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Integer, Real

from constant.directory import logs_dir, model_dir
from utils.create_features import create_features


def get_data(measurement: str = "first", field: str = "airTemperature") -> pd.DataFrame:
    load_dotenv()

    token = os.getenv("TOKEN")
    url = os.getenv("URL")
    org = os.getenv("ORG")
    bucket = os.getenv("BUCKET")
    start_date = os.getenv("START_DATE")

    client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

    query_api = client.query_api()

    query = f"""from(bucket: "{bucket}")
                |> range(start: {start_date})
                |> filter(fn: (r) => r["_measurement"] == "{measurement}")
                |> filter(fn: (r) => r["_field"] == "{field}")
                |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
                |> yield(name: "mean")"""

    tables = query_api.query(query=query, org=org)

    data = []
    for table in tables:
        for record in table.records:
            data.append(
                {
                    "datetime": record.get_time(),
                    f"{field}": round(record.get_value(), 2),
                }
            )

    df = pd.DataFrame(data)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None)
    df = df.set_index("datetime").asfreq("1min")

    return df


def replace_outlier(data: pd.DataFrame, column: str) -> pd.DataFrame:
    temp = data.copy()

    q1, q3 = temp[f"{column}"].quantile(0.25), temp[f"{column}"].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    temp.loc[temp[f"{column}"] > upper_limit, f"{column}"] = upper_limit
    temp.loc[temp[f"{column}"] < lower_limit, f"{column}"] = lower_limit

    return temp


def fill_na(data: pd.DataFrame, column: str) -> pd.DataFrame:
    # Get mean HH:MM
    resampled_df = data.copy()
    resampled_df = resampled_df.resample("min").mean()
    resampled_df["datetime"] = resampled_df.index
    resampled_df["hour_minutes"] = resampled_df["datetime"].dt.strftime("%H:%M")
    resampled_df = resampled_df.groupby("hour_minutes").mean()
    resampled_df = resampled_df.drop(columns=["datetime"])
    resampled_df = resampled_df.round({f"{column}": 2})

    # Fill NaN with mean HH:MM
    df_fillna = data.copy()
    df_fillna["hour_minutes"] = df_fillna.index.strftime("%H:%M")
    df_fillna[f"{column}"] = df_fillna.apply(
        lambda row: (
            row[f"{column}"]
            if pd.notnull(row[f"{column}"])
            else resampled_df.loc[row["hour_minutes"], f"{column}"]
        ),
        axis=1,
    )
    df_fillna = df_fillna.drop(columns=["hour_minutes"])

    return df_fillna


def preprocessing() -> tuple:
    # Get data from InfluxDB
    df_airTemp = get_data(measurement="first", field="airTemperature")
    df_humidity = get_data(measurement="first", field="humidity")

    # Replace outlier with
    df_airTemp = replace_outlier(df_airTemp, "airTemperature")
    df_humidity = replace_outlier(df_humidity, "humidity")

    # Fill NaN with mean HH:MM
    df_airTemp = fill_na(df_airTemp, "airTemperature")
    df_humidity = fill_na(df_humidity, "humidity")

    # Create features
    df_airTemp = create_features(df_airTemp)
    df_humidity = create_features(df_humidity)

    return df_airTemp, df_humidity


def create_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=10,
        subsample=1,
        colsample_bytree=1,
        gamma=2.5,
        reg_alpha=0,
        reg_lambda=7,
    )

    model.fit(X_train, y_train)

    return model


def create_model_with_hyperparameter_tuning(
    X_train: pd.DataFrame, y_train: pd.DataFrame
) -> xgb.XGBRegressor:
    search_space = {
        "n_estimators": Integer(100, 1000),
        "learning_rate": Real(0.01, 0.1, prior="uniform"),
        "max_depth": Integer(3, 10),
        "subsample": Real(0.5, 1.0, prior="uniform"),
        "colsample_bytree": Real(0.5, 1.0, prior="uniform"),
        "gamma": Real(0, 5, prior="uniform"),
        "reg_alpha": Real(0, 10, prior="uniform"),
        "reg_lambda": Real(0, 10, prior="uniform"),
    }

    opt = BayesSearchCV(
        xgb.XGBRegressor(),
        search_space,
        n_iter=50,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    opt.fit(X_train, y_train)

    return opt.best_estimator_


def save_model(model: xgb.XGBRegressor, model_name: str) -> None:
    filepath = Path(f"{model_dir}XGBoost_{model_name}.pkl")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(model, open(filepath, "wb"))


def save_matrix(
    model, X_test: pd.DataFrame, y_test: pd.DataFrame, column_name: str
) -> None:
    test_time = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
    model = model

    y_test["prediction"] = model.predict(X_test)

    mae = mean_absolute_error(y_test[f"{column_name}"], y_test["prediction"])
    mape = mean_absolute_percentage_error(
        y_test[f"{column_name}"], y_test["prediction"]
    )
    rmse = root_mean_squared_error(y_test[f"{column_name}"], y_test["prediction"])

    # Define the header and data
    header = ["datetime", "mae", "mape", "rmse"]
    data = [test_time, mae, mape, rmse]

    # Check if the file exists and has a header
    file_exists = os.path.isfile(f"{logs_dir}metrix_{column_name}.csv")
    header_exists = False

    if file_exists:
        with open(f"{logs_dir}metrix_{column_name}.csv", "r") as file:
            reader = csv.reader(file)
            # Check the first row to see if it's the header
            header_exists = any(row == header for row in reader)

    # Open the file in append mode
    with open(f"{logs_dir}metrix_{column_name}.csv", "a", newline="") as file:
        writer = csv.writer(file)
        # Write the header if it doesn't exist
        if not header_exists:
            writer.writerow(header)
        # Write the data
        writer.writerow(data)


def training_model(is_hyperparameter_tuning: bool = False) -> None:
    df_temp, df_hum = preprocessing()

    # Split data to X and y
    X_temp, y_temp = (
        df_temp.drop(columns=["airTemperature"]),
        df_temp[["airTemperature"]],
    )
    X_hum, y_hum = (
        df_hum.drop(columns=["humidity"]),
        df_hum[["humidity"]],
    )

    # Split data to train and test
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        X_temp, y_temp, test_size=0.3, shuffle=False
    )
    X_train_hum, X_test_hum, y_train_hum, y_test_hum = train_test_split(
        X_hum, y_hum, test_size=0.3, shuffle=False
    )

    if is_hyperparameter_tuning:
        model_temp = create_model_with_hyperparameter_tuning(X_train_temp, y_train_temp)
        model_hum = create_model_with_hyperparameter_tuning(X_train_hum, y_train_hum)
    else:
        model_temp = create_model(X_train_temp, y_train_temp)
        model_hum = create_model(X_train_hum, y_train_hum)

    save_model(model_temp, "airTemperature")
    save_model(model_hum, "humidity")

    save_matrix(model_temp, X_test_temp, y_test_temp, "airTemperature")
    save_matrix(model_hum, X_test_hum, y_test_hum, "humidity")
