import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
import xgboost as xgb
from utils.create_features import create_features

from constant.directory import model_dir, predict_dir


def load_model(filepath: str) -> xgb.XGBRegressor:
    model = pickle.load(open(filepath, "rb"))

    return model


def single_predict(
    model: xgb.XGBRegressor, column_name: str, start_date: str
) -> pd.DataFrame:
    startdate = (
        start_date if start_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    date_range = pd.date_range(start=startdate, periods=(60 * 24), freq="min")

    df_future = pd.DataFrame({"datetime": date_range})
    df_future = df_future.set_index("datetime")
    df_future = create_features(df_future)

    df_future[f"{column_name}"] = model.predict(df_future)

    return df_future[[f"{column_name}"]]


def predict(start_date: str) -> pd.DataFrame:
    model_temperature = load_model(f"{model_dir}XGBoost_airTemperature.pkl")
    model_humidity = load_model(f"{model_dir}XGBoost_humidity.pkl")

    df_temperature = single_predict(model_temperature, "airTemperature", start_date)
    df_humidity = single_predict(model_humidity, "humidity", start_date)

    df = pd.concat([df_temperature, df_humidity], axis=1)

    filepath = Path(f"{predict_dir}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=True)

    return df
