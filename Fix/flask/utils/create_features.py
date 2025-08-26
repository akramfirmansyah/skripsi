import pandas as pd


def create_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    data = dataframe.copy()
    data["year"] = data.index.year
    data["month"] = data.index.month
    data["day"] = data.index.day
    data["hour"] = data.index.hour
    data["minute"] = data.index.minute
    data["dayofyear"] = data.index.dayofyear
    data["dayofweek"] = data.index.dayofweek

    return data
