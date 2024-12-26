from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from skfuzzy import control as ctrl
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split

import csv
import glob
import influxdb_client
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import skfuzzy as fuzz
import warnings
import xgboost as xgb

warnings.filterwarnings("ignore")


class AdaptiveControll:
    def __init__(self) -> None:
        self.logs_dir = "logs/"
        self.model_dir = "model/"
        self.plot_dir = "plots/"

    def FuzzyLogicNutrientPump(
        self, airTemperature: int | float, humidity: int | float
    ) -> int:
        """
        Implements a fuzzy logic-based control system to calculate the spraying delay
        of a nutrient pump based on air temperature and humidity.

        **Input**:
        - `airTemperature` : The current air temperature (in °C) used as input for the fuzzy system.
        - `humidity` : The current humidity (in %) used as input for the fuzzy system.

        **Output**:
        - Returns an integer representing the spraying delay (in minutes), calculated based on the fuzzy logic system.

        **Fuzzy Logic Rules Table**:

        | Air Temperature | Humidity   | Spraying Delay |
        |------------------|------------|----------------|
        | Hot             | Dry        | Short          |
        | Hot             | Optimal    | Short          |
        | Hot             | Moist      | Normal         |
        | Optimal         | Dry        | Short          |
        | Optimal         | Optimal    | Normal         |
        | Optimal         | Moist      | Normal         |
        | Cool            | Dry        | Normal         |
        | Cool            | Optimal    | Normal         |
        | Cool            | Moist      | Long           |

        Example Usage:
            # Example inputs
            adactiveControll = AdaptiveControll(25, 80)

            # Calculate spraying delay
            spraying_delay = self.FuzzyLogicNutrientPump()
            print(f"Recommended Spraying Delay: {spraying_delay} seconds")
        """

        memberAirTemperature = ctrl.Antecedent(np.arange(0, 45 + 1), "Air Temperature")
        memberHumidity = ctrl.Antecedent(np.arange(0, 100 + 1), "Humidity")
        memberDelay = ctrl.Consequent(np.arange(0, 45 + 1), "Spraying Delay")

        # Create Membership
        memberAirTemperature["cool"] = fuzz.trapmf(
            memberAirTemperature.universe, [0, 0, 13, 18]
        )
        memberAirTemperature["optimal"] = fuzz.trapmf(
            memberAirTemperature.universe, [13, 18, 25, 30]
        )
        memberAirTemperature["hot"] = fuzz.trapmf(
            memberAirTemperature.universe, [25, 30, 45, 45]
        )

        memberHumidity["dry"] = fuzz.trapmf(memberHumidity.universe, [0, 0, 65, 70])
        memberHumidity["optimal"] = fuzz.trapmf(
            memberHumidity.universe, [65, 70, 80, 85]
        )
        memberHumidity["moist"] = fuzz.trapmf(
            memberHumidity.universe, [80, 85, 100, 100]
        )

        memberDelay["short"] = fuzz.trapmf(memberDelay.universe, [0, 0, 10, 15])
        memberDelay["normal"] = fuzz.trapmf(memberDelay.universe, [10, 15, 30, 35])
        memberDelay["long"] = fuzz.trapmf(memberDelay.universe, [30, 35, 45, 45])

        # Save membership as png
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)
        pngs = glob.glob(self.plot_dir + "*.png")

        if len(pngs) == 0:
            memberAirTemperature.view()
            plt.title("Air Temperature Membership")
            plt.savefig(f"{self.plot_dir}Air Temperature Membership.png")

            memberHumidity.view()
            plt.title("Humidity Membership")
            plt.savefig(f"{self.plot_dir}Humidity Membership.png")

            memberDelay.view()
            plt.title("Spraying Delay Membership")
            plt.savefig(f"{self.plot_dir}Spraying Delay Membership.png")

        # Define rules
        rule1 = ctrl.Rule(
            memberAirTemperature["hot"] & memberHumidity["dry"], memberDelay["short"]
        )
        rule2 = ctrl.Rule(
            memberAirTemperature["hot"] & memberHumidity["optimal"],
            memberDelay["short"],
        )
        rule3 = ctrl.Rule(
            memberAirTemperature["hot"] & memberHumidity["moist"], memberDelay["normal"]
        )
        rule4 = ctrl.Rule(
            memberAirTemperature["optimal"] & memberHumidity["dry"],
            memberDelay["short"],
        )
        rule5 = ctrl.Rule(
            memberAirTemperature["optimal"] & memberHumidity["optimal"],
            memberDelay["normal"],
        )
        rule6 = ctrl.Rule(
            memberAirTemperature["optimal"] & memberHumidity["moist"],
            memberDelay["normal"],
        )
        rule7 = ctrl.Rule(
            memberAirTemperature["cool"] & memberHumidity["dry"], memberDelay["normal"]
        )
        rule8 = ctrl.Rule(
            memberAirTemperature["cool"] & memberHumidity["optimal"],
            memberDelay["normal"],
        )
        rule9 = ctrl.Rule(
            memberAirTemperature["cool"] & memberHumidity["moist"], memberDelay["long"]
        )

        spraying_ctrl = ctrl.ControlSystem(
            [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
        )

        spraying_delay = ctrl.ControlSystemSimulation(spraying_ctrl)

        spraying_delay.input["Air Temperature"] = airTemperature
        spraying_delay.input["Humidity"] = humidity

        spraying_delay.compute()

        return int(spraying_delay.output["Spraying Delay"])

    def get_data(self, measurement="first", field="airTemperature"):
        print(f"Get data {field} from database...")

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
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index("datetime").asfreq("1min")

        return df

    def preprocessing(self):
        print("Preprocessing...")

        # Get data from InfluxDB
        df_airTemp = self.get_data(measurement="first", field="airTemperature")
        df_humidity = self.get_data(measurement="first", field="humidity")

        # Replace outlier with
        df_airTemp = self.replace_outlier(df_airTemp, "airTemperature")
        df_humidity = self.replace_outlier(df_humidity, "humidity")

        # Fill NaN with mean HH:MM
        df_airTemp = self.fill_na(df_airTemp, "airTemperature")
        df_humidity = self.fill_na(df_humidity, "humidity")

        # Create features
        print("Create features data airTemperature and humidity...")

        df_airTemp = self.create_features(df_airTemp)
        df_humidity = self.create_features(df_humidity)

        return df_airTemp, df_humidity

    def replace_outlier(self, data, column):
        print(f"Replace outlier data {column}...")

        q1, q3 = data[f"{column}"].quantile(0.25), data[f"{column}"].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        data.loc[data[f"{column}"] > upper_limit, f"{column}"] = upper_limit
        data.loc[data[f"{column}"] < lower_limit, f"{column}"] = lower_limit

        return data

    def fill_na(self, data, column):
        print(f"Fill NaN data {column}")

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

    def create_features(self, dataframe):
        data = dataframe.copy()
        data["year"] = data.index.year
        data["month"] = data.index.month
        data["day"] = data.index.day
        data["hour"] = data.index.hour
        data["minute"] = data.index.minute
        data["dayofyear"] = data.index.dayofyear
        data["dayofweek"] = data.index.dayofweek

        return data

    def create_model(self, X_train, y_train):
        print("Create XGBoost Model...")

        model = xgb.XGBRegressor(
            n_estimators=2500,
            learning_rate=0.01,
            max_depth=3,
            subsample=1.0,
            colsample_bytree=1.0,
            gamma=0,
            reg_alpha=1,
            reg_lambda=2,
        )

        model.fit(X_train, y_train)

        return model

    def save_model(self, model: xgb.XGBRegressor, model_name: str):
        print("Save model...")

        filepath = Path(f"{self.model_dir}XGBoost_{model_name}.pkl")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(model, open(filepath, "wb"))

    def save_matrix(self, model, X_test, y_test, column_name: str):
        print("Save training matrix as csv...")

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
        file_exists = os.path.isfile(f"{self.logs_dir}metrix_{column_name}.csv")
        header_exists = False

        if file_exists:
            with open(f"{self.logs_dir}metrix_{column_name}.csv", "r") as file:
                reader = csv.reader(file)
                # Check the first row to see if it's the header
                header_exists = any(row == header for row in reader)

        # Open the file in append mode
        with open(f"{self.logs_dir}metrix_{column_name}.csv", "a", newline="") as file:
            writer = csv.writer(file)
            # Write the header if it doesn't exist
            if not header_exists:
                writer.writerow(header)
            # Write the data
            writer.writerow(data)

    def training_model(self):
        print("Training Model")

        # Retrieving preprocessed data
        df_temp, df_hum = self.preprocessing()

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

        model_temp = self.create_model(X_train_temp, y_train_temp)
        model_hum = self.create_model(X_train_hum, y_train_hum)

        self.save_model(model_temp, "airTemperature")
        self.save_model(model_hum, "humidity")

        self.save_matrix(model_temp, X_test_temp, y_test_temp, "airTemperature")
        self.save_matrix(model_hum, X_test_hum, y_test_hum, "humidity")

    def load_model(self, filepath: str):
        """
        Function for load model
        """
        model = pickle.load(open(filepath, "rb"))

        return model

    def single_predict(self, model, column_name):
        startdate = datetime.now().strftime("%Y-%m-%d, %H:%M:%S")
        date_range = pd.date_range(start=startdate, periods=(60 * 24), freq="min")

        df_future = pd.DataFrame({"datetime": date_range})
        df_future = df_future.set_index("datetime")
        df_future = self.create_features(df_future)

        df_future[f"{column_name}"] = model.predict(df_future)

        return df_future[[f"{column_name}"]]

    def predict(self) -> pd.DataFrame:
        model_temperature = self.load_model(
            f"{self.model_dir}XGBoost_airTemperature.pkl"
        )
        model_humidity = self.load_model(f"{self.model_dir}XGBoost_humidity.pkl")

        df_temperature = self.single_predict(model_temperature, "airTemperature")
        df_humidity = self.single_predict(model_humidity, "humidity")

        df = pd.concat([df_temperature, df_humidity], axis=1)

        return df

    def compute(self):
        df = self.predict()

        index = 0
        num_active = 0
        result = []
        for iter in range(len(df)):
            if iter == index and num_active < 5:
                value = 0

                index += 1
                num_active += 1
            elif num_active == 5:
                value = 1

                delay = self.FuzzyLogicNutrientPump(
                    df.iloc[index]["airTemperature"], df.iloc[index]["humidity"]
                )

                index += delay
                num_active = 0

            result.append(value)

        result = np.array(result)

        df["is_pump_not_active"] = result

        filepath = Path("data/Prediction.csv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath)


# Main fuction in Class Adaptive Controll
# 1. function FuzzyLogicNutrientPump for get value interval spraying delay of Nutrient Pump
# 2. function training_model, for training XGBoost model with new data
# 3. function predict, for predict future
# 4. function compute, for label pump is active or not
