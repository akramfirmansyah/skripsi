import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import influxdb_client
from dotenv import load_dotenv
import os
import pandas as pd


class AdaptiveControll:
    def __init__(self, airTemperature: int | float, humidity: int | float) -> None:
        self.airTemperature = airTemperature
        self.humidity = humidity

    def FuzzyLogicNutrientPump(self):
        memberAirTemperature = ctrl.Antecedent(np.arange(0, 45 + 1), "Air Temperature")
        memberHumidity = ctrl.Antecedent(np.arange(0, 100 + 1), "Humidity")
        memberDelay = ctrl.Consequent(np.arange(10, 45 + 1), "Spraying Delay")

        memberAirTemperature["cool"] = fuzz.trapmf(
            memberAirTemperature.universe, [0, 0, 10, 15]
        )
        memberAirTemperature["optimal"] = fuzz.trimf(
            memberAirTemperature.universe, [11, 18, 26]
        )
        memberAirTemperature["hot"] = fuzz.trapmf(
            memberAirTemperature.universe, [22, 30, 45, 45]
        )

        memberHumidity["dry"] = fuzz.trapmf(memberHumidity.universe, [0, 0, 70, 80])
        memberHumidity["optimal"] = fuzz.trimf(
            memberHumidity.universe, [70, (90 + 70) / 2, 90]
        )
        memberHumidity["moist"] = fuzz.trapmf(
            memberHumidity.universe, [80, 90, 100, 100]
        )

        memberDelay["short"] = fuzz.trapmf(memberDelay.universe, [0, 0, 20, 25])
        memberDelay["normal"] = fuzz.trapmf(memberDelay.universe, [20, 25, 35, 40])
        memberDelay["long"] = fuzz.trapmf(memberDelay.universe, [35, 40, 45, 45])

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

        spraying_delay.input["Air Temperature"] = self.airTemperature
        spraying_delay.input["Humidity"] = self.humidity

        spraying_delay.compute()

        return int(spraying_delay.output["Spraying Delay"])

    def get_data(self, measurement="first", field="airTemperature"):
        load_dotenv()

        token = os.getenv("TOKEN")
        url = os.getenv("URL")
        org = os.getenv("ORG")

        client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)

        query_api = client.query_api()

        query = f"""from(bucket: "{os.getenv("BUCKET")}")
                |> range(start: {os.getenv("START_DATE")})
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
        # Get data from InfluxDB
        df_airTemp = self.get_data(measurement="first", field="airTemperature")
        df_humidity = self.get_data(measurement="first", field="humidity")

        # Replace outlier with
        df_airTemp = self.replace_outlier(df_airTemp, "airTemperature")
        df_humidity = self.replace_outlier(df_humidity, "humidity")

        # Fill NaN with mean HH:MM
        df_airTemp = self.fill_na(df_airTemp, "airTemperature")
        df_humidity = self.fill_na(df_humidity, "humidity")

        return df_airTemp, df_humidity

    def replace_outlier(self, data, column):
        q1, q3 = data[f"{column}"].quantile(0.25), data[f"{column}"].quantile(0.75)
        iqr = q3 - q1
        lower_limit = q1 - 1.5 * iqr
        upper_limit = q3 + 1.5 * iqr

        data[f"{column}"].loc[data[f"{column}"] > upper_limit] = upper_limit
        data[f"{column}"].loc[data[f"{column}"] < lower_limit] = lower_limit

        return data

    def fill_na(self, data, column):
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

    def training_model(self):
        return self


airTemperature = np.random.randint(0, 45)
humidity = np.random.randint(0, 100)
adaptiveControl = AdaptiveControll(airTemperature, humidity)
