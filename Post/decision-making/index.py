import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class Aeroponics:
    def __init__(self, airTemperature: float, humidity: float):
        self.airTemperature = airTemperature
        self.humidity = humidity

    def compute(self, raw: bool = False):
        # Initialization of Fuzzy variable
        antecAirTemp = ctrl.Antecedent(np.arange(0, 45 + 1), "Air Temperature")
        antecHumidity = ctrl.Antecedent(np.arange(0, 100 + 1), "Humidity")
        consDelay = ctrl.Consequent(np.arange(10, 60 + 1), "Spraying Delay")

        # Generate fuuzy membership
        antecAirTemp["cool"] = fuzz.trapmf(antecAirTemp.universe, [0, 0, 10, 15])
        antecAirTemp["optimal"] = fuzz.trimf(antecAirTemp.universe, [10, 20, 30])
        antecAirTemp["hot"] = fuzz.trapmf(antecAirTemp.universe, [25, 30, 45, 45])

        antecHumidity["dry"] = fuzz.trapmf(antecHumidity.universe, [0, 0, 70, 80])
        antecHumidity["optimal"] = fuzz.trimf(antecHumidity.universe, [70, 80, 90])
        antecHumidity["moist"] = fuzz.trapmf(antecHumidity.universe, [80, 90, 100, 100])

        consDelay["short"] = fuzz.trapmf(consDelay.universe, [0, 0, 20, 25])
        consDelay["normal"] = fuzz.trapmf(consDelay.universe, [20, 25, 35, 40])
        consDelay["long"] = fuzz.trapmf(consDelay.universe, [35, 40, 60, 60])

        # Define rulesets fuzzy
        rule1 = ctrl.Rule(
            antecAirTemp["hot"] & antecHumidity["dry"], consDelay["short"]
        )
        rule2 = ctrl.Rule(
            antecAirTemp["hot"] & antecHumidity["optimal"], consDelay["short"]
        )
        rule3 = ctrl.Rule(
            antecAirTemp["hot"] & antecHumidity["moist"], consDelay["normal"]
        )
        rule4 = ctrl.Rule(
            antecAirTemp["optimal"] & antecHumidity["dry"], consDelay["short"]
        )
        rule5 = ctrl.Rule(
            antecAirTemp["optimal"] & antecHumidity["optimal"], consDelay["normal"]
        )
        rule6 = ctrl.Rule(
            antecAirTemp["optimal"] & antecHumidity["moist"], consDelay["normal"]
        )
        rule7 = ctrl.Rule(
            antecAirTemp["cool"] & antecHumidity["dry"], consDelay["normal"]
        )
        rule8 = ctrl.Rule(
            antecAirTemp["cool"] & antecHumidity["optimal"], consDelay["normal"]
        )
        rule9 = ctrl.Rule(
            antecAirTemp["cool"] & antecHumidity["moist"], consDelay["long"]
        )

        # Add rulesets to fuzzy
        spraying_ctrl = ctrl.ControlSystem(
            [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9]
        )

        # Calculate result from spraying_ctrl
        spraying_delay = ctrl.ControlSystemSimulation(spraying_ctrl)

        # Input Air Temperature & Humidity to Fuzzy Logic
        spraying_delay.input["Air Temperature"] = self.airTemperature
        spraying_delay.input["Humidity"] = self.humidity

        # Calculate the result
        spraying_delay.compute()

        # Return Number Spraying Delay in minutes
        if raw:
            return spraying_delay.output["Spraying Delay"]
        else:
            return np.round(spraying_delay.output["Spraying Delay"])


if __name__ == "__main__":
    aeroponics = Aeroponics(45, 0)

    print(aeroponics.compute(raw=True))
    print(aeroponics.compute())
