import skfuzzy as fuzz
import numpy as np


def CalculateSprayingDelay(airTemperature: int | float, humidity: int | float) -> int:
    if airTemperature < 0:
        airTemperature = 0
    elif airTemperature > 45:
        airTemperature = 45

    if humidity < 0:
        humidity = 0
    elif humidity > 100:
        humidity = 100

    # Defining membership scope
    memberAirTemperature = np.arange(0, 45 + 1)
    memberHumidity = np.arange(0, 100 + 1)
    memberDelay = np.arange(5, 45 + 1)

    # Create Membership Air Temperature
    memberAirTemperature_cool = fuzz.trapmf(memberAirTemperature, [0, 0, 13, 18])
    memberAirTemperature_optimal = fuzz.trapmf(memberAirTemperature, [13, 18, 25, 30])
    memberAirTemperature_hot = fuzz.trapmf(memberAirTemperature, [25, 30, 45, 45])

    # Create Membership Humidity
    memberHumidity_dry = fuzz.trapmf(memberHumidity, [0, 0, 55, 60])
    memberHumidity_optimal = fuzz.trapmf(memberHumidity, [55, 60, 80, 85])
    memberHumidity_moist = fuzz.trapmf(memberHumidity, [80, 85, 100, 100])

    # Create Membership Spraying Delay
    memberDelay_short = fuzz.trapmf(memberDelay, [5, 5, 25, 30])
    memberDelay_normal = fuzz.trimf(memberDelay, [25, 30, 35])
    memberDelay_long = fuzz.trapmf(memberDelay, [30, 35, 45, 45])

    # Calculating the degree of membership Air Temperature
    memberAirTemperature_cool_degree = fuzz.interp_membership(
        memberAirTemperature, memberAirTemperature_cool, airTemperature
    )
    memberAirTemperature_optimal_degree = fuzz.interp_membership(
        memberAirTemperature, memberAirTemperature_optimal, airTemperature
    )
    memberAirTemperature_hot_degree = fuzz.interp_membership(
        memberAirTemperature, memberAirTemperature_hot, airTemperature
    )

    # Calculating the degree of membership Humidity
    memberHumidity_dry_degree = fuzz.interp_membership(
        memberHumidity, memberHumidity_dry, humidity
    )
    memberHumidity_optimal_degree = fuzz.interp_membership(
        memberHumidity, memberHumidity_optimal, humidity
    )
    memberHumidity_moist_degree = fuzz.interp_membership(
        memberHumidity, memberHumidity_moist, humidity
    )

    # Create rules
    rule1 = np.fmin(memberAirTemperature_hot_degree, memberHumidity_dry_degree)
    rule2 = np.fmin(memberAirTemperature_hot_degree, memberHumidity_optimal_degree)
    rule3 = np.fmin(memberAirTemperature_hot_degree, memberHumidity_moist_degree)
    rule4 = np.fmin(memberAirTemperature_optimal_degree, memberHumidity_dry_degree)
    rule5 = np.fmin(memberAirTemperature_optimal_degree, memberHumidity_optimal_degree)
    rule6 = np.fmin(memberAirTemperature_optimal_degree, memberHumidity_moist_degree)
    rule7 = np.fmin(memberAirTemperature_cool_degree, memberHumidity_dry_degree)
    rule8 = np.fmin(memberAirTemperature_cool_degree, memberHumidity_optimal_degree)
    rule9 = np.fmin(memberAirTemperature_cool_degree, memberHumidity_moist_degree)

    # Mamdani Inference System
    delay_short = np.fmax(rule1, np.fmax(rule2, rule4))  # Rule 1, 2, 4
    delay_normal = np.fmax(rule3, np.fmax(rule5, rule7))  # Rule 3, 5, 7
    delay_long = np.fmax(rule6, np.fmax(rule8, rule9))  # Rule 6, 8, 9

    activation_short = np.fmin(delay_short, memberDelay_short)
    activation_normal = np.fmin(delay_normal, memberDelay_normal)
    activation_long = np.fmin(delay_long, memberDelay_long)

    # Aggregated
    aggregated = np.fmax(activation_short, np.fmax(activation_normal, activation_long))

    # Calculate spraying delay (Defuzzification with Centroid Method)
    spraying_delay = fuzz.defuzz(memberDelay, aggregated, "centroid")

    # Return result as Decimal
    return int(spraying_delay)
