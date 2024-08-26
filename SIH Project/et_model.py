import numpy as np

def calculate_et(temperature, humidity, solar_radiation, wind_speed):
    # Simplified Penman-Monteith equation for ET calculation
    delta = 4098 * (0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))) / (temperature + 237.3)**2
    gamma = 0.665 * 10**-3 * 101.3  # assuming a standard atmospheric pressure of 101.3 kPa
    eto = (0.408 * delta * (solar_radiation / 100) + gamma * (900 / (temperature + 273)) * wind_speed * (humidity / 100)) / (delta + gamma * (1 + 0.34 * wind_speed))
    
    # Convert ETo from mm/hour to mm/day (24 hours)
    eto_per_day = eto * 24
    return round(eto_per_day, 2)
