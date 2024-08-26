import cv2
import numpy as np

def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a thermal colormap
    thermal_image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    # Simulate temperature, humidity, and evaporation rate calculation
    temperature = np.mean(gray) / 255 * 100  # Example calculation
    humidity = 50 + np.random.rand() * 10  # Example calculation
    evaporation_rate = 2.5 + np.random.rand()  # Example calculation

    # Add the legend with details to the thermal image
    cv2.putText(thermal_image, f'Temp: {temperature:.2f}C', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(thermal_image, f'Humidity: {humidity:.2f}%', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(thermal_image, f'Evaporation: {evaporation_rate:.2f}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return thermal_image, temperature, humidity, evaporation_rate
