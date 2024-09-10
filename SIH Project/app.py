from flask import Flask, render_template, Response, redirect, url_for
import cv2
import requests
import thermal_model
import et_model
import irrigation_model
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pickle
from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import os


app = Flask(__name__)

# Initialize camera capture
camera = cv2.VideoCapture(0)

# API Configuration for weather data (example using OpenWeatherMap)
WEATHER_API_KEY = '0cd4907cf8934fdf81f202119242508'
LOCATION = 'patna'

def fetch_weather_data():
    #url = f'http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={LOCATION}&aqi=no'
    #response = requests.get(url)
    #data = response.json()

    # Extracting necessary fields
    temperature = 35
    #data['current']['temp_c']  # Temperature in Celsius
    humidity = 16
    #data['current']['humidity']  # Humidity percentage
    solar_radiation = 5
    #data['current'].get('uv', 0)  # UV index as a proxy for solar radiation
    wind_speed = 17
    #data['current']['wind_kph']  # Wind speed in kph

    return temperature, humidity, solar_radiation, wind_speed
    
def generate_thermal_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            thermal_frame, temp, humidity, evaporation_rate = thermal_model.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', thermal_frame)
            thermal_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + thermal_frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/thermal_surveillance')
def thermal_surveillance():
    return render_template('thermal_surveillance.html')

@app.route('/thermal_feed')
def thermal_feed():
    return Response(generate_thermal_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
def generate_live_video_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/real_surveillance')
def live_video():
    return render_template('real_surveillance.html')  # Assuming this HTML shows live normal video feed

@app.route('/live_video_feed')
def live_video_feed():
    return Response(generate_live_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/et_sensing')
def et_sensing():
    # Fetch weather data
    temperature, humidity, solar_radiation, wind_speed = fetch_weather_data()

    # Calculate ET using the gathered data
    et_value = et_model.calculate_et(temperature, humidity, solar_radiation, wind_speed)

    return render_template('et_sensing.html', et_value=et_value)

@app.route('/irrigation_accounting')
def irrigation_accounting():
    irrigation_model.account_water_usage()
    return redirect(url_for('index'))

crop_model = joblib.load('crop_recommendation_model.pkl')


@app.route('/crop_recommendation')
def crop_recommendation():
    return render_template('crop_recommendation.html')

@app.route('/predict_crop', methods=['POST'])
def predict_crop():
    # Get form data
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])

    # Make a prediction
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predicted_crop = crop_model.predict(features)[0]

    return render_template('crop_recommendation.html', prediction=predicted_crop)

#model
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, 15)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
               'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transforms(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0] * 100

    return class_names[predicted.item()], probability[predicted.item()].item()


@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded file
        image_path = os.path.join('static', file.filename)
        file.save(image_path)

        # Make a prediction
        prediction, confidence = predict_image(image_path)

        return render_template('upload.html', prediction=prediction, confidence=confidence, image_url=image_path)

with open('fertilizer_model.pkl', 'rb') as f:
    fertilizer_model = pickle.load(f)

with open('soil_encoder.pkl', 'rb') as f:
    soil_encoder = pickle.load(f)

with open('crop_encoder.pkl', 'rb') as f:
    crop_encoder = pickle.load(f)

with open('fertilizer_encoder.pkl', 'rb') as f:
    fertilizer_encoder = pickle.load(f)

@app.route('/fertilizer_result', methods=['POST'])
def fertilizer_result():
    # Fetch the form data and process it
    user_input = request.form
    # Extract individual values
    temperature = float(user_input['temperature'])
    humidity = float(user_input['humidity'])
    moisture = float(user_input['moisture'])
    soil_type = soil_encoder.transform([user_input['soil_type']])[0]
    crop_type = crop_encoder.transform([user_input['crop_type']])[0]
    nitrogen = float(user_input['nitrogen'])
    potassium = float(user_input['potassium'])
    phosphorous = float(user_input['phosphorous'])

    # Create input array
    input_data = [[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]]
    
    # Predict the fertilizer
    fertilizer_pred = fertilizer_model.predict(input_data)
    fertilizer_name = fertilizer_encoder.inverse_transform(fertilizer_pred)[0]

    # Render the result in a template
    return render_template('fertilizer_result.html', fertilizer_name=fertilizer_name)


@app.route('/fertilizer_recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        moisture = float(request.form['moisture'])
        soil_type = soil_encoder.transform([request.form['soil_type']])[0]
        crop_type = crop_encoder.transform([request.form['crop_type']])[0]
        nitrogen = float(request.form['nitrogen'])
        potassium = float(request.form['potassium'])
        phosphorous = float(request.form['phosphorous'])

        # Create input array
        input_data = [[temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous]]
        
        # Predict the fertilizer
        fertilizer_pred = fertilizer_model.predict(input_data)
        fertilizer_name = fertilizer_encoder.inverse_transform(fertilizer_pred)[0]

        return render_template('fertilizer_result.html', fertilizer_name=fertilizer_name)
    
    return render_template('fertilizer_recommendation.html')
if __name__ == "__main__":
    app.run(debug=True)
