import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('Fertilizer Prediction.csv')

# Encode categorical columns
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

le_crop = LabelEncoder()
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

le_fertilizer = LabelEncoder()
df['Fertilizer Name'] = le_fertilizer.fit_transform(df['Fertilizer Name'])

# Split data into features and target
X = df[['Temparature', 'Humidity ', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Potassium', 'Phosphorous']]
y = df['Fertilizer Name']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and encoders
with open('fertilizer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('soil_encoder.pkl', 'wb') as f:
    pickle.dump(le_soil, f)

with open('crop_encoder.pkl', 'wb') as f:
    pickle.dump(le_crop, f)

with open('fertilizer_encoder.pkl', 'wb') as f:
    pickle.dump(le_fertilizer, f)
