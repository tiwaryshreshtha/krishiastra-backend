import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
import torchvision

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define data transforms
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class names
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
               'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite', 
               'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 
               'Tomato_healthy']

# Load the ResNet50 model
model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Modify the final layer to match the number of classes in your model (15 classes)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# Load the trained model weights
model.load_state_dict(torch.load('best_model.pth', map_location=device))

# Move the model to the device
model = model.to(device)

def predict_image(model, image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        probability = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    return class_names[predicted.item()], probability[predicted.item()].item()

# Test on a single image
test_image_path = 'PlantVillage/Pepper__bell___Bacterial_spot/0022d6b7-d47c-4ee2-ae9a-392a53f48647___JR_B.Spot 8964.JPG'  # Replace with an actual image path
predicted_class, confidence = predict_image(model, test_image_path)

# Display the image with prediction
plt.figure(figsize=(8, 6))
img = Image.open(test_image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f'Predicted: {predicted_class}\nConfidence: {confidence:.2f}%')
plt.show()
