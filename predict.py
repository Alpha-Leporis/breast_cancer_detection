import torch
from torchvision import transforms
from PIL import Image

# Load the trained model
model = torch.load('vit_breast_cancer.pth')
model.eval()

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess the input image
image_path = 'data/raw/benign/benign (1).png' # 'path/to/your/image.jpg'
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Make predictions
with torch.no_grad():
    outputs = model(image)

# Interpret the prediction
_, predicted = torch.max(outputs, 1)
class_names = ['Benign', 'Malignant', 'Normal']
predicted_class = class_names[predicted.item()]

print(f'Predicted class: {predicted_class}')
