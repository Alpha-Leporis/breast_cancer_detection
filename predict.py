import torch, os
from torchvision import transforms
from PIL import Image
from src.model import get_vit_model  # Import your model function
import matplotlib.pyplot as plt

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_dir = './model'
model_path = os.path.join(model_dir, 'vit_breast_cancer_last.pth')
# Load the state dictionary with map_location to the appropriate device
state_dict = torch.load(model_path, map_location=device)

# Create an instance of the model
model = get_vit_model(num_classes=3)  # Make sure to match the number of classes

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Move the model to the GPU if available
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load and preprocess the input image
image_path = '/content/breast_cancer_detection/data/raw/benign/benign (103).png' # '/img_path/img.png'
image = Image.open(image_path).convert('RGB')  # Convert to RGB mode

# Display the original image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# Preprocess the image
preprocessed_image = transform(image).unsqueeze(0)  # Add batch dimension

# Move the input to the device
preprocessed_image = preprocessed_image.to(device)

# Make predictions
with torch.no_grad():
    outputs = model(preprocessed_image)

# Interpret the prediction
_, predicted = torch.max(outputs, 1)
class_names = ['Benign', 'Malignant', 'Normal']
predicted_class = class_names[predicted.item()]

# Display the preprocessed image
plt.subplot(1, 2, 2)
plt.imshow(preprocessed_image.cpu().squeeze().permute(1, 2, 0))  # Move tensor to CPU for displaying
plt.title('Preprocessed Image')
plt.axis('off')

# Show the images and prediction
plt.suptitle(f'Predicted class: {predicted_class}')
plt.show()