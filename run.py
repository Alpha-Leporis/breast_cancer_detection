import os
import torch
from src.dataset import get_data_loaders
from src.model import get_vit_model
from src.train import train_model
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    data_dir = 'data/processed'
    train_loader, val_loader = get_data_loaders(data_dir)
    model = get_vit_model(num_classes=3)  # Ensure it matches the number of classes
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    num_epochs = 10
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs)

    # Ensure the models directory exists
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Save the last model in the models directory
    model_path = os.path.join(models_dir, 'vit_breast_cancer_last.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Last model saved to {model_path}')
