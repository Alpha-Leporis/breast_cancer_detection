import torch
import torch.optim as optim
from src.dataset import get_data_loaders
from src.model import get_vit_model
from src.evaluate import validate_model
import torch.nn as nn

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total

        val_loss, val_acc = validate_model(model, val_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

if __name__ == "__main__":
    data_dir = 'data/processed'
    train_loader, val_loader = get_data_loaders(data_dir)
    model = get_vit_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10)
