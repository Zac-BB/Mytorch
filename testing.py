from torchvision import datasets, transforms
import torch
import torch.nn as nn

transform = transforms.Compose([transforms.ToTensor()])

mnist_test = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    mnist_test,
    batch_size=64,
    shuffle=False
)


def TestOperation(model, test_loader, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()  # IMPORTANT: evaluation mode

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradients for testing
        for images, labels in test_loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    return avg_loss, accuracy

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def ConfusionMatrix(model, test_loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.size(0), -1)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.numpy())
            y_pred.extend(preds.numpy())

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

import torch.nn as nn
import torch.nn.functional as F
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(784, 128)  # Input size 784, output size 128
        self.fc2 = nn.Linear(128, 64)   # Input size 128, output size 64
        self.fc3 = nn.Linear(64, 10)    # Input size 64, output size 10 (for classification)

    def forward(self, x):
        # Apply linear transformation and an activation function (e.g., ReLU)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # The final layer often doesn't have an activation
        return x
    
model = SimpleNN()
TestOperation(model, test_loader)
