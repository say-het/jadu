export default function handler(req, res) {
  res.send(`

    import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import Compose, Resize, Normalize
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

class Cifar10MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.pretrained_net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

        for param in self.pretrained_net.features.parameters():
            param.requires_grad = False

        self.pretrained_net.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        return self.pretrained_net(x)

model = Cifar10MobileNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 3

def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

for epoch in range(n_epochs):
    print(f"--- Epoch {epoch+1}/{n_epochs} ---")
    train_loss, train_acc = train_one_epoch(train_loader, model, loss_fn, optimizer)
    print(f"Epoch {epoch+1} Summary: Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
    print("-" * 50)

print("Training finished!")

`);
}
