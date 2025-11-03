export default function handler(req, res) {
  res.send(`
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!unzip mnist_train.csv.zip

def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path, engine='python')
    data = df.iloc[:, 1:].values.astype('float32')
    labels = df.iloc[:, 0].values.astype('int64')
    data = data / 255.0
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    return TensorDataset(data, labels)

data_frame_train = load_csv_dataset('/content/mnist_test.csv')
data_frame_test = load_csv_dataset('/content/mnist_test.csv')

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 10, figsize=(15, 4))
found_digits = {}
for img, label in data_frame_train:
    digit = label.item()
    if digit not in found_digits:
        found_digits[digit] = img.reshape(28, 28).numpy()
        axes[digit].imshow(found_digits[digit], cmap='gray')
        axes[digit].set_title(f"Label: {digit}")
        axes[digit].axis('off')
    if len(found_digits) == 10:
        break
plt.show()

batch_size = 64
lr = 0.001
epochs = 50

from torch.utils.data import DataLoader,TensorDataset

train_loader = DataLoader(
    data_frame_train,
    batch_size=batch_size,
    shuffle=True
    )
test_loader = DataLoader(
    data_frame_test,
    batch_size=batch_size,
    shuffle=False
    )

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.dropout=nn.Dropout(0.5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2=nn.Dropout(0.5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

model = CNN().to(device)
sum=0;
for p in model.parameters():
  sum+=p.numel()
print(sum)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

epochs = 20
train_losses = []
test_accuracies = []

for epoch in range(epochs):

    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%")
`);
}
