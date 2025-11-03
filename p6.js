export default function handler(req, res) {
  res.send(`

import cv2
from google.colab.patches import cv2_imshow
import torch

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class ImageFeatureExtractor(nn.Module):
    def __init__(self,input):
        super(ImageFeatureExtractor, self).__init__()
        self.l1=nn.Linear(input,20)
        self.a1=nn.ReLU()
        self.l2=nn.Linear(20,7)
        self.a2=nn.ReLU()
        self.l3=nn.Linear(7,5)
        self.a3=nn.ReLU()
        self.l4 = torch.nn.Linear(5,1)
        self.a4 = torch.nn.Sigmoid()

    def forward(self,X):
        X=self.l1(X)
        X=self.a1(X)
        X=self.l2(X)
        X=self.a2(X)
        X=self.l3(X)
        X=self.a3(X)
        X=self.l4(X)
        X=self.a4(X)
        return X

from torchsummary import summary
model1=ImageFeatureExtractor(12288)
summary(model1, input_size=(12288,))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)
print(model1)

import h5py
import numpy as np

def load_dataset():
    train = h5py.File("/content/train_catvsnoncat.h5", "r")
    test = h5py.File("/content/test_catvsnoncat.h5","r")

    train_x = torch.tensor(train['train_set_x'], dtype=torch.float32)
    train_y = torch.tensor(train['train_set_y'], dtype=torch.float32).reshape(1,-1)

    test_x = torch.tensor(test['test_set_x'], dtype=torch.float32)
    test_y = torch.tensor(test['test_set_y'], dtype=torch.float32).reshape(1, -1)

    original_x = np.array(train['train_set_x'])
    return train_x, train_y, test_x, test_y, original_x

X, Y, test_x, test_y, o_x = load_dataset()
X, Y, test_x, test_y, o_x = load_dataset()
print(X.shape)
print(Y.shape)
print(test_x.shape)
print(test_y.shape)

train_x = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
train_x = train_x/255.

nf = train_x.shape[1]
model=ImageFeatureExtractor(nf)

optimizer = optim.Adam(model.parameters(), lr=0.0075)

import torch.nn.functional as F

for i in range(3000):
  optimizer.zero_grad()
  ypred=model(train_x)
  cost = F.binary_cross_entropy(ypred,Y.T)
  cost.backward()
  optimizer.step()
  if i % 100 == 0:
        print(f"iteration {i} and cost {cost}")

def classify(preds):
    preds[preds > 0.5] = 1
    preds[preds <= 0.5] = 0
    return preds

import matplotlib.pyplot as plt
def show_dataset(x, y, rows=6, cols=6):
    fig, axs = plt.subplots(rows, cols, figsize=(7,7))
    for i, ax in enumerate(axs.flat):
        ax.imshow(x[i])
        ax.text(0.5, -0.1, f"is cat: {y[0][i]}", va='bottom')
        ax.axis("off")

preds = classify(ypred)
show_dataset(o_x, preds.T)

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    accuracy = correct / labels.size(1) * 100
    return accuracy

train_predictions = classify(model(train_x))
train_accuracy = calculate_accuracy(train_predictions.T, Y)
print(f"Training Accuracy: {train_accuracy:.2f}%")

test_x_flattened = test_x.reshape(test_x.shape[0], -1)
test_predictions = classify(model(test_x_flattened))
test_accuracy = calculate_accuracy(test_predictions.T, test_y)
print(f"Testing Accuracy: {test_accuracy:.2f}%")

tes=model(test_x.reshape(test_x.shape[0],test_x.shape[1]*test_x.shape[2]*test_x.shape[3]))
ped=classify(tes)
show_dataset(o_x,ped.T)

`);
}
