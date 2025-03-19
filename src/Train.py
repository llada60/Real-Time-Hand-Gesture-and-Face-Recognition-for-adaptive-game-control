import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Dataloader import GestureDataset
from Classification_model import ClassificationNet

def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for data, label in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}')

if __name__ == "__main__":
    dataset = GestureDataset('data.pkl')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    classification_model = ClassificationNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classification_model.parameters(), lr=0.001)
    train(classification_model, dataloader, criterion, optimizer, epochs=10)