import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from Dataloader import GestureDataset
from Classification_model import ClassificationNet

def predict(model, dataloader):
    model.eval()
    predictions = []
    for data, label in dataloader:
        output = model(data)
        predictions.append(torch.argmax(output, dim=1))
    return predictions
