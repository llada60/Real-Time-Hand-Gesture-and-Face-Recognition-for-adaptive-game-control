import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
    
# Classification Model
class ClassificationNet(nn.Module):
    def __init__(self, input_dim=127, embedding_dim=128, num_classes=8):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, embedding_dim)
        self.relu = nn.ReLU()
        self.residual = nn.Linear(input_dim, embedding_dim)

        self.classify_fc1 = nn.Linear(embedding_dim, 64)
        self.classify_fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        residual = self.residual(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x + residual
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    
