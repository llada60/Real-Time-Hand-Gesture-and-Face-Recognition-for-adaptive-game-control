import torch
from torch.utils.data import DataLoader, Dataset

class GestureDataset(Dataset):
    def __init__(self, pkl_file):
        self.pkl_file = torch.load(pkl_file)
        self.labels = self.pkl_file['label']
        self.data = self.pkl_file['data']
    
    def __len__(self):  
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
