import torch
import os
from torch.utils.data import Dataset
import numpy as np
import json

class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = os.path.join(data_path)
        self.models = os.listdir(self.img_path)
    
    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):

        path = os.path.join(self.data_path, self.models[index])

        with open(path) as f:
           data = json.load(f)

           X = torch.tensor(data["x"])
           Y = torch.tensor(data["y"])
           Z = torch.tensor(data["z"])
           Id = torch.tensor(data["id"])
           sdf = torch.tensor(data["sdf"])
        
        return X,Y,Z,Id,sdf
