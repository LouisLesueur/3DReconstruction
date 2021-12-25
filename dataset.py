import torch
import os
from torch.utils.data import Dataset
import numpy as np
import json

class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = os.path.join(data_path)
        self.models = os.listdir(self.data_path)
    
    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):

        path = os.path.join(self.data_path, self.models[index])

        with open(path) as f:
           data = json.load(f)

           X = data["x"]
           Y = data["y"]
           Z = data["z"]
           Id = [data["id"] for _ in range(len(X))]

           point_cloud = torch.tensor([X,Y,Z, Id]).T
           sdf = torch.tensor(data["sdf"])
        
        return point_cloud,sdf.T
