import torch
import os
from torch.utils.data import Dataset
import numpy as np
import json

class PointCloudDataset(Dataset):
    def __init__(self, input_json):
        with open(input_json, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data["sdf"])

    def __getitem__(self, index):

        point_cloud = torch.tensor(self.data["points"][index])
        sdf = torch.tensor(self.data["sdf"][index])

        return point_cloud,sdf
