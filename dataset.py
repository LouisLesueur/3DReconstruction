import torch
import os
from torch.utils.data import Dataset
import numpy as np
import json
from torch.utils.data import DataLoader

class ShapeDataset(Dataset):
    def __init__(self, data_path, shape_id):
        self.data_path = data_path
        shapes = os.listdir(self.data_path)
        path = os.path.join(self.data_path, shapes[shape_id])
        with open(path, 'r') as f:
            data = json.load(f)

        self.point_cloud = data["points"]
        self.sdf = data["sdf"]

    def __len__(self):
        return len(self.sdf)

    def __getitem__(self, index):
        return torch.tensor(self.point_cloud[index]), torch.tensor(self.sdf[index])
