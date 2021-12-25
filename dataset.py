import torch
import os
from torch.utils.data import Dataset
import numpy as np
import json
from torch.utils.data import DataLoader

class PointCloudDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.shapes = os.listdir(self.data_path)

    def __len__(self):
        return len(self.shapes)

    def __getitem__(self, index):

        path = os.path.join(self.data_path, self.shapes[index])

        with open(path, 'r') as f:
            self.data = json.load(f)

        point_cloud = torch.tensor(self.data["points"])
        sdf = torch.tensor(self.data["sdf"])
        shape_id = int(self.data["id"])*torch.ones(len(sdf), dtype=int)

        return point_cloud, sdf, shape_id
