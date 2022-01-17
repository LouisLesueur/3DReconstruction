import torch
import os
from torch.utils.data import Dataset
import numpy as np
import json

class ShapeDataset(Dataset):
    '''Dataset to store a shape, SDF or occupancy

    Args:
        data_path: path to a directory of JSON file containing point cloud and sdf
        shape_id: shape id to open in the directory
        occupancy: the 3d representation
        n_points: points to subsample
    '''

    def __init__(self, data_path, shape_id, occupancy=False, n_points=None):
        self.data_path = data_path
        shapes = os.listdir(self.data_path)
        path = os.path.join(self.data_path, shapes[shape_id])
        with open(path, 'r') as f:
            data = json.load(f)
        self.path = path

        self.point_cloud = torch.tensor(data["points"])
        self.sdf = torch.tensor(data["sdf"])

        if n_points is not None:
            indices = torch.randperm(len(self.sdf))[:n_points]
            self.sdf = self.sdf[indices]
            self.point_cloud = self.point_cloud[indices]

        self.occupancy = occupancy

        if occupancy:
            # Occupancy map: 1 if inside, 0 if outside
            occ = torch.zeros_like(self.sdf)
            occ[self.sdf<=0.01] = 1
            self.sdf = occ.float()

    def split(self, x):
        self.sdf = self.sdf[self.point_cloud.T[0]<x]
        self.point_cloud = self.point_cloud[self.point_cloud.T[0]<x]

    def __len__(self):
        return len(self.sdf)

    def __getitem__(self, index):
        return self.point_cloud[index], self.sdf[index]

    def get_cloud(self, n_pts=300):

        if self.occupancy:
            cloud = self.point_cloud[self.sdf==1]
        else:
            cloud = self.point_cloud[self.sdf <= 0.01]

        indices = torch.randperm(len(cloud))[:n_pts]

        return cloud[indices]
