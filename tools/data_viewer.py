import numpy as np
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
import argparse
import os
from tqdm import tqdm
import time
import json

parser = argparse.ArgumentParser(description="View a preprocessed data")
parser.add_argument('--path', type=str, help="input mesh")
args = parser.parse_args()


if __name__ == "__main__":
    
    mesh_path = args.path
    with open(mesh_path, 'r') as f:
        data = json.load(f)

        X = np.array(data["x"])
        Y = np.array(data["y"])
        Z = np.array(data["z"])

        points = np.array([X,Y,Z]).T
        sdf = np.array(data["sdf"])

        # Compute occupancy map
        oc = np.zeros(sdf.shape)
        oc[sdf<0] = 1

        # Color map for plot
        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1

        pc = trimesh.PointCloud(points, colors)
        pc.show()
