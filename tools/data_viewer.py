import numpy as np
import trimesh
import argparse
import os
from tqdm import tqdm
import time
import json

parser = argparse.ArgumentParser(description="View a preprocessed data")
parser.add_argument('--path', type=str, help="input mesh")
parser.add_argument('--input_data', type=str, help="input meshes directory", default="data/raw")
args = parser.parse_args()


if __name__ == "__main__":
    
    mesh_path = args.path
    mesh_files = args.input_data
    mesh_list = np.sort(os.listdir(mesh_files))

    with open(mesh_path, 'r') as f:
        data = json.load(f)

        print(f"You are viewing shape number {data['id']}")

        points = np.array(data["points"])
        sdf = np.array(data["sdf"])

        # Compute occupancy map
        oc = np.zeros(sdf.shape)
        oc[sdf<0] = 1

        # Color map for plot
        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1

        original = os.path.join(mesh_files, mesh_list[data['id']], 'models', 'model_normalized.obj')
        mesh = trimesh.load(original)
        mesh.show()

        pc = trimesh.PointCloud(points, colors)
        pc.show()
