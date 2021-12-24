import numpy as np
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
import argparse
import os
from tqdm import tqdm
import time
import json

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_data', type=str, help="input meshes directory", default="../data/raw")
parser.add_argument('--output_data', type=str, help="output meshes directory", default="../data/preprcessed")
parser.add_argument('--n_samples', type=int, help="how much points to sample", default=500000)

args = parser.parse_args()


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh



if __name__ == "__main__":
    
    mesh_path = args.input_data
    mesh_list = os.listdir(mesh_path)

    start_time = time.time()

    shape_id = 0

    for mesh_name in tqdm(mesh_list):
        path = os.path.join(mesh_path, mesh_name, 'models', 'model_normalized.obj')
        scene = trimesh.load_mesh(path)

        mesh = as_mesh(scene)
        json_path = os.path.join(args.output_data, f"model_{shape_id}.json")

        # Sample points and compute SDF
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=args.n_samples)
        
        data = {}
        data["id"] = shape_id
        data["x"] = points.T[0].tolist()
        data["y"] = points.T[1].tolist()
        data["z"] = points.T[2].tolist()
        data["sdf"] = sdf.tolist()

        with open(json_path, 'w') as f:
            json_data = json.dump(data, f)

        shape_id += 1

    end_time = time.time()
    print(f"All meshes done in {end_time-start_time} s")
