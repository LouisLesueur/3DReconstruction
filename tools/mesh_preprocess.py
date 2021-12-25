import trimesh
from mesh_to_sdf import sample_sdf_near_surface
import argparse
import os
from tqdm import tqdm
import time
import json
import numpy as np

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_data', type=str, help="input meshes directory", default="data/raw")
parser.add_argument('--output_data', type=str, help="output meshes directory", default="data/preprocessed")
parser.add_argument('--n_samples', type=int, help="how much points to sample", default=50000)
parser.add_argument('--n_shapes', type=int, help="how much points to sample", default=100)
parser.add_argument('--start_from', type=int, help="start from the i-th file (alpha order)", default=None)


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
    mesh_list = np.sort(os.listdir(mesh_path))

    start_time = time.time()

    shape_id = 0

    print(f"Preprocessing {args.n_shapes} files over {len(mesh_list)}")

    start = 0
    if args.start_from is not None:
        start = args.start_from

    for index in tqdm(range(args.n_shapes)):
        data = {}
        data["points"] = []
        data["id"] = []
        data["sdf"] = []
        data["id"] = shape_id
        mesh_name = mesh_list[index]
        path = os.path.join(mesh_path, mesh_name, 'models', 'model_normalized.obj')
        scene = trimesh.load_mesh(path)

        mesh = as_mesh(scene)

        # Sample points and compute SDF
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=args.n_samples)
        X = points.T[0].tolist()
        Y = points.T[1].tolist()
        Z = points.T[2].tolist()
        sdf = sdf.tolist()

        for i in range(len(points)):
            data["points"].append([X[i], Y[i], Z[i]])
            data["sdf"].append(sdf[i])
    
        json_path = os.path.join(args.output_data, f"model_{shape_id}.json")
        with open(json_path, 'w') as f:
            json_data = json.dump(data, f)

        shape_id += 1


    end_time = time.time()
    print(f"All meshes done in {end_time-start_time} s")
