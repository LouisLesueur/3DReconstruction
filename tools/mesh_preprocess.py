import trimesh
import argparse
import os
import time
import json
import torch
import logging
import sys
from datetime import datetime
from pysdf import SDF
import numpy as np

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_data', type=str, help="input meshes directory", default="data/raw")
parser.add_argument('--output_data', type=str, help="output meshes directory", default="data/preprocessed/train")
parser.add_argument('--n_samples', type=int, help="how much points to sample", default=100000)
parser.add_argument('--prop', type=int, help="proportion of surface points", default=0.5)
parser.add_argument('--sigma', type=int, help="noise to add", default=None)
parser.add_argument('--n_shapes', type=int, help="how much points to sample", default=None)
parser.add_argument('--start_from', type=int, help="start from the i-th file (alpha order)", default=0)
parser.add_argument('--logloc', type=str, help="logging location", default="logs/")


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

    now = datetime.now()
    date = now.strftime("%d_%m_%Y-%H_%M_%S")

    N_surface = int(args.prop*args.n_samples)
    N_other = args.n_samples - N_surface

    log_file = os.path.join(args.logloc, f"{date}-mesh_preprocess.log")
    logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ])

    mesh_path = args.input_data
    mesh_list = np.sort(os.listdir(mesh_path))

    start_time = time.time()
    shape_id = args.start_from

    if args.n_shapes is not None:
        limit = min(args.n_shapes, len(mesh_list))
    else:
        limit = len(mesh_list)
    
    logging.info(f"Preprocessing {limit} files over {len(mesh_list)}, with {args.n_samples} samples per mesh")
    logging.info(f"Starting from number {args.start_from}")

    for index in range(shape_id, limit):
        data = {}
        data["points"] = []
        data["sdf"] = []
        mesh_name = mesh_list[index]
        path = os.path.join(mesh_path, mesh_name, 'models', 'model_normalized.obj')
        scene = trimesh.load_mesh(path)

        logging.info(f"Opening {path}...")
        mesh = as_mesh(scene)

        # Sample points and compute SDF
        logging.info(f"Computing sdf")

        surface_points = mesh.sample(N_surface)
        if args.sigma is not None:
            surface_points += args.sigma*np.random.randn(N_surface, 3)
        other_points = np.random.randn(N_other, 3)

        points = np.concatenate((surface_points, other_points))

        try:
            f = SDF(mesh.vertices, mesh.faces)

            sdf = f(points)
            
            X = points.T[0].tolist()
            Y = points.T[1].tolist()
            Z = points.T[2].tolist()
            sdf = sdf.tolist()

            for i in range(len(points)):
                data["points"].append([X[i], Y[i], Z[i]])
                data["sdf"].append(sdf[i])
                data["id"] = index

            json_path = os.path.join(args.output_data, f"model_{index}.json")
            logging.info(f"Saving {json_path}")
            with open(json_path, 'w') as f:
                json_data = json.dump(data, f)
        except:
            logging.error(f"error with mesh {index}")


    end_time = time.time()
    print(f"All meshes done in {end_time-start_time} s")
