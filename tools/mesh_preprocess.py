import numpy as np
import trimesh
from mesh_to_sdf import sample_sdf_near_surface
import argparse
import os

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
        assert(isinstance(mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh



if __name__ == "__main__":
    
    mesh_path = args.input_data
    mesh_list = os.listdir(mesh_path)

    for mesh_name in mesh_list:
        path = os.path.join(mesh_path, mesh_name)
        scene = trimesh.load_mesh(path)

        mesh = as_mesh(scene)

        # Sample points and compute SDF
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=args.n_samples)
        
        # Compute occupancy map
        oc = np.zeros(sdf.shape)
        oc[sdf>0] = 1

        # Color map for plot
        colors = np.zeros(points.shape)
        colors[sdf < 0, 2] = 1
        colors[sdf > 0, 0] = 1

        pc = trimesh.PointCloud(points, colors)
        pc.show()
