import numpy as np
import trimesh
import argparse
from nets.autoencoder import ONet
import json
from tqdm import tqdm
import torch
import mcubes
import matplotlib.pyplot as plt
from utils import SDFRegLoss
import os
from dataset import ShapeDataset
from torch.utils.data import DataLoader
#import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D, ChamferDistancePytorch.fscore

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_dir', type=str, help="input dir", default = "data/preprocessed/test")
parser.add_argument('--model', type=str, help="path to model")
parser.add_argument('--batch_size', type=int, help="path to model", default=4096)
parser.add_argument('--n_points', type=int, help="path to model", default=None)

args = parser.parse_args()

N_SHAPES = len(os.listdir(args.input_dir))

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(args.model)
model = ONet(code_dim=checkpoint["latent_size"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

for shape_id in range(N_SHAPES):
    # Data loaders
    global_data = ShapeDataset(args.input_dir, shape_id, occupancy=True)
    global_loader = DataLoader(global_data, batch_size=args.batch_size, num_workers=2)

    input_cloud = global_data.get_cloud().to(device).unsqueeze(0)
    cloud = global_data.get_cloud(args.n_points)
    #pc = trimesh.PointCloud(cloud)
    
    pts_per_dim = 40
    line = torch.linspace(-1,1,pts_per_dim)
    grid = torch.cartesian_prod(line,line,line).to(device)
    output = model(input_cloud, grid)
    occ = 1-torch.sigmoid(output)
    print("pute", occ.min(), occ.max())
    occ = occ.view((pts_per_dim, pts_per_dim, pts_per_dim)).detach().cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(occ, 0.2)

    # Normalisation for Chamfer
    vertices = np.array(vertices)
    vertices = -1 + (((vertices-0)*2)/(pts_per_dim-0))

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    cloud2,_ =  trimesh.sample.sample_surface(mesh, args.n_points)
    
    scene = trimesh.Scene()
    #scene.add_geometry(pc)
    scene.add_geometry(mesh)
    scene.show()

    #chamLoss = ChamferDistancePytorch.chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    #dist1, dist2, idx1, idx2 = chamLoss(cloud.float().unsqueeze(0).to(device), torch.tensor(cloud2).float().unsqueeze(0).to(device))
    #f_score, precision, recall = ChamferDistancePytorch.fscore.fscore(dist1, dist2)

    #print(torch.mean(dist1), torch.mean(dist2))
    #print(precision, recall)

