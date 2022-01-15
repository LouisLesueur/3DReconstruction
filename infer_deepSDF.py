import mesh_to_sdf
import numpy as np
import trimesh
import argparse
from nets.decoders import DeepSDF
import json
import mcubes
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils import SDFRegLoss
import os
from dataset import ShapeDataset
from torch.utils.data import DataLoader
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D, ChamferDistancePytorch.fscore

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_dir', type=str, help="input dir", default = "data/test")
parser.add_argument('--model', type=str, help="path to model")
parser.add_argument('--lr', type=float, help="path to model", default = 0.0001)
parser.add_argument('--niter', type=int, help="path to model", default=2)
parser.add_argument('--batch_size', type=int, help="path to model", default=1024)
parser.add_argument('--n_points', type=int, help="path to model", default=10000)

args = parser.parse_args()

N_SHAPES = len(os.listdir(args.input_dir))

device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = SDFRegLoss(0.1, 0.0001)

checkpoint = torch.load(args.model)
model = DeepSDF(code_dim=checkpoint["latent_size"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()
for param in model.parameters():
    param.requires_grad = False

infer_vector = torch.FloatTensor(N_SHAPES, checkpoint["latent_size"]).to(device)
torch.nn.init.xavier_normal_(infer_vector)
infer_vector.requires_grad_()

optimizer = torch.optim.Adam(params=[infer_vector], lr=args.lr)

for shape_id in range(N_SHAPES):
    # Data loaders
    global_data = ShapeDataset(args.input_dir, shape_id)
    global_loader = DataLoader(global_data, batch_size=args.batch_size, num_workers=2)

    cloud = global_data.get_cloud(args.n_points)
    pc = trimesh.PointCloud(cloud)
    pc.show()


    print(f"Looking for best latent vector...")
    for epoch in range(args.niter):

        for batch_idx, (points, sdfs) in enumerate(tqdm(global_loader)):
            points, sdfs = points.to(device), sdfs.to(device)

            output = model(infer_vector, points)
            loss = criterion(output.T[0], sdfs, infer_vector[shape_id])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pts_per_dim = 40
    line = torch.linspace(-1,1,pts_per_dim)
    grid = torch.cartesian_prod(line,line,line).to(device)
    final_output = model(infer_vector, grid)
    final_sdf = final_output.T[0]
    final_sdf = final_sdf.view((pts_per_dim, pts_per_dim, pts_per_dim)).detach().cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(final_sdf, 0.01)

    # Normalisation for Chamfer
    vertices = np.array(vertices)
    vertices = -1 + (((vertices-0)*2)/(pts_per_dim-0))

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    cloud2,_ =  trimesh.sample.sample_surface(mesh, args.n_points)
    pc = trimesh.PointCloud(cloud2)
    pc.show()

    #mesh.show()

    chamLoss = ChamferDistancePytorch.chamfer3D.dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = chamLoss(cloud.float().unsqueeze(0).to(device), torch.tensor(cloud2).float().unsqueeze(0).to(device))
    f_score, precision, recall = ChamferDistancePytorch.fscore.fscore(dist1, dist2)

    print(dist1.mean(), dist2.mean())
    print(f_score, precision, recall)
