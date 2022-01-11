import numpy as np
import trimesh
import argparse
from nets.deepSDF import DeepSDF
import json
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from utils import SDFRegLoss
import os
from dataset import ShapeDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_dir', type=str, help="input dir", default = "data/test")
parser.add_argument('--model', type=str, help="path to model")
parser.add_argument('--lr', type=float, help="path to model", default = 0.001)
parser.add_argument('--niter', type=int, help="path to model", default=50)
parser.add_argument('--batch_size', type=int, help="path to model", default=65536)

args = parser.parse_args()

N_SHAPES = len(os.listdir(args.input_dir))

device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = SDFRegLoss(0.1, 0.0001)

checkpoint = torch.load(args.model)
model = DeepSDF(code_dim=checkpoint["latent_size"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

infer_vector = torch.FloatTensor(N_SHAPES, checkpoint["latent_size"]).to(device)
torch.nn.init.xavier_normal_(infer_vector)
infer_vector.requires_grad_()

optimizer = torch.optim.Adam(params=[infer_vector], lr=args.lr)

for shape_id in range(N_SHAPES):
    # Data loaders
    global_data = ShapeDataset(args.input_dir, shape_id)
    global_loader = DataLoader(global_data, batch_size=args.batch_size, num_workers=2)

    print(f"Looking for best latent vector...")
    for epoch in range(args.niter):

        for batch_idx, (points, sdfs) in enumerate(tqdm(global_loader)):
            points, sdfs = points.to(device), sdfs.to(device)

            output = model(infer_vector, points)
            loss = criterion(output.T[0], sdfs, infer_vector[shape_id])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    line = torch.arange(-1,1,0.05)
    grid = torch.cartesian_prod(line,line,line).to(device)

    final_output = model(infer_vector, grid)
    final_sdf = final_output.T[0]

    final_sdf = final_sdf.detach().cpu().numpy()

    # Color map for plot
    grid = grid.detach().cpu().numpy()
    colors = np.zeros(grid.shape)

    red = np.zeros(grid.shape)
    red.T[0] = (final_sdf-np.min(final_sdf))/(np.max(final_sdf)-np.min(final_sdf))
    blue = np.zeros(grid.shape)
    blue.T[2] = 1 - (final_sdf-np.min(final_sdf))/(np.max(final_sdf)-np.min(final_sdf))

    colors = red + blue

    pc = trimesh.PointCloud(grid[final_sdf<0], colors[final_sdf<0])
    pc.show()
