import numpy as np
import trimesh
import argparse
from nets.autoencoder import ONet
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
parser.add_argument('--batch_size', type=int, help="path to model", default=65536)

args = parser.parse_args()

N_SHAPES = len(os.listdir(args.input_dir))

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(args.model)
model = ONet(code_dim=checkpoint["latent_size"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

for shape_id in range(N_SHAPES):
    # Data loaders
    global_data = ShapeDataset(args.input_dir, shape_id)
    global_loader = DataLoader(global_data, batch_size=args.batch_size, num_workers=2)

    print(f"Looking for best latent vector...")
    cloud = global_data.get_cloud().to(device)
    line = torch.arange(-1,1,0.04)
    grid = torch.cartesian_prod(line,line,line).to(device)
    output = model(cloud, grid)
    
    occ_logits = output.T[0].detach().cpu().numpy()
    occ = np.zeros_like(occ_logits)
    occ[occ_logits>=0.5] = 1

    # Color map for plot
    grid = grid.detach().cpu().numpy()

    pc = trimesh.PointCloud(grid[occ==1])
    pc.show()
