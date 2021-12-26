import numpy as np
import trimesh
import argparse
from nets.deepSDF import DeepSDF
import json
from tqdm import tqdm
import torch

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_json', type=str, help="input json")
parser.add_argument('--model', type=str, help="path to model")
parser.add_argument('--lr', type=float, help="path to model", default = 0.0001)
parser.add_argument('--niter', type=int, help="path to model", default=100)

args = parser.parse_args()

apply_reg = False
sigma=10

device = "cuda" if torch.cuda.is_available() else "cpu"

def criterion(x1,x2,latent):
    l1_loss = torch.nn.L1Loss(reduction="sum")
    Delta = 0.1*torch.ones_like(x1)
    X1 = torch.minimum(Delta, torch.maximum(-Delta, x1))
    X2 = torch.minimum(Delta, torch.maximum(-Delta, x2))
    
    if apply_reg:
        reg = (1/sigma) * torch.sum(latent**2)
    else:
        reg = 9
    return l1_loss(X1,X2)+reg

model = DeepSDF().to(device)
checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint["model"])

latent = torch.ones(checkpoint["latent_size"]).normal_(mean=0, std=0.01).to(device)

optimizer = torch.optim.Adam([latent], lr=args.lr)

model.eval()

print(f"Opening {args.input_json}")
with open(args.input_json) as f:
    data = json.load(f)
    points = torch.tensor(data["points"]).to(device)
    sdf = torch.tensor(data["sdf"]).to(device)

    print(f"Looking for best latent vector...")
    for epoch in tqdm(range(args.niter)):
        data = torch.cat([points, latent.repeat(len(points), 1)], dim=1)
        output = model(data)
        loss = criterion(output.T[0], sdf, latent)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    points = points.detach().cpu().numpy()
    sdf = sdf.detach().cpu().numpy()
    final_sdf = output.T[0].detach().cpu().numpy()

    # Color map for plot
    colors = np.zeros(points.shape)
    colors[final_sdf < 0, 2] = 1
    colors[final_sdf > 0, 0] = 1

    pc = trimesh.PointCloud(points, colors)
    pc.show()
