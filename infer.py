import numpy as np
import trimesh
import argparse
from nets import DeepSDF
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_json', type=str, help="input json")
parser.add_argument('--model', type=str, help="path to model")
parser.add_argument('--lr', type=float, help="path to model", default = 0.0001)
parser.add_argument('--niter', type=int, help="path to model", default=100)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

criterion = torch.nn.L1Loss()

model = DeepSDF().to(device)
checkpoint = torch.load(LOAD)
model.load_state_dict(checkpoint["model"])

latent = torch.ones(1, checkpoint["latent_size"]).normal_(mean=0, std=0.01)

optimizer = torch.otim.Adam([latent], lr=args.lr)

model.eval()

with open(args.input_json) as f:
    data = json.load(f)
    points = data["points"]
    sdfs = data["sdf"]

    best_Id = 0
    for epoch in tqdm(range(args.niter)):
        loss = 0
        for i, point in enumerate(points):
            point, sdf = torch.tensor(point).to(device), torch.tensor(sdfs[i]).to(device)
            data = torch.cat([point, latent], dim=1)
            output = model(data)
            loss = criterion(output.T[0], sdf)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    points = np.array(points)
    final_sdf = output.T[0].detach().cpu().numpy()

    print(final_sdf)
