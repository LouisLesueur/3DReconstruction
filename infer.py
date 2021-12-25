import numpy as np
import trimesh
import argparse
from nets import DeepSDF
import json

parser = argparse.ArgumentParser(description="Preprocessing meshes for proper training")

parser.add_argument('--input_json', type=str, help="input json")
parser.add_argument('--model', type=str, help="path to model")

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DeepSDF().to(device)

with open(args.input_json) as f:
    data = json.load(f)
    X = data["x"]
    Y = data["y"]
    Z = data["z"]

    best_Id = 0
    for i in range(model)


    point_cloud = torch.tensor([X,Y,Z, Id]).T
