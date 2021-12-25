import torch
import os
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from nets import DeepSDF
from dataset import PointCloudDataset
from torch.utils.data import DataLoader, random_split
import sys


PARAMS = {
        "batch_size": 128,
        "data_dir": 'data/preprocessed',
        "epochs": 100,
        "lr": 0.001,
        "load": None,
        "latent_size": 100
}

device = "cuda" if torch.cuda.is_available() else "cpu"

global_data = PointCloudDataset(PARAMS["data_dir"])
prop = 0.7
train_size = int(prop*len(global_data))
train_data, val_data = random_split(global_data, [train_size, len(global_data)-train_size])

train_loader = DataLoader(train_data, batch_size=PARAMS["batch_size"], num_workers=2)
val_loader = DataLoader(val_data, batch_size=PARAMS["batch_size"], num_workers=2)

lat_vecs = torch.nn.Embedding(len(global_data), PARAMS["latent_size"], max_norm=1).to(device)
torch.nn.init.normal_(lat_vecs.weight.data, 0.0, 0.01)

model = DeepSDF(code_dim=PARAMS["latent_size"]).to(device)
model.known_shapes = len(global_data)

PARAMS["model"] = model.name

if PARAMS["load"] is not(None):
    checkpoint = torch.load(LOAD)
    model.load_state_dict(checkpoint["model"])
    lat_vecs.load_state_dict(checkpoint["latent_vecs"])
    model.known_shapes = checkpoint["shapes"]

optimizer = optim.Adam([{"params": model.parameters(), "lr": PARAMS["lr"]},
                        {"params": lat_vecs.parameters(), "lr": PARAMS["lr"]}])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
criterion = torch.nn.L1Loss(reduction="sum")

PARAM_TEXT = ""
for key, value in PARAMS.items():
    PARAM_TEXT += f"{key}: {value}\n"
