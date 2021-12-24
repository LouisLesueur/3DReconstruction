import torch
import os
from tqdm import tqdm
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from nets import DeepSDF
from dataset import PointCloudDataset
from torch.utils.data import DataLoader
import sys


PARAMS = {
        "batch_size": 128,
        "data_dir": 'data/preprcessed',
        "epochs": 100,
        "lr": 0.0001,
        "load": None,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepSDF()

PARAMS["model"] = model.name

if PARAMS["load"] is not(None):
    model.load_state_dict(torch.load(LOAD))

optimizer = optim.Adam(model.parameters(), lr=PARAMS["lr"], weight_decay=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

criterion = # FIND A CRITERION !

global_data = PointCloudDataset(PARAMS["data_dir"])
global_loader = DataLoader(global_data, batch_size=PARAMS["batch_size"], num_workers=2)
# FIND HOW TO SPLIT TRAIN/VAL !

PARAM_TEXT = ""
for key, value in PARAMS.items():
    PARAM_TEXT += f"{key}: {value}\n"
