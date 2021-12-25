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
        "batch_size": 2,
        "data_dir": 'data/preprocessed',
        "epochs": 100,
        "lr": 0.0001,
        "load": None,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepSDF().to(device)

PARAMS["model"] = model.name

if PARAMS["load"] is not(None):
    model.load_state_dict(torch.load(LOAD))

optimizer = optim.Adam(model.parameters(), lr=PARAMS["lr"], weight_decay=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

shape_criterion = torch.nn.L1Loss()
criterion = torch.nn.MSELoss()

global_data = PointCloudDataset(PARAMS["data_dir"])
prop = 0.7
train_size = int(prop*len(global_data))
train_data, val_data = random_split(global_data, [train_size, len(global_data)-train_size])

train_loader = DataLoader(train_data, batch_size=PARAMS["batch_size"], num_workers=2)
val_loader = DataLoader(val_data, batch_size=PARAMS["batch_size"], num_workers=2)
# FIND HOW TO SPLIT TRAIN/VAL !

PARAM_TEXT = ""
for key, value in PARAMS.items():
    PARAM_TEXT += f"{key}: {value}\n"
