import torch
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from nets.deepSDF import DeepSDF
from dataset import PointCloudDataset
from torch.utils.data import DataLoader, random_split
import sys
import logging
from datetime import datetime

# Training parameters
PARAMS = {
        "batch_size": 32,
        "data_dir": 'data/preprocessed',
        "epochs": 100,
        "lr": 0.0001,
        "load": None,
        "latent_size": 100,
        "logloc": "logs",
        "delta": 0.1,
        "sigma": 10,
        "reg": True
}

# Logger
now = datetime.now()
date = now.strftime("%d_%m_%Y-%H_%M_%S")
log_file = os.path.join(PARAMS["logloc"], f"{date}-training.log")
logging.basicConfig(
        encoding='utf-8',
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ])

device = "cuda" if torch.cuda.is_available() else "cpu"

# Data loaders
global_data = PointCloudDataset(PARAMS["data_dir"])
prop = 0.7 # training/val split
train_size = int(prop*len(global_data))
train_data, val_data = random_split(global_data, [train_size, len(global_data)-train_size])
train_loader = DataLoader(train_data, batch_size=PARAMS["batch_size"], num_workers=1)
val_loader = DataLoader(val_data, batch_size=PARAMS["batch_size"], num_workers=1)

# Latent vectors
lat_vecs = torch.nn.Embedding(len(global_data), PARAMS["latent_size"], max_norm=1).to(device)
torch.nn.init.normal_(lat_vecs.weight.data, 0.0, 0.01)

# Model
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


PARAM_TEXT = ""
for key, value in PARAMS.items():
    PARAM_TEXT += f"{key}: {value}\n"

logging.info(f"Starting training, with parameters: \n{PARAM_TEXT}")

def criterion(x1, x2, latent):
    '''
    Clamped L1 loss
    '''
    l1_loss = torch.nn.L1Loss(reduction="sum")
    Delta = PARAMS["delta"]*torch.ones_like(x1)
    X1 = torch.minimum(Delta, torch.maximum(-Delta, x1))
    X2 = torch.minimum(Delta, torch.maximum(-Delta, x2))

    if PARAMS["reg"]:
        reg = (1/PARAMS["sigma"]) * torch.sum(latent**2)
    else:
        reg = 9

    return l1_loss(X1, X2) + reg

def validation():
    model.eval()
    validation_loss = 0

    with torch.no_grad():
        for points, sdfs, indices in tqdm(val_loader):
            points, sdfs, indices = points.to(device), sdfs.to(device), indices.to(device)
            latent_vec = lat_vecs(indices)
            data = torch.cat([points, latent_vec], dim=2)

            for i, point in enumerate(data):
                output = model(point)
                validation_loss += criterion(output.T[0], sdfs[i], lat_vecs.weight.data)

    return validation_loss / len(val_loader)


if __name__ == "__main__":

    iteration = 0
    writer = SummaryWriter()
    command = ''

    writer.add_text("Params", PARAM_TEXT)

    for epoch in range(1, PARAMS["epochs"]):
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
        model.train()

        for batch_idx, (points, sdfs, indices) in enumerate(tqdm(train_loader)):
            points, sdfs, indices = points.to(device), sdfs.to(device), indices.to(device)

            latent_vec = lat_vecs(indices)
            data = torch.cat([points, latent_vec], dim=2)
            
            output = model(data[0])
            loss = criterion(output.T[0], sdfs[0], lat_vecs.weight.data)
            
            writer.add_scalar("Train/Loss", loss.data.item(), iteration)

            iteration += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = validation()
        writer.add_scalar("Val/Loss", val_loss, epoch)
        scheduler.step(val_loss)
        model_file = os.path.join("checkpoints", f"{model.name}_{epoch}.pth")
        torch.save({"model": model.state_dict(),
                    "latent_vecs": lat_vecs.state_dict(),
                    "latent_size": PARAMS["latent_size"],
                    "shapes": model.known_shapes}, model_file)

        logging.info(f"Training, epoch {epoch} finished, train loss: {loss.data.item()}, val loss: {val_loss}")
