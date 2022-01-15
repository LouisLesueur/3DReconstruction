import numpy as np
import torch
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from nets.autoencoder import ONet
from dataset import ShapeDataset
from torch.utils.data import DataLoader
import sys
import os
import logging
from datetime import datetime
from utils import SDFRegLoss

# Training parameters
PARAMS = {
        "batch_size": 2048,
        "train_dir": 'data/preprocessed/train',
        "val_dir": 'data/preprocessed/val',
        "lr": 0.00001,
        "load": None,
        "latent_size": 256,
        "logloc": "logs",
        "n_points": None,
        "n_shapes": 5,
        "pc_size": 300,
        "epochs": 100,
        "save_frec": 25,
        "occupancy": True
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model = ONet(code_dim=PARAMS["latent_size"]).to(device)
PARAMS["model"] = model.name

if PARAMS["n_shapes"] is None:
    PARAMS["n_shapes"] = len(os.listdir(PARAMS["train_dir"]))

# Logger
now = datetime.now()
date = now.strftime("%d_%m_%Y-%H_%M_%S")
log_file = os.path.join(PARAMS["logloc"], f"{date}-training.log")
logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ])

if PARAMS["load"] is not(None):
    checkpoint = torch.load(PARAMS["load"])
    model.load_state_dict(checkpoint["model"])
        
optimizer = optim.Adam(params=model.parameters(), lr=PARAMS["lr"])

PARAM_TEXT = ""
for key, value in PARAMS.items():
    PARAM_TEXT += f"{key}: {value}\n"

logging.info(f"Starting training, with parameters: \n{PARAM_TEXT}")

if PARAMS["occupancy"]:
    criterion = torch.nn.BCEWithLogitsLoss()
else:
    criterion = SDFRegLoss(PARAMS["delta"], PARAMS["sigma"])


writer = SummaryWriter()


def validate(epoch):

    model.eval()

    with torch.no_grad():

        val_loss = 0
        logging.info(f"Validation step ...")

        for shape_id in tqdm(range(len(os.listdir(PARAMS["val_dir"])))):

            # Data loaders
            global_data = ShapeDataset(PARAMS["val_dir"], shape_id, n_points=PARAMS["n_points"], occupancy=PARAMS["occupancy"])
            global_loader = DataLoader(global_data, batch_size=PARAMS["batch_size"], num_workers=2)

            cloud = global_data.get_cloud(PARAMS["pc_size"]).to(device)


            for batch_idx, (points, occ) in enumerate(global_loader):
                points, occ = points.to(device), occ.to(device)

#                if shape_id==0 and epoch==1 and batch_idx==0:
#                    writer.add_graph(model, input_to_model=(torch.tensor(shape_id), points))

                output = model(cloud, points)
                val_loss += criterion(output.T[0], occ)

    return val_loss/len(os.listdir(PARAMS["val_dir"]))


if __name__ == "__main__":

    command = ''

    writer.add_text("Params", PARAM_TEXT)

    iteration = 0

    for epoch in range(PARAMS["epochs"]):

        model.train()

        for shape_id in range(PARAMS["n_shapes"]):

            # Data loaders
            global_data = ShapeDataset(PARAMS["train_dir"], shape_id, n_points=PARAMS["n_points"], occupancy=PARAMS["occupancy"])
            global_loader = DataLoader(global_data, batch_size=PARAMS["batch_size"], num_workers=2)

            cloud = global_data.get_cloud(PARAMS["pc_size"]).to(device)

            logging.info(f"Starting training on shape number {shape_id}, cloud: {cloud.shape}")

            model.train()

            for batch_idx, (points, occ) in enumerate(tqdm(global_loader)):
                points, occ = points.to(device), occ.to(device)

#                if shape_id==0 and epoch==1 and batch_idx==0:
#                    writer.add_graph(model, input_to_model=(torch.tensor(shape_id), points))

                optimizer.zero_grad()
                output = model(cloud, points)
                loss = criterion(output.T[0], occ)
                loss.backward()
                optimizer.step()

            writer.add_scalar("Train/Loss", loss, iteration)
            iteration += 1

            if iteration % PARAMS["save_frec"] == 0:

                model_file = os.path.join("checkpoints", f"{model.name}_{epoch}.pth")
                torch.save({"model": model.state_dict(), 
                            "n_shapes": PARAMS["n_shapes"], 
                            "latent_size": PARAMS["latent_size"]}, model_file)
                logging.info(f"Saving {model_file}")

        val_loss = validate(epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)
