import numpy as np
import torch
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from nets.decoders import *
from dataset import ShapeDataset
from torch.utils.data import DataLoader
import sys
import os
import logging
from datetime import datetime
from utils import SDFRegLoss

# Training parameters
PARAMS = {
        "batch_size": 4096,
        "data_dir": 'data/preprocessed/train',
        "lr_latent": 0.000001,
        "lr_model": 0.00001,
        "load": None,
        "latent_size": 256,
        "logloc": "logs",
        "n_points": None,
        "delta": 0.1,
        "sigma": 0.1,
        "n_shapes": None,
        "epochs": 100,
        "save_frec": 25
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
model = DeepSDF(code_dim=PARAMS["latent_size"], size=512).to(device)
#model = OccupancyNet(code_dim=PARAMS["latent_size"]).to(device)

PARAMS["model"] = model.name

if PARAMS["n_shapes"] is None:
    PARAMS["n_shapes"] = len(os.listdir(PARAMS["data_dir"]))

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
    checkpoint = torch.load(LOAD)
    model.load_state_dict(checkpoint["model"])
    model.known_shapes = checkpoint["shapes"]
        
latent_vectors = torch.FloatTensor(PARAMS["n_shapes"], PARAMS["latent_size"]).to(device)
torch.nn.init.uniform_(latent_vectors)
latent_vectors.requires_grad_()


optimizer = optim.Adam([{"params": model.parameters(), "lr": PARAMS["lr_latent"]},
                        {"params": latent_vectors}], lr=PARAMS["lr_model"])

PARAM_TEXT = ""
for key, value in PARAMS.items():
    PARAM_TEXT += f"{key}: {value}\n"

logging.info(f"Starting training, with parameters: \n{PARAM_TEXT}")

criterion = SDFRegLoss(PARAMS["delta"], PARAMS["sigma"])

if __name__ == "__main__":

    writer = SummaryWriter()
    command = ''

    writer.add_text("Params", PARAM_TEXT)

    iteration = 0
    global_iteration = 0

    for epoch in range(PARAMS["epochs"]):

        for shape_id in range(PARAMS["n_shapes"]):


            # Data loaders
            global_data = ShapeDataset(PARAMS["data_dir"], shape_id, n_points=PARAMS["n_points"])
            global_loader = DataLoader(global_data, batch_size=PARAMS["batch_size"], num_workers=2)

            logging.info(f"Starting training on shape number {shape_id}")

            model.train()

            shape_loss = 0

            for batch_idx, (points, sdfs) in enumerate(tqdm(global_loader)):
                points, sdfs = points.to(device), sdfs.to(device)

#                if shape_id==0 and epoch==1 and batch_idx==0:
#                    writer.add_graph(model, input_to_model=(torch.tensor(shape_id), points))
                optimizer.zero_grad()

                output = model(latent_vectors[shape_id], points)
                loss = criterion(output.T[0], sdfs, latent_vectors[shape_id])
                
                writer.add_scalar("Train/L1Loss", criterion.shape_loss, iteration)
                writer.add_scalar("Train/RegLoss", criterion.reg_loss, iteration)
                writer.add_scalar("Train/Loss", loss, iteration)
                iteration += 1
                
                loss.backward()

                shape_loss += loss

                optimizer.step()

            writer.add_scalar("Train/ShapeLoss", shape_loss/len(global_loader), global_iteration)

            if shape_id==0:
                writer.add_histogram(f"latent_vectors_{shape_id}", latent_vectors[shape_id], global_step=epoch)
        

            if global_iteration % PARAMS["save_frec"] == 0:
                model_file = os.path.join("checkpoints", f"{model.name}_{epoch}.pth")
                torch.save({"model": model.state_dict(), 
                            "n_shapes": PARAMS["n_shapes"], 
                            "latent_size": PARAMS["latent_size"]}, model_file)
                logging.info(f"Saving {model_file}")

            global_iteration += 1
