import numpy as np
import torch
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from nets.deepSDF import DeepSDF
from dataset import ShapeDataset
from torch.utils.data import DataLoader, random_split
import sys
import logging
from datetime import datetime
from utils import SDFRegLoss

# Training parameters
PARAMS = {
        "batch_size": 4096,
        "data_dir": 'data/preprocessed',
        "epochs": 10,
        "lr": 0.001,
        "load": None,
        "latent_size": 256,
        "logloc": "logs",
        "delta": 0.1,
        "sigma": 0.1,
        "n_shapes": 500
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

# Model
model = DeepSDF(n_shapes=PARAMS["n_shapes"], code_dim=PARAMS["latent_size"]).to(device)
PARAMS["model"] = model.name
if PARAMS["load"] is not(None):
    checkpoint = torch.load(LOAD)
    model.load_state_dict(checkpoint["model"])
    model.known_shapes = checkpoint["shapes"]
model.eval()

optimizer = optim.Adam(params=model.parameters(), lr= PARAMS["lr"])

PARAM_TEXT = ""
for key, value in PARAMS.items():
    PARAM_TEXT += f"{key}: {value}\n"

logging.info(f"Starting training, with parameters: \n{PARAM_TEXT}")

criterion = SDFRegLoss(PARAMS["delta"], PARAMS["sigma"])



if __name__ == "__main__":

    iteration = 0
    writer = SummaryWriter()
    command = ''

    writer.add_text("Params", PARAM_TEXT)

    global_epoch = 0

    for shape_id in range(PARAMS["n_shapes"]):
        # Data loaders
        global_data = ShapeDataset(PARAMS["data_dir"], shape_id)
        global_loader = DataLoader(global_data, batch_size=PARAMS["batch_size"], num_workers=1)

        logging.info(f"Starting training on shape number {shape_id}")

        for epoch in range(1, PARAMS["epochs"]):
            writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
            model.train()

            for batch_idx, (points, sdfs) in enumerate(tqdm(global_loader)):
                points, sdfs = points.to(device), sdfs.to(device)

                if shape_id==0 and epoch==1 and batch_idx==0:
                    writer.add_graph(model, input_to_model=(torch.tensor(shape_id), points))

                optimizer.zero_grad()
                output = model(shape_id, points)
                loss = criterion(output.T[0], sdfs, model.codes()[shape_id])
                writer.add_scalar("Train/Loss", loss.item(), global_epoch)
                
                iteration += 1
                loss.backward()
                optimizer.step()

                if shape_id==0:
                    writer.add_histogram(f"latent_vectors_{shape_id}", model.codes()[shape_id], global_step=epoch)

            global_epoch += 1

                #pc_plot = points[-1].unsqueeze(0)
                #final_sdf = output.T[0]
                #colors = torch.zeros_like(pc_plot)
                #colors[0,final_sdf < 0, 2] = 255
                #colors[0,final_sdf > 0, 0] = 255
                #writer.add_mesh("last_mesh", vertices=pc_plot, colors=colors)

        model_file = os.path.join("checkpoints", f"{model.name}_{shape_id}.pth")
        torch.save({"model": model.state_dict(), 
                    "n_shapes": PARAMS["n_shapes"], 
                    "latent_size": PARAMS["latent_size"]}, model_file)
