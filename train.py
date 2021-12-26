import torch
import os
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import sys
from params import *


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
                validation_loss += criterion(output.T[0], sdfs[i])

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
            loss = criterion(output.T[0], sdfs[0]) / len(train_loader)
            
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

