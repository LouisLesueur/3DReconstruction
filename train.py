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
        for data, sdf in tqdm(val_loader):
            data, sdf = data.to(device), sdf.to(device)
            output = model(data)
            validation_loss += criterion(output.T[0], sdf)

    return validation_loss / len(val_loader.dataset)


if __name__ == "__main__":

    iteration = 0
    writer = SummaryWriter()
    command = ''

    writer.add_text("Params", PARAM_TEXT)

    for epoch in range(1, PARAMS["epochs"]):
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
        model.train()

        for batch_idx, (data, sdf) in enumerate(tqdm(train_loader)):
            data, sdf = data.to(device), sdf.to(device)
            output = model(data)

            loss = criterion(output.T[0], sdf)
            
            writer.add_scalar("Train/Loss", loss.data.item(), iteration)

            iteration += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = validation()
        writer.add_scalar("Val/Loss", val_loss, epoch)
        scheduler.step(val_loss)
        model_file = os.path.join("checkpoints", f"{model.name}_{epoch}.pth")
        torch.save({"model": model.state_dict(), "shapes": model.known_shapes}, model_file)
