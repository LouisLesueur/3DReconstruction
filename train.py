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
        for X, Y, Z, Id, sdf in tqdm(val_loader):
            X, Y, Z, Id, sdf = X.to(device), Y.to(device), Z.to(device), Id.to(device), sdf.to(device)
            output = []
            for i in range(len(X)):
                output.append(model(X[i], Y[i], Z[i], Id).item())

            output = torch.tensor(output).to(device)
            validation_loss += criterion(output, sdf)

    return validation_loss / len(val_loader.dataset)


if __name__ == "__main__":

    iteration = 0
    writer = SummaryWriter()
    command = ''

    writer.add_text("Params", PARAM_TEXT)

    for epoch in range(1, PARAMS["epochs"]):
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)
        model.train()

        for batch_idx, (X, Y, Z, Id, sdf) in enumerate(tqdm(train_loader)):
            X, Y, Z, Id, sdf = X.to(device), Y.to(device), Z.to(device), Id.to(device), sdf.to(device)

            writer.add_scalar("Train/Loss", loss.data.item(), iteration)
            output = []
            for i in range(len(X)):
                output.append(model(X[i], Y[i], Z[i], Id).item())

            output = torch.tensor(output).to(device)
            loss = criterion(output, sdf)

            iteration += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = validation()
        writer.add_scalar("Val/Loss", val_loss, epoch)
        scheduler.step(val_loss)
        model_file = os.path.join("checkpoints", f"{model.name}_{epoch}.pth")
        torch.save(model.state_dict(), model_file)
