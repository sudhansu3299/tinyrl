#This helps resume the training from already saved weights

#1. model params
#2. optimizer params, including Adams run stats
#3. total env steps

import os
import torch


def save_checkpoint(
        path,
        model,
        optimizer,
        global_step,
        update,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step,
            "update": update,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, device="cpu"):
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) # storing both model and optim

    global_step = checkpoint["global_step"]
    update = checkpoint["update"]

    return global_step, update
