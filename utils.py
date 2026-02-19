import os
import torch

def continue_train(model, optimizer, config, device):
    models = os.listdir(config.model_saved_path)
    if len(models) == 0:
        return 0, 0
    model_path = os.path.join(config.model_saved_path, models[0])
    checkpoint = torch.load(model_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from {model_path}")
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Loaded optimizer state from {model_path}")

    start_epoch = checkpoint["epoch"] + 1
    loss = checkpoint["loss"]
    print(f"Resuming training from epoch {start_epoch} with loss {loss:.4f}")
    return start_epoch, loss
