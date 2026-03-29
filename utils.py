import os
import torch

def continue_train(model, optimizer, config, device, saved_checkpoint=None):
    if saved_checkpoint is not None:
        model_path = os.path.join(config.model_saved_path, saved_checkpoint)
    else:
        models = sorted(os.listdir(config.model_saved_path))
        models = [m for m in models if m.endswith(".pth")]
        if len(models) == 0:
            return 0, 999
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


def train_batch(model: torch.nn.Module, batch, criterion, optimizer, scheduler, device: torch.device):
    model.train()
    input1 = batch["image1"].to(device)
    metadata1 = batch["metadata1"].to(device)
    input2 = batch["image2"].to(device)
    metadata2 = batch["metadata2"].to(device)
    labels = batch["label"].to(device)
    
    optimizer.zero_grad()
    scaler = torch.amp.GradScaler()
    with torch.amp.autocast(device_type=device.type):
        outputs = model(input1, metadata1, input2, metadata2)
        loss = criterion(outputs, labels)

    if torch.isnan(outputs).any():
        print("Bad output")

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    if scheduler is not None:
        scheduler.step()
    running_loss = loss.item()
    torch.cuda.empty_cache()
    return running_loss