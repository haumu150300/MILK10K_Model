import timm

from src.loss.ASLWithLogitsLoss import ASLWithLogitsLoss
from data.TrainDataset import CombinedDataset, make_dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.EfficientnetAtt import EfficientnetAtt
import random
from sklearn.model_selection import train_test_split
from config import Config
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CyclicLR
from utils import continue_train


random.seed(42)
torch.manual_seed(42)

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
        outputs = model(input1, input2)
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

def val_data(model: nn.Module, dataloader: DataLoader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.numel()
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device: ", torch.cuda.get_device_name(device) if torch.cuda.is_available() else device.type)

    config = Config()
    config.crop_size = 384
    config.model_saved_path = "./chkpts/att_1_late_fusion"
    img_root_folder = config.img_root_folder
    model_saved_path = config.model_saved_path

    train_metadata = pd.read_csv("./MILK10k_Training_Metadata.csv")
    # train_supplement = pd.read_csv("./MILK10k_Training_Supplement.csv")
    # all_df = pd.merge(train_metadata, train_supplement, on="isic_id", how="inner")

    all_df = train_metadata
    train_gt_df = pd.read_csv(config.train_gt_path)
    train_gt_df.set_index("lesion_id", inplace=True)
    train_dataset = CombinedDataset(config, all_df, train_gt_df)

    # def split_dataframe(df, train_frac=0.8):
    #     train_df, val_df = train_test_split(df, train_size=train_frac, random_state=42)
    #     return train_df, val_df

    # train_df, val_df = split_dataframe(all_df, train_frac=0.95)
    # limit to 500 rows for testing
    # train_df = train_df.head(50)

    total_steps = 100000 + 1
    batch_size = 32
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # try 2 or 4 in Colab
        pin_memory=True,
        persistent_workers=True,
    )
    torch.backends.cudnn.benchmark = True
    model = EfficientnetAtt()

    criterion = ASLWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, weight_decay=1e-6)
    # scheduler = CyclicLR(
    #     optimizer, base_lr=8e-5, max_lr=1e-3, step_size_up=1000, mode="triangular2", cycle_momentum=False
    # )
    scheduler = None
    model.to(device)
    init_step, best_val_loss = 0, 0
    # init_step, best_val_loss = continue_train(
    #     model, optimizer, config, device, saved_checkpoint='best_model_val_loss_0.00197_50000.pth'
    # )

    # val_df = val_df.head(50)
    # val_dataset = CombinedDataset(config, val_df, train_gt_df)
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=2,  # try 2 or 4 in Colab
    #     pin_memory=True,
    # )
    writer = SummaryWriter(log_dir="./runs/att_1_late_fusion")

    print("start training...")
    with tqdm.tqdm(
        initial=init_step, total=total_steps, desc="Training Progress"
    ) as pbar:
        data_iter = iter(train_loader)
        for epoch in range(init_step, total_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            train_loss = train_batch(model, batch, criterion, optimizer, scheduler, device)
            # val_loss, val_accuracy = val_data(model, val_loader, criterion, device)
            val_loss, val_accuracy = 0.0, 0.0
            pbar.update(1)
            pbar.set_postfix({"Train Loss": f"{train_loss:.9f}"})
            writer.add_scalar("Loss/train", train_loss, epoch)
            # writer.add_scalar("Lr", scheduler.get_last_lr()[0], epoch)

            if epoch > 0 and epoch % 5000 == 0: 
                train_dataset.toggle_aug_tf(not train_dataset.need_aug)
                # best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": train_loss,
                    },
                    f"{model_saved_path}/best_model_val_loss_{train_loss:.9f}_{epoch}.pth",
                )
