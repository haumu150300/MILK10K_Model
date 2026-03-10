from data.TrainDataset import CombinedDataset, make_dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.Efficientnet import Efficientnet
import random
from sklearn.model_selection import train_test_split
from config import Config
import tqdm
from utils import continue_train
random.seed(42)
torch.manual_seed(42)

def train_one_epoch(
    model: nn.Module, dataloader: DataLoader, criterion, optimizer, device: torch.device
):
    model.train()
    running_loss = 0.0
     
    with tqdm.tqdm(total=dataloader.__len__(), desc="Epoch Progress") as pbar:
        for i, batch in enumerate(dataloader):
            # if i >= 100:
            #     break
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            scaler = torch.amp.GradScaler(device.type)
            with torch.amp.autocast(device.type):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            running_loss += loss.item() * inputs.size(0)
            pbar.update(1)
        epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device.type)

    config = Config()
    img_root_folder = config.img_root_folder
    model_saved_path = config.model_saved_path

    train_metadata = pd.read_csv("./MILK10k_Training_Metadata.csv")
    train_metadata = train_metadata[train_metadata["image_type"] == "dermoscopic"]
    # train_supplement = pd.read_csv("./MILK10k_Training_Supplement.csv")
    # all_df = pd.merge(train_metadata, train_supplement, on="isic_id", how="inner")

    all_df = train_metadata
    train_gt_df = pd.read_csv(config.train_gt_path)
    train_gt_df.set_index("lesion_id", inplace=True)
    train_dataset = CombinedDataset(config, all_df, train_gt_df)

    def split_dataframe(df, train_frac=0.8):
        train_df, val_df = train_test_split(df, train_size=train_frac, random_state=42)
        return train_df, val_df

    train_df, val_df = split_dataframe(all_df, train_frac=0.8)
    # limit to 500 rows for testing
    # train_df = train_df.head(50)
    print(train_df.head())
    train_dataset = CombinedDataset(config, train_df, train_gt_df)

    epochs = 500
    batch_size = 6
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # try 2 or 4 in Colab
        pin_memory=True,
    )

    model = Efficientnet(image_size=256, num_classes=11)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
    model.to(device)
    init_step, best_val_loss = continue_train(model, optimizer, config, device)

    # val_df = val_df.head(50)
    val_dataset = CombinedDataset(config, val_df, train_gt_df)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # try 2 or 4 in Colab
        pin_memory=True,
    )
    print("start training...")
    with tqdm.tqdm(initial=init_step, total=epochs, desc="Training Progress") as pbar:
        for epoch in range(init_step, epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = val_data(model, val_loader, criterion, device)
            pbar.update(1)
            print(
                f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
            if epoch > 1 and epoch % 50 == 0:
                best_val_loss = val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": val_loss,
                    },
                    f"{model_saved_path}/best_model_val_loss_{best_val_loss:.4f}_{epoch}.pth",
                )
