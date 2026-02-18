from data.TrainDataset import CombinedDataset, make_dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.MyModel import MyCNN
import random


random.seed(42)
torch.manual_seed(42)

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels = batch['dermoscopic'].to(device), batch['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def val_data(model: nn.Module, dataloader: DataLoader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch['dermoscopic'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

img_root_folder = './MILK10k_Training_Input/MILK10k_Training_Input'
train_metadata = pd.read_csv('./MILK10k_Training_Metadata.csv')
train_supplement = pd.read_csv('./MILK10k_Training_Supplement.csv')


all_df = pd.merge(train_metadata, train_supplement, on='isic_id', how='inner')
for idx, row in all_df.iterrows():
    img_id = row.lesion_id
    img_paths = make_dataset(img_id, img_root_folder)
    all_df.at[idx, 'close-up'] = img_paths['close-up']
    all_df.at[idx, 'dermoscopic'] = img_paths['dermoscopic']

def split_dataframe(df, train_frac=0.8):
    train_size = int(len(df) * train_frac)
    train_df = df.sample(n=train_size, random_state=42)
    val_df = df.drop(train_df.index)
    return train_df, val_df

train_df, val_df = split_dataframe(all_df, train_frac=0.8)
#limit to 500 rows for testing
train_df = train_df.head(500)
train_dataset = CombinedDataset(train_df)

epochs = 1000
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = MyCNN(image_size=256)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)

val_dataset = CombinedDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
best_val_loss = 999
model_saved_path = './'
for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = val_data(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        if(epoch > 100 and val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }, f'{model_saved_path}best_model_val_loss_{best_val_loss:.4f}.pth')
