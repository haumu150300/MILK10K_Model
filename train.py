from data.TrainDataset import CombinedDataset, make_dataset
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.model.MyModel import MyCNN

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs, labels = batch['dermoscopic'], batch['label']
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
            inputs, labels = batch['dermoscopic'], batch['label']
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(dataloader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


img_root_folder = './MILK10k_Training_Input/MILK10k_Training_Input'
train_metadata = pd.read_csv('./MILK10k_Training_Metadata.csv')
train_supplement = pd.read_csv('./MILK10k_Training_Supplement.csv')


new_df = pd.merge(train_metadata, train_supplement, on='isic_id', how='inner')
for idx, row in new_df.iterrows():
    img_id = row.lesion_id
    img_paths = make_dataset(img_id, img_root_folder)
    new_df.at[idx, 'close-up'] = img_paths['close-up']
    new_df.at[idx, 'dermoscopic'] = img_paths['dermoscopic']

#limit to 500 rows for testing
new_df = new_df.head(500)
train_dataset = CombinedDataset(new_df)

epochs = 20
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = MyCNN(image_size=256)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.to(device)

val_df = new_df.sample(frac=0.2, random_state=42)
val_dataset = CombinedDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
best_val_loss = -999
for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = val_data(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        if(val_loss < best_val_loss):
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_val_loss_{best_val_loss:.4f}.pth')
