import timm

from config import Config
import os
import pandas as pd
import torch
from src.model.EfficientnetAtt import EfficientnetAtt
import torchvision.transforms as transforms
from PIL import Image
import tqdm 
from data.TrainDataset import CombinedDataset
test_dir = '../MILK10k_Test_Input'

submit_df = pd.read_csv("MILK10k_Sample_Submit.csv")
submit_df.set_index("lesion_id", inplace=True)

test_df = pd.read_csv("./MILK10k_Test_Metadata.csv")

        
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device: ', device.index)
    
    checkpoint = torch.load("./chkpts/att_1_loss_custom/best_model_val_loss_0.000733107_40000.pth", map_location=device)
    print("Loaded checkpoint from epoch: ", checkpoint['epoch'], " with train_loss: ", checkpoint['loss'])
    model = EfficientnetAtt()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    config = Config()
    config.crop_size = 480
    config.img_root_folder = test_dir
    test_dataset = CombinedDataset(config, test_df, None, phase="test")
    
    total_lesions = len(test_df) // 2
    for idx in tqdm.tqdm(range(total_lesions)):
        row = test_dataset[idx]
        input1, metadata1 = row["image1"].to(device), row["metadata1"].to(device)
        input2, metadata2 = row["image2"].to(device), row["metadata2"].to(device)
        lesion_id = row["leision_id"]
        
        with torch.no_grad():
            output = model(input1.unsqueeze(0), input2.unsqueeze(0))
            logis = torch.sigmoid(output.squeeze()).clip(1e-8, 1-1e-8)
            submit_df.loc[lesion_id] = logis.detach().cpu().tolist()
        
    # for lesion in tqdm.tqdm(dirs):
    #     logis = []
    #     for img in os.listdir(os.path.join(test_dir, lesion)):
    #         img_path = os.path.join(test_dir, lesion, img)
    #         image = Image.open(img_path).convert("RGB")
    #         input = transform(image)
    #         with torch.no_grad():
    #             output = model(input.unsqueeze(0).to(device))
    #             logis.append(torch.sigmoid(output.squeeze()))
    #     # Average logits for the lesion
    #     avg_logis = torch.stack(logis).mean(dim=0).clip(1e-6, 1-1e-6)
    #     submit_df.loc[lesion] = avg_logis.detach().cpu().tolist()
print('Done!')
submit_df.to_csv("submission_att_custom_loss.csv")


