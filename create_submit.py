from config import Config
import os
import pandas as pd
import torch
from src.model.MyModel import MyCNN
import torchvision.transforms as transforms
from PIL import Image
import tqdm 
test_dir = 'MILK10k_Test_Input/MILK10k_Test_Input'
dirs = [i for i in os.listdir(test_dir) if i.startswith('IL')]

submit_df = pd.read_csv("MILK10k_Sample_Submit.csv")
submit_df.set_index("lesion_id", inplace=True)
 
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print('device: ', device.type)
    
    checkpoint = torch.load("chkpt/best_model_val_loss_1.4967_850.pth", map_location=device)
    model = MyCNN(image_size=256)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for lesion in tqdm.tqdm(dirs):
        logis = []
        for img in os.listdir(os.path.join(test_dir, lesion)):
            img_path = os.path.join(test_dir, lesion, img)
            image = Image.open(img_path).convert("RGB")
            input = transform(image)
            with torch.no_grad():
                output = model(input.unsqueeze(0).to(device))
                logis.append(torch.sigmoid(output.squeeze()))
        # Average logits for the lesion
        avg_logis = torch.stack(logis).mean(dim=0)
        submit_df.loc[lesion] = avg_logis.detach().cpu().tolist()
print('Done!')
submit_df.to_csv("submission.csv")


