import pandas as pd
from data.TrainDataset import CombinedDataset, idx_to_lbl, make_dataset
from config import Config
from src.model.MyModel import MyCNN
import torch
from utils import continue_train



device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

exit()
config = Config()
config.img_root_folder = './MILK10k_Test_Input/MILK10k_Test_Input'
model = MyCNN(image_size=256)

continue_train(model, None, config, device)

test_df = pd.read_csv("MILK10k_Test_Metadata.csv")
submit_df = pd.read_csv("MILK10k_Sample_Submit.csv")

for idx, row in test_df.iterrows():
    img_id = row.lesion_id
    img_paths = make_dataset(img_id, config.img_root_folder)
    test_df.at[idx, "close-up"] = img_paths["close-up"]
    test_df.at[idx, "dermoscopic"] = img_paths["dermoscopic"]
test_dataset = CombinedDataset(test_df)

for idx in range(len(test_dataset)):
	data = test_dataset[idx]
	dermoscopic = data["dermoscopic"].unsqueeze(0)  # add batch dimension
	logits = model(dermoscopic)
	probs = torch.sigmoid(logits).squeeze().tolist()  # convert to list of probabilities
	for prob_idx in range(len(probs)):
		submit_df.at[idx, idx_to_lbl[prob_idx]] = 1.0 if probs[prob_idx] >= 0.5 else 0.0
  
submit_df.to_csv("MILK10k_Submit.csv", index=False)
# LABEL_COLUMNS = [
# 	"AKIEC",
# 	"BCC",
# 	"BEN_OTH",
# 	"BKL",
# 	"DF",
# 	"INF",
# 	"MAL_OTH",
# 	"MEL",
# 	"NV",
# 	"SCCKA",
# 	"VASC",
# ]

# def build_submit_template(
# 	test_metadata_path: str,
# 	output_path: str,
# ) -> None:
# 	test_df = pd.read_csv(test_metadata_path, usecols=["lesion_id"])
# 	lesion_ids = test_df["lesion_id"].drop_duplicates().sort_values()

# 	submit_df = pd.DataFrame({"lesion_id": lesion_ids})
# 	for col in LABEL_COLUMNS:
# 		submit_df[col] = 0.0

# 	submit_df.to_csv(output_path, index=False)


# if __name__ == "__main__":
# 	build_submit_template(
# 		"MILK10k_Test_Metadata.csv",
# 		"MILK10k_Submit.csv",
# 	)
