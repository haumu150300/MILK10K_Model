import numpy as np
import pandas as pd
from data.BaseDataset import BaseDataset, get_closeup_transform, get_dermoscopic_transform, get_test_transform
import os
from PIL import Image
import torch
from config import Config
from sklearn.preprocessing import MinMaxScaler

def make_dataset(img_id, data_root_folder):
    img_folder_path = os.path.join(data_root_folder, img_id)
    imgs = os.listdir(img_folder_path)
    return {
        "close-up": os.path.join(img_folder_path, imgs[0]),
        "dermoscopic": os.path.join(img_folder_path, imgs[1]),
    }
def tabular_features(df, idx):
    # 'age_approx', 'sex', 'skin_tone_class', 'site',
    features = [ 'MONET_ulceration_crust', 'MONET_hair', 'MONET_vasculature_vessels', 'MONET_erythema', 'MONET_pigmented', 'MONET_gel_water_drop_fluid_dermoscopy_liquid', 'MONET_skin_markings_pen_ink_purple_pen']
    return torch.tensor([x for x in df.iloc[idx][features].values], dtype=torch.float32)

isic_dx_to_abbr = {
    # AKIEC
    "Solar or actinic keratosis": "AKIEC",
    "Squamous cell carcinomsitu": "AKIEC",
    "Bowen's disease": "AKIEC",
    "Squamous cell carcinoma in situ, Bowens disease": "AKIEC",
    # BCC
    "Basal cell carcinoma": "BCC",
    # BEN_OTH
    "Benign - Other": "BEN_OTH",
    "Benign soft tissue proliferations - Fibro-histiocytic": "BEN_OTH",
    "Benign soft tissue proliferations - Vascular": "BEN_OTH",
    "Collision - Only benign proliferations": "BEN_OTH",
    "Cylindroma": "BEN_OTH",
    "Exogenous": "BEN_OTH",
    "Fibroepithelial polyp": "BEN_OTH",
    "Fibroma, Infundibular or epidermal cyst": "BEN_OTH",
    "Juvenile xanthogranuloma": "BEN_OTH",
    "Mastocytosis": "BEN_OTH",
    "Mucosal melanotic macule": "BEN_OTH",
    "Scar": "BEN_OTH",
    "Sebaceous hyperplasia": "BEN_OTH",
    "Spiradenoma": "BEN_OTH",
    "Supernumerary nipple": "BEN_OTH",
    "Trichilemmal or isthmic-catagen or pilar cyst": "BEN_OTH",
    "Trichoblastoma": "BEN_OTH",
    "Infundibular or epidermal cyst": "BEN_OTH",
    # BKL
    "Clear cell acanthoma": "BKL",
    "Ink-spot lentigo": "BKL",
    "Lichen planus like keratosis": "BKL",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    # DF
    "Dermatofibroma": "DF",
    # INF
    "Inflammatory or infectious diseases": "INF",
    "Molluscum": "INF",
    "Porokeratosis": "INF",
    "Verruca": "INF",
    # MAL_OTH
    "Atypical fibroxanthoma": "MAL_OTH",
    "Collision - At least one malignant proliferation": "MAL_OTH",
    "Kaposi sarcoma": "MAL_OTH",
    "Lymphocytic proliferations - T-Cell/NK": "MAL_OTH",
    "Malignant peripheral nerve sheath tumor": "MAL_OTH",
    "Merkel cell carcinoma": "MAL_OTH",
    # MEL
    "Melanoma Invasive": "MEL",
    "Melanoma in situ": "MEL",
    "Melanoma metastasis": "MEL",
    # NV
    "Blue nevus": "NV",
    "Nevus": "NV",
    "Nevus, Acral": "NV",
    "Nevus, BAP-1 deficient": "NV",
    "Nevus, Balloon cell": "NV",
    "Nevus, Combined": "NV",
    "Nevus, Congenital": "NV",
    "Nevus, Deep penetrating": "NV",
    "Nevus, NOS, Compound": "NV",
    "Nevus, NOS, Dermal": "NV",
    "Nevus, NOS, Junctional": "NV",
    "Nevus, Recurrent or persistent": "NV",
    "Nevus, Reed": "NV",
    "Nevus, Spilus": "NV",
    "Nevus, Spitz": "NV",
    # SCCKA
    "Keratoacanthoma": "SCCKA",
    "Squamous cell carcinoma, Invasive": "SCCKA",
    # VASC
    "Angiokeratoma": "VASC",
    "Arterio-venous malformation": "VASC",
    "Hemangioma": "VASC",
    "Hemangioma, Hobnail": "VASC",
    "Lymphangioma": "VASC",
    "Pyogenic granuloma": "VASC",
}
LABELS = list(set(isic_dx_to_abbr.values()))
lbl_to_idx = {v: i for i, v in enumerate(LABELS)}
idx_to_lbl = {i: v for i, v in enumerate(LABELS)}

 
class CombinedDataset(BaseDataset):
    def __init__(self,opt, all_df: pd.DataFrame, gt_df, phase: str = "train"):
        BaseDataset.__init__(self, opt)
        self.dermos_df = all_df[all_df["image_type"] == "dermoscopic"]
        self.closeup_df = all_df[all_df["image_type"] == "clinical: close-up"]
        self.gt_df = gt_df
        self.opt = opt
        
        self.phase = phase
        self.need_aug = False
        if self.phase == "train":
            self.closeup_transform = get_closeup_transform(opt, is_augment=self.need_aug)
            self.dermos_transform = get_dermoscopic_transform(opt, is_augment=self.need_aug)
        else:
            self.test_transform = get_test_transform(opt)

        # Initialize the scaler
        self.scaler = MinMaxScaler()
        self.pre_metadata()
        
    def toggle_aug_tf(self, need_aug: bool):
        self.closeup_transform = get_closeup_transform(self.opt, is_augment=need_aug)
        self.dermos_transform = get_dermoscopic_transform(self.opt, is_augment=need_aug)
        self.need_aug = need_aug

    def pre_metadata(self):
        self.dermos_df['age_approx'] = self.scaler.fit_transform(self.dermos_df[['age_approx']])
        self.dermos_df['sex'] = self.dermos_df['sex'].map({'male': 0, 'female': 1})
        self.dermos_df['site'] = self.dermos_df['site'].astype('category').cat.codes 
        num_cols = self.dermos_df.select_dtypes(include='number').columns
        self.dermos_df[num_cols] = self.dermos_df[num_cols].fillna(0)
        
        # self.dataset["target"] = self.dataset["diagnosis_full"].map(isic_dx_to_abbr)
        # self.dataset.drop(columns=['diagnosis_full'], inplace=True)
        # self.dataset.set_index("isic_id", inplace=True)

    def encode_row_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        # Implement your encoding logic here
        # For example, you can use one-hot encoding for categorical variables
        encoded_metadata = pd.get_dummies(metadata)
        return encoded_metadata

    def __len__(self):
        return len(self.dermos_df)
    
    def getRowData(self, row, idx):
        leision_id = row.lesion_id
        img_path = os.path.join(self.opt.img_root_folder, leision_id, f'{row["isic_id"]}.jpg')
        img = Image.open(img_path).convert("RGB")
        return img, tabular_features(self.dermos_df, idx), leision_id
    
    def __getitem__(self, idx):
        row = self.dermos_df.iloc[idx]
        row2 = self.closeup_df.iloc[idx]
        
        img1, metadata1, leision_id = self.getRowData(row, idx)
        img2, metadata2, _leision_id1 = self.getRowData(row2, idx)
        if leision_id != _leision_id1:
            print("Error: Lesion IDs do not match for the same index.")
        
        image1, image2 = None, None
        if self.phase == "train":
            image1 = self.dermos_transform(image=np.array(img1))["image"]
            image2 = self.closeup_transform(image=np.array(img2))["image"]
        else:
            image1 = self.test_transform(img1)
            image2 = self.test_transform(img2)
        return {
                "image1": image1,
                "image2": image2,
                "metadata1": metadata1,
                "metadata2": metadata2,
                "leision_id": leision_id,
                "label": None if self.phase == "test" else torch.tensor([i for i in self.gt_df.loc[leision_id].values], dtype=torch.float32),
            }


def combine_pandas_datasets(dfs):
    combined_df = pd.merge(dfs[0], dfs[1], on="isic_id", how="inner")
    return combined_df
