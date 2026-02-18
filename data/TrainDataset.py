import pandas as pd
from data.BaseDataset import BaseDataset, get_transform
import os
from PIL import Image
import torch

def make_dataset(img_id, data_root_folder):
    img_folder_path = os.path.join(data_root_folder, img_id)
    imgs = os.listdir(img_folder_path)
    return {
        'close-up': os.path.join(img_folder_path, imgs[0]),
        'dermoscopic': os.path.join(img_folder_path, imgs[1]),
    }
    
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
print(' LABELS: ', LABELS)
lbl_to_idx = {v: i for i, v in enumerate(LABELS)}
class Option:
    pass

class CombinedDataset(BaseDataset):
    def __init__(self, df, device: str):
        BaseDataset.__init__(self, None)
        self.device = device
        self.dataset = df
        opt = Option()
        opt.preprocess = 'crop'
        opt.no_flip = True
        opt.crop_size = 256
        
        self.transform = get_transform(opt)
        self.pre_metadata()
    
    def pre_metadata(self):
        self.dataset['target'] = self.dataset['diagnosis_full'].map(isic_dx_to_abbr)
        # self.dataset.drop(columns=['diagnosis_full'], inplace=True)
        # self.dataset.set_index("isic_id", inplace=True)
        
    def encode_row_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        # Implement your encoding logic here
        # For example, you can use one-hot encoding for categorical variables
        encoded_metadata = pd.get_dummies(metadata)
        return encoded_metadata
        
        

    def __len__(self):
        return len(self.dataset)

    # convert data to current using device before feeding into model
    def pre_device(self, data):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(self.device)
        return data
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        dermoscopic_img = Image.open(row['dermoscopic']).convert('RGB')
        row_meta = row.drop(['isic_id', 'lesion_id', 'close-up', 'dermoscopic', 'image_manipulation', 'copyright_license', 'attribution', 'image_type', 'invasion_thickness_interval'])
        row_meta = self.encode_row_metadata(row_meta)
        
        return self.pre_device({
            "dermoscopic": self.transform(dermoscopic_img),
            # "metadata": row_meta,
            "label": lbl_to_idx[row['target']]
        })

def combine_pandas_datasets(dfs):
    combined_df = pd.merge(dfs[0], dfs[1], on='isic_id', how='inner')
    return combined_df



