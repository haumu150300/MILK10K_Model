class Config:
    def __init__(self):
        self.img_root_folder = "/content/drive/MyDrive/Collab_storage/skin_leision/train/MILK10k_Training_Input"
        self.model_saved_path = "/content/drive/MyDrive/Collab_storage/skin_leision/custom_model/"
        # self.img_root_folder = "./MILK10k_Training_Input/MILK10k_Training_Input"
        # self.model_saved_path = "./chkpt"
        
        self.train_metadata_path = "./MILK10k_Training_Metadata.csv"
        self.train_supplement_path = "./MILK10k_Training_Supplement.csv"
        self.train_gt_path = "./MILK10k_Training_GroundTruth.csv"
        self.preprocess = "crop"
        self.no_flip = True
        self.crop_size = 256
