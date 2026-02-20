from config import Config
import os
import pandas as pd
import torch


df = pd.read_csv("./MILK10k_Training_GroundTruth.csv")

row = [i for i in df.iloc[0]]


print(row[1:])
print(torch.tensor(row[1:]))