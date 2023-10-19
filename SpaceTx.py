import torch
import pandas as pd
from SpaceTx import SpaceTx, SpaceDataset

dataset = SpaceDataset("./Data/train.xlsx", 10, 10, False)

print(next(iter(dataset)))
