import torch
import pandas as pd
from SpaceTx import SpaceTx

_device, _dtype, _cfg = torch.device("cpu"), torch.float32, "./cfg.yml"
spaceTx = SpaceTx(_cfg, device=_device, dtype=_dtype)
_src = torch.randn((5, 60, 4), dtype=_dtype, device=_device)
srcmsk, tgtmsk = torch.randint(0, 2, (5, 60)).bool(), torch.zeros((5, 60), dtype=_dtype)
tgtmsk[:, :30] += 1

srcdate, tgtdate = torch.stack(
    (
        torch.randint(0, 12, size=(5, 60)),
        torch.randint(0, 31, (5, 60)),
    ),
    2,
), torch.stack(
    (
        torch.randint(0, 12, size=(5, 60)),
        torch.randint(0, 31, (5, 60)),
    ),
    2,
)
spaceTx(_src, srcdate, srcmsk, tgtdate, tgtmsk)
print(1)

df = pd.read_excel("./Data/train.xlsx", engine="openpyxl")
print(df.shape)
df = df.dropna()
print(df.shape)