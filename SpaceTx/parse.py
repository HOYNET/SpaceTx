import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class SpaceDataset(Dataset):
    def __init__(self, pth, srcLength, tgtLength, isAE):
        self.data = pd.read_excel(pth, engine="openpyxl").dropna()
        self.srcLength, self.tgtLength, self.isAE = srcLength, tgtLength, isAE
        self.length = self.data.shape[0] - self.srcLength - self.tgtLength

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        srcDates, srcFlux = (
            self.date2np(self.data.iloc[idx : idx + self.srcLength, 0]),
            self.data.iloc[idx : idx + self.srcLength, 1].to_numpy(),
        )
        if self.isAE:
            tgtDates, tgtFlux = (
                self.date2np(self.data.iloc[idx : idx + self.srcLength, 0]),
                self.data.iloc[idx : idx + self.srcLength, 1].to_numpy(),
            )
        else:
            tgtDates, tgtFlux = (
                self.date2np(
                    self.data.iloc[
                        idx + self.srcLength : idx + self.srcLength + self.tgtLength, 0
                    ]
                ),
                self.data.iloc[
                    idx + self.srcLength : idx + self.srcLength + self.tgtLength, 1
                ].to_numpy(),
            )

        return {
            "src": srcFlux,
            "srcDates": srcDates,
            "tgt": tgtFlux,
            "tgtDates": tgtDates,
        }

    def date2np(self, dates):
        months, days = dates.dt.month.to_numpy(dtype=np.int32), dates.dt.day.to_numpy(
            dtype=np.int32
        )
        return np.stack((months, days), axis=-1)
