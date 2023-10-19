import torch
from torch import nn


class DateEncoding(nn.Module):
    def __init__(self, d_model, device, dtype=torch.float16):
        super(DateEncoding, self).__init__()

        self.device = device
        self.dtype = dtype

        months = torch.arange(1, 12 + 1, device=self.device, dtype=self.dtype)
        days = torch.arange(1, 31 + 1, device=self.device, dtype=self.dtype)
        _2i = torch.arange(0, d_model, step=2, device=self.device, dtype=self.dtype)

        months[::2] = torch.sin(months[::2] / (10000 ** (_2i / d_model)))
        months[1::2] = torch.cos(months[1::2] / (10000 ** (_2i / d_model)))
        days[::2] = torch.sin(days[::2] / (10000 ** (_2i / d_model)))
        days[1::2] = torch.cos(days[1::2] / (10000 ** (_2i / d_model)))

        months = torch.tile(months, (31, 1)).transpose(0, 1)
        days = torch.tile(days, (12, 1))

        self.encoding = months + days

    def forward(self, src, dates) -> torch.tensor:
        encoded = self.encoding[dates[..., 0], dates[..., 1]]
        return src + encoded


class Encoder(nn.Module):
    def __init__(
        self, nlayer, d_model, nhead, dim_feedforward, dropout, device: torch.device
    ):
        super(Encoder, self).__init__()
        self.d_model, self.device = device, d_model

        self.dateEncoding = DateEncoding(self.d_model, device)
        self.encoderLayer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            "relu",
            batch_first=True,
            device=device,
        )

        self.txEncoder = nn.TransformerEncoder(
            self.encoderLayer, nlayer, mask_check=True
        )

    def forward(self, src, dates, msk):
        src = self.dateEncoding(src, dates)
        return self.txEncoder(src, src_key_padding_mask=msk)

