import torch
from torch import nn


class DateEncoding(nn.Module):
    def __init__(self, d_model, device, dtype=torch.float16):
        super(DateEncoding, self).__init__()

        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        months = (
            torch.arange(1, 12 + 1, device=self.device, dtype=self.dtype)
            .unsqueeze(1)
            .repeat(1, self.d_model)
        )
        days = (
            torch.arange(1, 31 + 1, device=self.device, dtype=self.dtype)
            .unsqueeze(1)
            .repeat(1, self.d_model)
        )
        _2i = torch.arange(0, d_model, step=2, device=self.device, dtype=self.dtype)

        months[:, ::2] = torch.sin(months[:, ::2] / (10000 ** (_2i / d_model)))
        months[:, 1::2] = torch.cos(months[:, 1::2] / (10000 ** (_2i / d_model)))
        days[:, ::2] = torch.sin(days[:, ::2] / (10000 ** (_2i / d_model)))
        days[:, 1::2] = torch.cos(days[:, 1::2] / (10000 ** (_2i / d_model)))

        months = torch.tile(months, (31, 1, 1)).transpose(0, 1)
        days = torch.tile(days, (12, 1, 1))

        self.encoding = months + days
        self.encoding.requires_grad = False

    def forward(self, src, dates) -> torch.tensor:
        encoded = self.encoding[dates[..., 0], dates[..., 1]]
        return src + encoded


class Encoder(nn.Module):
    def __init__(
        self,
        nlayer,
        d_model,
        nhead,
        dim_feedforward,
        dropout,
        device: torch.device,
        dtype=torch.float16,
    ):
        super(Encoder, self).__init__()

        self.d_model, self.device, self.dtype = d_model, device, dtype
        self.dateEncoding = DateEncoding(self.d_model, self.device, self.dtype)

        self.nlayer, self.nhead, self.dim_feedforward, self.dropout = (
            nlayer,
            nhead,
            dim_feedforward,
            dropout,
        )
        self.encoderLayer = nn.TransformerEncoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            "relu",
            batch_first=True,
            device=self.device,
            dtype=self.dtype,
        )

        self.txEncoder = nn.TransformerEncoder(
            self.encoderLayer, self.nlayer, mask_check=True
        )

    def forward(self, src, dates, keyMsk=None):
        src, seqs = self.dateEncoding(src, dates), src.shape[1]
        result = self.txEncoder(src, src_key_padding_mask=keyMsk)
        assert not torch.isnan(result).any()
        return result


_device, _dtype = torch.device("cpu"), torch.float32
encoder = Encoder(4, 4, 4, 64, 0.1, _device, _dtype)
_src = torch.randn((5, 60, 4), dtype=_dtype, device=_device)
_month, _day, _msk = (
    torch.randint(0, 12, size=(5, 60)),
    torch.randint(0, 31, (5, 60)),
    torch.randint(0, 2, (5, 60)).bool(),
)
_date = torch.stack((_month, _day), 2)
result = encoder(_src, _date, _msk)
print(result)
