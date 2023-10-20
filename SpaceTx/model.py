import torch
from torch import nn
import yaml


class DateEncoding(nn.Module):
    def __init__(self, d_model, device, dtype=torch.float32):
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
        encoded = self.encoding[dates[..., 0] - 1, dates[..., 1] - 1]
        return src + encoded


class Encoder(nn.Module):
    def __init__(self, cfg, device=torch.device, dtype=torch.float32):
        super(Encoder, self).__init__()
        self.d_model, self.device, self.dtype = cfg["d_model"], device, dtype
        self.dateEncoding = DateEncoding(self.d_model, self.device, self.dtype)

        self.nlayer, self.nhead, self.dim_feedforward, self.dropout = (
            cfg["nlayer"],
            cfg["nhead"],
            cfg["dim_feedforward"],
            cfg["dropout"],
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
        src, srcMsk = self.dateEncoding(src, dates), self.create_causal_mask(
            src.shape[1]
        )
        result = self.txEncoder(src, srcMsk, keyMsk, True)
        assert not torch.isnan(result).any()
        return result

    def create_causal_mask(self, nseq):
        mask = torch.triu(torch.ones(nseq, nseq, dtype=self.dtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-1e9")).masked_fill(
            mask == 0, float(0.0)
        )
        return mask


class Decoder(nn.Module):
    def __init__(self, cfg, device=torch.device, dtype=torch.float32):
        super(Decoder, self).__init__()
        (
            self.d_model,
            self.srcLength,
            self.tgtLength,
            self.hiddens,
            self.device,
            self.dtype,
        ) = (
            cfg["d_model"],
            cfg["srcLength"],
            cfg["tgtLength"],
            cfg["hiddens"],
            device,
            dtype,
        )

        self.conv1d0, convLength0 = (
            nn.Sequential(
                nn.Conv1d(self.d_model, self.hiddens[0], 1, 1),
                nn.ReLU(True),
                nn.Conv1d(self.hiddens[0], self.hiddens[1], 5, 1),
                nn.AvgPool1d(kernel_size=5, stride=1),
            ),
            self.srcLength - 8,
        )

        self.conv1d1, convLength1 = (
            nn.Sequential(
                nn.Conv1d(self.d_model, self.hiddens[0], 5, 1),
                nn.ReLU(True),
                nn.Conv1d(self.hiddens[0], self.hiddens[1], 10, 1),
                nn.AvgPool1d(kernel_size=5, stride=1),
            ),
            self.srcLength - 17,
        )

        self.conv1d2, convLength2 = (
            nn.Sequential(
                nn.Conv1d(self.d_model, self.hiddens[0], 10, 1),
                nn.ReLU(True),
                nn.Conv1d(self.hiddens[0], self.hiddens[1], 20, 1),
                nn.AvgPool1d(kernel_size=5, stride=1),
            ),
            self.srcLength - 32,
        )

        self.cnnFusion = nn.Sequential(
            nn.Linear(
                convLength0 + convLength1 + convLength2,
                self.tgtLength,
            ),
            nn.ReLU(True),
            nn.Conv1d(self.hiddens[1], self.d_model, 1, 1),
        )

    def forward(self, src):
        src = src.transpose(-1, -2)
        cnv = torch.concat(
            [self.conv1d0(src), self.conv1d1(src), self.conv1d2(src)],
            dim=2,
        )
        result = self.cnnFusion(cnv).transpose(-1, -2)
        return result


class SpaceTx(nn.Module):
    def __init__(
        self,
        pth=None,
        cfg=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        super(SpaceTx, self).__init__()

        if pth:
            with open(pth) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)
                device = torch.device(cfg["device"])

        self.cfg, self.device, self.dtype = cfg, device, dtype
        self.encoder, self.decoder = Encoder(
            self.cfg["encoder"], self.device, dtype
        ), Decoder(self.cfg["decoder"], self.device, dtype)

        if "weight" in self.cfg:
            self.load_state_dict(
                torch.load(self.cfg["weight"], map_location=self.device)
            )

        self.to(self.device)

    def forward(self, src, srcDates, srcMsk):
        src = self.encoder(src, srcDates, srcMsk)
        result = self.decoder(src)
        return result
