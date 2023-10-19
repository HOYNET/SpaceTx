import torch
from torch import nn
import yaml


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
        src, srcMsk = self.dateEncoding(src, dates), create_causal_mask(src.shape[1])
        result = self.txEncoder(src, srcMsk, keyMsk, True)
        assert not torch.isnan(result).any()
        return result


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()

        self.d_model, self.device, self.dtype = d_model, device, dtype
        self.dateEncoding = DateEncoding(self.d_model, self.device, self.dtype)

        self.nlayer, self.nhead, self.dim_feedforward, self.dropout = (
            nlayer,
            nhead,
            dim_feedforward,
            dropout,
        )
        self.decoderLayer = nn.TransformerDecoderLayer(
            self.d_model,
            self.nhead,
            self.dim_feedforward,
            self.dropout,
            "relu",
            batch_first=True,
            device=self.device,
            dtype=self.dtype,
        )

        self.txDecoder = nn.TransformerDecoder(self.decoderLayer, self.nlayer)

    def forward(self, tgt, src, dates, tgtKeyMsk=None, srcKeyMsk=None):
        tgt = self.dateEncoding(tgt, dates)
        srcMsk, tgtMsk = create_causal_mask(src.shape[1]), create_causal_mask(
            tgt.shape[1]
        )

        result = self.txDecoder(
            tgt, src, tgtMsk, srcMsk, tgtKeyMsk, srcKeyMsk, True, True
        )
        assert not torch.isnan(result).any()
        return result


def create_causal_mask(nseq):
    mask = torch.triu(torch.ones(nseq, nseq), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-1e9")).masked_fill(mask == 0, float(0.0))
    return mask


class SpaceTx(nn.Module):
    def __init__(
        self,
        pth,
        encoderCfg=None,
        decoderCfg=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    ):
        super(SpaceTx, self).__init__()

        self.device, self.dtype = device, dtype
        self.encoder, self.decoder = Cfg2Encoder(
            encoderCfg, pth, device, dtype
        ), Cfg2Decoder(decoderCfg, pth, device, dtype)

    def forward(self, src, srcDates, srcMsk, tgtDates, tgtMsk):
        src = self.encoder(src, srcDates, srcMsk)
        tgt = torch.zeros_like(src, dtype=self.dtype)
        tgt[:, 0] += src[:, -1]
        result = self.decoder(tgt, src, tgtDates, tgtMsk, srcMsk)
        return result


def Cfg2Encoder(cfg=None, pth=None, device=torch.device("cpu"), dtype=torch.float32):
    if pth:
        with open(pth) as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yml["encoder"]
    return Encoder(
        cfg["nlayer"],
        cfg["d_model"],
        cfg["nhead"],
        cfg["dim_feedforward"],
        cfg["dropout"],
        device,
        dtype,
    )


def Cfg2Decoder(cfg=None, pth=None, device=torch.device("cpu"), dtype=torch.float32):
    if pth:
        with open(pth) as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
            cfg = yml["decoder"]
    return Decoder(
        cfg["nlayer"],
        cfg["d_model"],
        cfg["nhead"],
        cfg["dim_feedforward"],
        cfg["dropout"],
        device,
        dtype,
    )


_device, _dtype, _cfg = torch.device("cpu"), torch.float32, "./cfg.yml"
spaceTx = SpaceTx(_cfg)
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
