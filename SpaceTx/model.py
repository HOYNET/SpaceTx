import torch
import torch.nn as nn
import yaml

# https://github.com/CVxTz/time_series_forecasting/blob/main/LICENSE


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class SpaceTx(nn.Module):
    def __init__(self, pth=None, cfg=None):
        super().__init__()

        if pth:
            with open(pth) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.ninput, self.dropout, self.nchannel, self.device = (
            cfg["ninput"],
            cfg["dropout"],
            cfg["nchannel"],
            torch.device(cfg["device"]),
        )

        self.input_pos_embedding = torch.nn.Embedding(1024, embedding_dim=self.nchannel)
        self.target_pos_embedding = torch.nn.Embedding(
            1024, embedding_dim=self.nchannel
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.nchannel,
            nhead=16,
            dropout=self.dropout,
            dim_feedforward=4 * self.nchannel,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.nchannel,
            nhead=16,
            dropout=self.dropout,
            dim_feedforward=4 * self.nchannel,
        )

        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.input_projection = nn.Linear(self.ninput[0], self.nchannel)
        self.output_projection = nn.Linear(self.ninput[1], self.nchannel)

        self.linear = nn.Linear(self.nchannel, 1)

        self.do = nn.Dropout(p=self.dropout)

        if "weight" in cfg:
            self.load_state_dict(torch.load(cfg["weight"], map_location=self.device))
            print("load weight")

        self.to(self.device)

    def encode_src(self, src, srcMsk):
        src_start = self.input_projection(src).permute(1, 0, 2)

        in_sequence_len, batch_size = src_start.size(0), src_start.size(1)
        pos_encoder = (
            torch.arange(0, in_sequence_len, device=src.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )

        pos_encoder = self.input_pos_embedding(pos_encoder).permute(1, 0, 2)

        src = src_start + pos_encoder

        src = self.encoder(src, src_key_padding_mask=srcMsk) + src_start

        return src

    def decode_trg(self, trg, tgtMsk, memory, memMsk):
        trg_start = self.output_projection(trg).permute(1, 0, 2)

        out_sequence_len, batch_size = trg_start.size(0), trg_start.size(1)

        pos_decoder = (
            torch.arange(0, out_sequence_len, device=trg.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_decoder = self.target_pos_embedding(pos_decoder).permute(1, 0, 2)

        trg = pos_decoder + trg_start

        trg_mask = gen_trg_mask(out_sequence_len, trg.device)

        out = (
            self.decoder(
                tgt=trg,
                memory=memory,
                memory_key_padding_mask=memMsk,
                tgt_mask=trg_mask,
            )
            + trg_start
        )

        out = out.permute(1, 0, 2)

        out = self.linear(out)

        return out

    def forward(self, src, srcMsk, tgt, tgtMsk):
        srcMsk, tgtMsk = srcMsk, tgtMsk
        src = self.encode_src(src, srcMsk)
        out = self.decode_trg(trg=tgt, tgtMsk=tgtMsk, memory=src, memMsk=srcMsk)

        return out


if __name__ == "__main__":
    n_classes = 100

    source = torch.rand(size=(32, 16, 1))
    target_in = torch.rand(size=(32, 16, 1))
    target_out = torch.rand(size=(32, 16, 1))

    ts = SpaceTx("./train3/SpaceTx.yml")

    pred = ts(source, target_in)

    print(pred.size())
