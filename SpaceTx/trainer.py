import torch
from torch import nn
from .parse import SpaceDataset
from .model import SpaceTx
from torch.utils.data import DataLoader, random_split
import yaml


class SapceTxTrainer:
    def __init__(self, pth=None, cfg=None, dtype=torch.float32):
        if pth:
            with open(pth) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.cfg, self.dtype, self.device = cfg, dtype, torch.device(cfg["device"])

        self.model = SpaceTx(self.cfg["model"], self.device, dtype=self.dtype)

        data = self.cfg["data"]
        dataset = SpaceDataset(
            data["pth"], data["srcLength"], data["tgtLength"], data["isAE"]
        )
        testLength, total = int(data["eval"] * len(dataset)), len(dataset)
        trainDataset, testDataset = random_split(
            dataset, [total - testLength, testLength]
        )
        self.trainLoader, self.testLoader = DataLoader(
            trainDataset,
            batch_size=data["batches"],
            shuffle=True,
        ), DataLoader(testDataset, batch_size=data["batches"], shuffle=True)

        self.lossFn, self.optimizer, self.ckpt = (
            nn.MSELoss(),
            torch.optim.Adam(self.model.parameters(), lr=cfg["lr"]),
            cfg["ckpt"],
        )

    def __call__(self, epochs, save):
        print(
            f"Start Training : SpaceTx\n  configuration {self.cfg}\n  weights will be saved on {self.ckpt}"
        )
        trainLoss, testLoss = [], []
        for i in range(epochs):
            trainLoss.append(self.train())
            testLoss.append(self.test())
            print(
                f"    Epoch {i}:\n        Avg Train Loss : {trainLoss[-1]}\n        Avg Test Loss : {testLoss[-1]}"
            )
            
            if i % save == 0:
                path = f"{self.ckpt}/SapceTx_{i}.pth"
                torch.save(self.model.state_dict(), path)

    def train(self):
        self.model.train()
        totalLoss, steps = 0.0, 0.0

        for idx, batch in enumerate(self.trainLoader):
            src, tgt = (
                self.preproc(batch["src"]).to(self.device).to(self.dtype).unsqueeze(-1),
                self.preproc(batch["tgt"]).to(self.device).to(self.dtype).unsqueeze(-1),
            )

            srcDates, tgtDates = batch["srcDates"].to(self.device).to(
                torch.int32
            ), batch["tgtDates"].to(self.device).to(torch.int32)

            _tgt = torch.zeros_like(tgt, device=self.device, dtype=self.dtype)
            _tgt[:, 0] += src[:, -1]

            self.optimizer.zero_grad()
            pred = self.model(src, srcDates, None, _tgt, tgtDates, None)
            assert not torch.isnan(pred).any()
            loss = torch.sqrt(self.lossFn(pred, tgt))
            loss.backward()
            self.optimizer.step()
            totalLoss, steps = totalLoss + loss.item(), steps + 1

        return totalLoss / steps

    def test(self):
        self.model.eval()
        totalLoss, steps = 0.0, 0.0

        with torch.no_grad():
            for idx, batch in enumerate(self.testLoader):
                src, tgt = (
                    self.preproc(batch["src"])
                    .to(self.device)
                    .to(self.dtype)
                    .unsqueeze(-1),
                    self.preproc(batch["tgt"])
                    .to(self.device)
                    .to(self.dtype)
                    .unsqueeze(-1),
                )

                srcDates, tgtDates = batch["srcDates"].to(self.device).to(
                    torch.int32
                ), batch["tgtDates"].to(self.device).to(torch.int32)

                _tgt = torch.zeros_like(tgt, device=self.device, dtype=self.dtype)
                _tgt[:, 0] += src[:, -1]

                pred = self.model(src, srcDates, None, _tgt, tgtDates, None)
                assert not torch.isnan(pred).any()
                loss = torch.sqrt(self.lossFn(pred, tgt))
                totalLoss, steps = totalLoss + loss.item(), steps + 1

        return totalLoss / steps

    def preproc(self, tensor):
        if tensor.dim() > 1:
            for i in range(tensor.size(0)):
                tensor[i, :] = self.preproc(tensor[i, :])
            return tensor

        isnan = torch.isnan(tensor)
        if torch.all(isnan):
            tensor = tensor.zero_()
            return tensor

        while torch.any(isnan):
            shifted = torch.roll(isnan, 1, dims=0)
            shifted[0] = False
            tensor[isnan] = tensor[shifted]
            isnan = torch.isnan(tensor)

        return tensor
