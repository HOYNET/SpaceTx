import torch
from torch import nn
from .parse import SpaceDataset
from .model import SpaceTx
from torch.utils.data import DataLoader, random_split
import yaml
from lion_pytorch import Lion


class SapceTxTrainer:
    def __init__(self, lr, pth=None, cfg=None, dtype=torch.float32):
        if pth:
            with open(pth) as f:
                cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.lr, self.cfg, self.dtype, self.device = (
            lr,
            cfg,
            dtype,
            torch.device(cfg["device"]),
        )

        self.model = SpaceTx(self.cfg["model"])

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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=self.optimizer, lr_lambda=lambda epoch: 0.95**epoch
        )

        self.lossFn, self.ckpt = (
            nn.MSELoss(),
            cfg["ckpt"],
        )

        self.ckptEpochs = 0

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
            self.scheduler.step(i)

            self.ckptEpochs += 1
            if self.ckptEpochs % save == 0:
                path = f"{self.ckpt}/SapceTx_{self.ckptEpochs}.pth"
                torch.save(self.model.state_dict(), path)

    def train(self):
        self.model.train()
        totalLoss, steps = 0.0, 0.0

        for idx, batch in enumerate(self.trainLoader):
            src, tgt = (
                batch["src"].to(self.device).to(self.dtype).unsqueeze(-1),
                batch["tgt"].to(self.device).to(self.dtype).unsqueeze(-1),
            )

            srcMsk, tgtMsk = torch.isnan(src).squeeze(-1), torch.isnan(tgt).squeeze(-1)
            src[srcMsk] = 0

            self.optimizer.zero_grad()
            _tgt = torch.zeros_like(tgt, dtype=tgt.dtype, device=tgt.device)
            _tgt[:, 0] += src[:, -1]
            pred = self.model(src, srcMsk, _tgt, tgtMsk)
            assert not torch.isnan(pred[~tgtMsk]).any()
            loss = torch.sqrt(self.lossFn(pred[~tgtMsk], tgt[~tgtMsk]))
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
                    batch["src"].to(self.device).to(self.dtype).unsqueeze(-1),
                    batch["tgt"].to(self.device).to(self.dtype).unsqueeze(-1),
                )

                srcMsk, tgtMsk = torch.isnan(src).squeeze(-1), torch.isnan(tgt).squeeze(
                    -1
                )
                src[srcMsk] = 0

                _tgt = torch.zeros_like(tgt, dtype=tgt.dtype, device=tgt.device)
                _tgt[:, 0] += src[:, -1]
                pred = self.model(src, srcMsk, _tgt, tgtMsk)
                assert not torch.isnan(pred[~tgtMsk]).any()
                loss = torch.sqrt(self.lossFn(pred[~tgtMsk], tgt[~tgtMsk]))
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
