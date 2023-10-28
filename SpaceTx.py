from SpaceTx import SapceTxTrainer

trainer = SapceTxTrainer(0.001, "./train4/trainer.yml")
trainer(5000, 1)


from SpaceTx import SpaceTx
import pandas as pd
import torch

model = SpaceTx("./train4/SpaceTx.yml")

data = pd.read_excel("./Data/test.xlsx")

src = torch.Tensor(data["flux"]).type(torch.float32).unsqueeze(0).unsqueeze(-1)
tgt = torch.zeros((1, 30, 1)).type(torch.float32)
srcMsk, tgtMsk = torch.isnan(src).squeeze(-1), torch.isnan(tgt).squeeze(-1)
tgt[:, 0] += src[:, -1]
pred = model(src, srcMsk, tgt, tgtMsk)

print(src)
print(pred)

output = pd.DataFrame(
    {
        "date": torch.arange(1, 31).detach().numpy(),
        "flux": pred.squeeze(-1).squeeze(0).detach().numpy(),
    }
)

output.to_excel("result.xlsx", index=False)
print(output)
