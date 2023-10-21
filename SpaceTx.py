from SpaceTx import SapceTxTrainer

trainer = SapceTxTrainer(0.01, "./train3/trainer.yml")
trainer(5000, 1)
