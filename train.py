from model import SemanticSegmentationModel
from cityscapes_dataset import CitySegDataset
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import datetime


ds = CitySegDataset(
    root="data/cityscapes_data",
    split="train",
    transforms_img=trans.Compose([trans.ToTensor(), trans.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
)
train_dl = DataLoader(dataset=ds, batch_size=32, shuffle=True, num_workers=8)

ds = CitySegDataset(
    root="data/cityscapes_data",
    split="val",
    transforms_img=trans.Compose([trans.ToTensor(), trans.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]),
)
val_dl = DataLoader(dataset=ds, batch_size=32, shuffle=False, num_workers=8)

model = SemanticSegmentationModel(num_classes=35)

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
nowname = model.__class__.__name__ + "_" + now
logger = WandbLogger(name=nowname, id=nowname, project="NN-compression")

trainer = pl.Trainer(
    benchmark=True,
    max_epochs=100,
    logger=logger,
    callbacks=ModelCheckpoint(dirpath=f"logs/{nowname}", monitor="val/loss", save_last=True),
)
trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
