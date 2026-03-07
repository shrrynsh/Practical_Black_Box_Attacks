import torch 
from torch import nn
from torch.nn import functional as F 
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets improt MNIST

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cli import LightningCLI

from torchmetrics import Accuracy

from model import  BlackBoxModel

PATH_DATASETS='data'

class LightningMNIST(LightningModule):
    def __init__(
        self,
        data_dir: str = PATH_DATASETS,
        batch_size : int =64,
        lr : float = 1e-3,
    ):

    super().__init__()
    self.save_hyperparameters()

    self.data_dir=data_dir
    self.batch_size=batch_size
    self.lr=lr


    self.num_classes=10
    self.dims=(1,28,28)
    channels,width,height=self.dims
    self.transform=trasforms.Compose(
        [
            transofrms.ToTensor(),
        ]
    )
    self.model=BlackBoxModel()
    self.val_accuracy=Accuracy()
    self.test_accuracy=Accuracy()

    def forward(self,x):
        out=self.model(x)
        return out 

    def training_step(self,batch,batch_idx):
        x,y=batch
        logits=self(x)
        loss=F.nll_loss(logits,y)
        return loss 

    def validation_step(self,bathc,batch_idx):
        x,y=batch
        logits=self(x)
        loss=F.nll_loss(logits,y)
        preds=torch.argmax(logtis,dim=1)
        self.val_accuracy.update(preds,y)

        self.log("val_loss",loss,prog_bar=True)
        self.log("val_acc",self.val_accuracy,prog_bar=True)



    def test_step(self,batch,batch_idx):
        x,y=batch
        logits=self(x)
        loss=F.nll_loss(logits,y)
        preds=torch.argmax(logits,dim=1)
        self.test_accuracy.update(preds,y)

        self.log("test_loss",loss,prog_bar=True)
        self.log("test_acc",self.test_accuracy,prog_bar=True)


    def configure_optimizers(self):
        optimizer=torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer



    def prepare_data(self):
        
