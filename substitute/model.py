import os 
import torch

import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader,Dataset

from torchvision.utils import save_image

from tqdm.notebook import tqdm, trange
from typing import Optional

device=troch.device("cuda" if torch.cuda.is_available() else 'cpu')


class SubstituteModel(nn.Model):
    def __init__(self,num_classes : int =10) -> None:
        super(SubstituteModel,self).__init__()

        self.conv=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),

        )

        self.classifier=nn.Sequential(
            nn.Linear(64*24*24,32),
            nn.ReLU(),
            nn.Linear(32,num_classes)

        )



        def forward(self, x: Tensor) -> Tensor:
            out=self.conv(x)
            out=torch.flatten(out,1)
            out=self.classifier(out)

            out =F.log_softmax(out,dim=1)
            return out 


        def add_optimizer(self,lr: float =1e-3):

            self.optimizer=torch.optim.SGD(self.parameters(),lr=lr,mommentum=0.9)


        def get_loss(self,predicition_batch :Tensor, class_batch: Tensor) -> Tensor:
            loss=F.nll_loss(prediciton_batch,class_batch.squeeze())
            return loss


        def _fit_batch(self,images_batches,class_batch):
            
