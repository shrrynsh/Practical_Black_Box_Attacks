import os 
import torch

import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader,Dataset

from torchvision.utils import save_image

from tqdm import tqdm, trange
from typing import Optional

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class SubstituteModel(nn.Module):
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

            self.optimizer=torch.optim.SGD(self.parameters(),lr=lr,momentum=0.9)


    def get_loss(self,predicition_batch :Tensor, class_batch: Tensor) -> Tensor:
            loss=F.nll_loss(prediction_batch,class_batch.squeeze())
            return loss


    def _fit_batch(self,images_batch,class_batch):
            images_batch,class_batch=images_batch.to(device),class_batch.to(device)
            self.optimizer.zero_grad()
            pred_batch=self(images_batch)
            loss=self.get_loss(pred_batch,class_batch)
            loss.backward()
            self.optimizer.step()

            return loss 



    def train_epoch(
            self,train_data :DataLoader,epoch :int,batch_size: Optional[int]=None
        ) ->  float:

            self.train()
            desc=f"Epoch{epoch}"
            total=len(train_data)*batch_size if batch_size else len(train_data)
            bar_fmt = "{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]"

            trn_loss=0
            trn_done=0

            pbar=tqdm(
                train_data,
                desc=desc,
                total=total,
                leave=False,
                miniters=1,
                unit_scale=True,
                bar_format=bar_fmt,
                position=1,
            )

            for imgs,labels in pbar:
                loss=self._fit_batch(imgs,labels)
                trn_loss+=loss.item()*imgs.shape[0]
                trn_done+=imgs.shape[0]


        

    def train_model(
        self,
        train_data :DataLoader,
        epochs: int,
        lr :float=1e-3,
        batch_size :Optional[int]=None,

    ) -> float:


        self.add_optimizer(lr)
        trnbar_fmt = "{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        pbar = tqdm(
            range(epochs),
            desc="Training",
            total=epochs,
            leave=False,
            unit="epoch",
            position=0,
            bar_format=trnbar_fmt,
        )


        for epoch in pbar:
            train_loss=self.train_epoch(train_data,epoch)
            pbar.set_postfix({"loss": "%.4g" % (train_loss)})


    def jacobian_data_augmentation(
        self,substitute_dataset: Dataset, p: int,lambda_:float, root_dir:str
    )  -> None:

        if not os.path.exists(root_dir):
            os.mkdir(root_dir)

        for i in trange(
            len(substitute_dataset), desc="Jacobian dataset augmentation", leave=False
        ):
            image, label = substitute_dataset.__getitem__(i)
            image, label = image.to(device), label.to(device)

            # The Jacobian has shape 10 x 28 x 28
            jacobian = torch.autograd.functional.jacobian(self, image.unsqueeze(dim=0)).squeeze()
            new_image = image + lambda_ * torch.sign(jacobian[label])

            # We save the tensors, some information was lost when saved as an image
            torch.save(image, f"{root_dir}/{i}.pt")
            torch.save(new_image, f"{root_dir}/{i + len(substitute_dataset)}.pt")
