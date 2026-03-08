import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm, trange

import torch
import torchinfo
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import os
import sys


from foolbox import PyTorchModel, accuracy,samples
from foolbox.attacks import FGSM,LinfPGD

if __package__ in (None, ""):
    # Allow running as: python substitute/train.py
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

from black_box.model import BlackBoxModel
from black_box.oracle import get_orcale_predictions
from substitute.dataset import SubstituteDataset, INDICES
from substitute.model import SubstituteModel

torch.manual_seed(42)

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')


#SUBSTITUTE TRAINING
def train_substitute_model(
p_epochs=6,
epochs=10,
lr=1e-2,
lambda_=0.1,
):


    substitute_dataset=SubstituteDataset(   
        root_dir='/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/substitute/data/training_set_0',
        get_predictions=get_orcale_predictions,
        transform=None
    )

    substitute_model=SubstituteModel()
    substitute_model.to(device)

    for p in trange(p_epochs + 1, desc='substitute training'):
        substitute_dataset = SubstituteDataset(
            root_dir=f'/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/substitute/data/training_set_{p}',
            get_predictions=get_orcale_predictions,
            transform=None,
        )

        train_dataloader=DataLoader(
            substitute_dataset,
            batch_size=8,
            shuffle=True
        )


        substitute_model.train_model(train_dataloader,epochs=epochs,lr=lr)

        substitute_model.jacobian_data_augmentation(
            substitute_dataset=substitute_dataset,
            p= (p+1),
            lambda_=lambda_,
            root_dir=f'/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/substitute/data/training_set_{p+1}',

        )

        torch.save(substitute_model.state_dict(),f'models/substitute_model_p_{p}.pt')
    torch.save(substitute_model.state_dict(), f'models/substitute_model.pt')




if __name__=="__main__":
    train_substitute_model()
