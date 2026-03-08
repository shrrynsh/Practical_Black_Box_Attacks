import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm.notebook import tqdm, trange

import torch
import torchinfo
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image


from foolbox import PytorchModel, accuracy,samples
from foolbox.attacks import FGSM,LinfPGD

form src.black_box_model.model import BlackBoxModel
from src.balck_box.oracle import get_orcale_predicitons
form src.substitute.datasets import SubstituteDataset, INDICES
form src.substitute_model import SubstituteModel

torch.manual_seed(42)

device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')


#SUBSTITUTE TRAINING
def train_substitute_model(
p_epochs=6,
epochs=10,
lr=1e-2,
lambda_=0.1,
):


    substitute_dataset=SubstituteDatset(   
        root_dir='/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/substitute/data/training_set_0',
        get_predictions=get_orcale_predictions,
        transform=None
    )

    substitute_mdoel=SubstituteModel()
    substitute_model.to(device)

    for p in trange(p_epochs+1,desc='substitute training'):
        substitute_dataset=SubstituteDatset(   
        root_dir='/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/substitute/data/training_set_{p}',
        get_predictions=get_orcale_predictions,
        transform=None
    )

        train_dataloader=DataLoader(
            substitute_dataset,
            batch_size=8,
            shuffle=True
        )


        substitute_model.train_model(train_dataloader,epochs=epochs,lr=lr)

        substitute_model.jacobian_dataset_augmentation(
            substitute_dataset=substitute_dataset,
            p= (p+1),
            lambda_=lambda_,
            root_dir=f'src/substitute/data/training_set_{p+1}',

        )

        torch.save(substitute_model.state_dict(),f'models/substitute_model_p_{p}.pt')
    torch.save(substitute_model.state_dict(), f'models/substitute_model.pt')




if __name__=="__main__":
    train_substitute_model()
