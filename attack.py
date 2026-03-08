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
if __name__=="__main__":
    substitute_model=SubstituteModel()
    substitute_model.load_state_dict(torch.load('/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/models/substitute_model.pt',map_location=device))
    substitute_model.eval()

    mnist_test=MNIST(
        root='/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/substitute/data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )


    test_indices=[i for i in range(len(mnist_test)-5000) if i not in INDICES]
    images=torch.stack([mnist_test.__getitem__(i)[0] for i in test_indices])
    labels=torch.tensor([mnist_test.__getitem__(i)[1] for i in test_indices])
    images,labels=images.to(device),labels.to(device)

    preprocessing=None
    fmodel=PyTorchModel(substitute_model,bounds=(0,1),preprocessing=preprocessing)

    attack=FGSM()
    epsilons=[0.00,0.05,0.20,0.25,0.30]

    raw_advs,clipped_advs,success=attack(fmodel,images,labels,epsilons=epsilons)


    for (i,epsilons) in enumerate(epsilons):
        y_true,y_pred_oracle=[],[]

        for (label,image_adversarial) in zip(labels,raw_advs[i]):
            label,image_adversarial=label.to(device),image_adversarial.to(device)
            prediction_oracle=get_orcale_predictions(image_adversarial.unsqueeze(dim=0))

            y_true.append(label.item())
            y_pred_oracle.append(prediction_oracle.item())

            accuracy_oracle=100*accuracy_score(y_true,y_pred_oracle)

            conf_matrix=confusion_matrix(y_true,y_pred_oracle)
            conf_matrix = 100 * (conf_matrix / conf_matrix.sum(axis=1))
            df_cm = pd.DataFrame(conf_matrix, range(10), range(10))
            plt.figure(figsize=(7,7))
            sns.heatmap(df_cm, annot=True, annot_kws={"size": 8}, cbar=False, cmap='gray', fmt='2.1f') 
            
            plt.xlabel('Predicted class', fontweight='bold')
            plt.ylabel('Ground truth class', fontweight='bold')
            plt.title(f'ε = {epsilon},\nAccuracy oracle = {accuracy_oracle:.2f}')
            
            plt.show()