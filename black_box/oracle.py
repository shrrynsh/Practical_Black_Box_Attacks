import os
import torch 
from torch import Tensor

from .model import BlackBoxModel

device=torch.device("cuda" if torchcuda.is_available() else 'cpu')

MODEL_PATH ="/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/black_box/blackbox.pt"

ORACLE=BlackBoxModel()
ORACLE.load_state_dict(torch.load(MODEL_PATH))

ORACLE.to(device)
ORACLE.eval()


def get_orcale_predictions(x : Tensor):

    x=x.to(device)
    return ORACLE(x).argmax(dim=-1)
