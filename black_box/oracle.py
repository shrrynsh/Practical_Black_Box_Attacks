import os
import torch 
from torch import Tensor

from .model import BlackBoxModel

device=torch.device("cuda" if torchcuda.is_available() else 'cpu')

MODEL_PATH ="/opt/watchdog/users/shreyansh/adv_diff/adv_ml/Practical_Black_Box_Attacks/black_box/blackbox.pt"
