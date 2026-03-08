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
