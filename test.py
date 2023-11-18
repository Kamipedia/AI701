import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch    
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
from monai.networks.nets import vit
from monai.utils import ensure_tuple_rep
import random
import gc
from glob import glob
from tqdm import tqdm
from model import *       # Importing model definition from 'model.py'
from dataset import *     # Importing dataset class from 'dataset.py'
from train_loop import *  # Importing training loop from 'train_loop.py'

DEVICE = 'cuda:3'
IMG_SIZE = 512

model = Unet(in_channels=1, out_channels=2)
model = model.to(DEVICE)

START_seed()
train_ds = DatasetClass(data='/share/sda/nurenzhaksylyk/maxim.popov/data/submission/syntax/train/images/*', 
                        target='/share/sda/nurenzhaksylyk/maxim.popov/data/miccai_binary/syntax/train/images/*')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)

test_ds = DatasetClass(data='/share/sda/nurenzhaksylyk/maxim.popov/data/cruz_134/images/*', 
                       target='/share/sda/nurenzhaksylyk/maxim.popov/data/cruz_134/masks/*')
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)


model.load_state_dict(torch.load('fvit_weights.pth'))
model.eval()
with torch.no_grad():
    for i, (x, y) in enumerate(test_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        y_pred = model(x)
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        ax[0].imshow(x[0, 0], cmap="gray")
        ax[0].set_title("Image")
        ax[1].imshow(y[0, 1], cmap="gray")
        ax[1].set_title("Ground Truth")
        ax[2].imshow(y_pred[0, 1], cmap="gray")
        ax[2].set_title("Prediction")
        ax[3].imshow(y_pred[0].argmax(axis=0), cmap="gray")
        ax[3].set_title("Prediction argmax")
        plt.plot()
        if i == 5:
            break
