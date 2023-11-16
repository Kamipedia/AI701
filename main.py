import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch    
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image, ImageReadMode
from monai.networks.nets import vit
from monai.utils import ensure_tuple_rep
import random
import gc
from glob import glob
from tqdm import tqdm
from model import *
from dataset import *
from train_loop import *


DEVICE='cuda:3'
TASK="fvit_unet_softmax_bce_loss"
LR=1e-3
IMG_SIZE = 512
EPOCHS = 50
VAL_EPOCHS = 1
BATCH_SIZE = 32
TOLERANCE=5
TOL_THRESHOLD=0.01


START_seed()

train_ds = DatasetClass(data='/share/sda/nurenzhaksylyk/maxim.popov/data/submission/syntax/train/images/*', 
                        target='/share/sda/nurenzhaksylyk/maxim.popov/data/miccai_binary/syntax/train/images/*'
                        )
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)

test_ds = DatasetClass(data='/share/sda/nurenzhaksylyk/maxim.popov/data/submission/syntax/test/images/*', 
                       target='/share/sda/nurenzhaksylyk/maxim.popov/data/miccai_binary/syntax/test/images/*'
                       )
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)


model = Unet(in_channels=1, out_channels=2)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.BCELoss()#nn.CrossEntropyLoss()
scheduler = CosineAnnealingLR(optimizer, 10, 0, -1, verbose=False)

if os.path.isfile(f'best_{TASK}.pth'):
    model.load_state_dict(torch.load(f'best_{TASK}.pth'))
    print('Weights has been preloaded')


torch.autograd.set_detect_anomaly(True)
hist_base = train(model, train_loader, 
                  test_loader, 
                  optimizer, 
                  criterion, 
                  scheduler, 
                  epochs=EPOCHS, 
                  val_ep=VAL_EPOCHS, 
                  task=TASK, 
                  tolerance=TOLERANCE, 
                  tol_threshold=TOL_THRESHOLD, 
                  early_stopping=False
                  )
