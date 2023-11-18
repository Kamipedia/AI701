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
from model import *       # Importing model definition from 'model.py'
from dataset import *     # Importing dataset class from 'dataset.py'

DEVICE='cuda:0'
model = Unet(1, 2)

def test_model(model, predictions=5):
  model = model.to(DEVICE)
  model.load_state_dict(torch.load('fvit_weights.pth'))
  model.eval()
  
  with torch.no_grad():
      for i, (x, y) in enumerate(test_loader):
          x = x.to(device)
          y = y.to(device)
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
          if i == predictions:
              break

test_model(model)
