import torch
import torch.nn as nn
from monai.networks.blocks.transformerblock import TransformerBlock
from monai.networks.blocks.selfattention import SABlock
from monai.networks.blocks import PatchEmbeddingBlock
from monai.networks.nets import vit
from torch.utils.data import Dataset, DataLoader
from albumentations import Resize, Normalize, Compose
from glob import glob
import numpy as np
import cv2
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
from monai.utils import ensure_tuple_rep
DEVICE='cuda:3'
LR=1e-3
IMG_SIZE = 512
EPOCHS = 500
VAL_EPOCHS = 1
BATCH_SIZE = 8
TOLERANCE=5
TOL_THRESHOLD=0.01



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dims),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class UpsampledViT(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, hidden_size, num_heads, spatial_dims):
        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.hidden_size = hidden_size
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))

        self.vit = vit.ViT(in_channels=in_channels, img_size=img_size, patch_size=patch_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=3, spatial_dims=spatial_dims, classification=False)
        self.transp1 = nn.ConvTranspose2d(hidden_size, out_channels, kernel_size=2, stride=2)
        self.transp2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.transp3 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(out_channels, out_channels)
        self.conv2 = DoubleConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]



    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x        

    def forward(self, x):
        # resudual unet block
        x, h = self.vit(x)
        x = self.proj_feat(x)
        x = self.transp1(x)
        x = self.transp2(x)
        conv1 = self.conv1(x)
        x = x.clone() + conv1
        x = self.bn1(x)
        x = self.transp3(x)
        conv2 = self.conv2(x)
        x = x.clone() + conv2
        x = self.bn2(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512], patch_size=8, hidden_dim=512, img_size=512, num_heads=4, spatial_dims=2):
        super().__init__()
        self.ups = nn.ModuleList()
        self.features = features
        self.extractors = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.hidden_size=hidden_dim
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        # Down part of U-Net
        for i, feature in enumerate(features):
            if i < len(features)//2:
            # if True:
                self.extractors.append(DoubleConv(in_channels, feature))
            else:
                self.extractors.append(UpsampledViT(in_channels=in_channels, 
                                                    out_channels=feature,
                                                    img_size=tuple(sz//(2**i) for sz in img_size), 
                                                    patch_size=self.patch_size, 
                                                    hidden_size=hidden_dim, 
                                                    num_heads=num_heads, 
                                                    spatial_dims=spatial_dims))
            self.downs.append(Downsample(feature))
            in_channels = feature

        # Up part of U-Net
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x):
        skip_connections = []
        for extractor, down in zip(self.extractors, self.downs):
            x = extractor(x)
            skip_connections.append(x)
            x = down(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

class DatasetClass(Dataset):
    def __init__(self, data, target, transforms=None):
        self.data = sorted(list(glob(data)))
        self.target = sorted(list(glob(target)))
        self.transforms=transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.target[idx], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask>0, 1, 0).astype(np.float32)
        mask = np.stack([mask, 1-mask], axis=0)
        data = {"image": img, "mask": mask}
        if self.transforms:
            return self.transforms(data)
        if len(data["image"].shape) > 2:
            data["image"] = np.transpose(data["image"], (2, 0, 1)).astype(np.float32)
        else:
            data["image"] = np.expand_dims(data["image"], axis=0).astype(np.float32)
        # data["mask"] = np.expand_dims(data["mask"], axis=0).astype(np.float32)
        return data["image"]/255.0, data["mask"]
    
train_ds = DatasetClass(data='/share/sda/nurenzhaksylyk/maxim.popov/data/submission/syntax/train/images/*', target='/share/sda/nurenzhaksylyk/maxim.popov/data/miccai_binary/syntax/train/images/*')
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
test_ds = DatasetClass(data='/share/sda/nurenzhaksylyk/maxim.popov/data/submission/syntax/test/images/*', target='/share/sda/nurenzhaksylyk/maxim.popov/data/miccai_binary/syntax/test/images/*')
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=24)

def train_epoch(model, train_dl, optimizer, criterion, scheduler=None):
    model.train()
    loss_history = 0
    for x_batch, y_batch in tqdm(train_dl, leave=False):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step(loss)
        loss_history += loss.item() * x_batch.size(0)
    return loss_history / len(train_dl.dataset)

@torch.no_grad()
def val_epoch(model, val_dl, criterion):
    model.eval()
    loss_history = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_dl, leave=False):
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss_history += loss.item() * x_batch.size(0)
            predicted = torch.argmax(y_pred.data, 1)
            y_batch = torch.argmax(y_batch.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return loss_history / len(val_dl.dataset), correct / total

@torch.no_grad()
def test_accuracy(model, val_dl):
    model = model.to(DEVICE)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in val_dl:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            y_pred = model(x_batch)
            predicted = torch.argmax(y_pred.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total


def train(model, train_dl, val_dl, optimizer, criterion, scheduler=None, epochs=50, val_ep=10, task="baseline", tolerance=3, tol_threshold=0.01, early_stopping=True):
    train_hist = []
    acc_hist = []
    val_loss = 0
    best_loss = np.inf
    for ep in range(epochs+1):
        train_loss = train_epoch(model, train_dl, optimizer, criterion, scheduler)
        if (ep) % val_ep == 0:
            val_loss, val_acc = val_epoch(model, val_dl, criterion)
            print(f'Epoch {ep}: train loss {train_loss:.4}, val loss {val_loss:.4}')
            if abs(val_loss) < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'best_{task}.pth')
        else:
            print(f'Epoch {ep}: train loss {train_loss:.4}, best val loss {best_loss:.4}')
        train_hist.append((train_loss, val_loss))
        acc_hist.append(val_acc)
        if early_stopping and ep > tolerance*val_ep and val_loss - np.mean(np.array(train_hist)[-tolerance*val_ep:, 1]) >= tol_threshold:
                print('Early stopping')
                break
    
    optimizer.zero_grad()
    gc.collect()
    torch.cuda.empty_cache()
    print("Training finished, best val loss: ", f"{best_loss:.4}", "recovering best model")
    model.load_state_dict(torch.load(f'best_{task}.pth'))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot([x[0] for x in train_hist], label='train')
    ax[0].plot([x[1] for x in train_hist], label='val', color='orange')
    ax[1].plot(acc_hist, label='validation accuracy', color='blue')
    ax[0].legend()
    ax[0].set_title('Loss')
    ax[1].legend()
    ax[1].set_title('Validation accuracy')
    plt.plot()
    return train_hist, acc_hist
import random


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

START_seed()
from monai.losses import DiceFocalLoss
model = UNet(in_channels=1)
model = model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = DiceFocalLoss()

torch.autograd.set_detect_anomaly(True)
hist_base = train(model, train_loader, test_loader, optimizer, criterion, epochs=EPOCHS, val_ep=VAL_EPOCHS, task="vit", tolerance=TOLERANCE, tol_threshold=TOL_THRESHOLD, early_stopping=False)
