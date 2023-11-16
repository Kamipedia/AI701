import os
import cv2
import numpy as np
import torch    
from torch import nn
from monai.networks.nets import vit
from monai.utils import ensure_tuple_rep
import random
import gc
from glob import glob
from tqdm import tqdm


class Resblock_torch(nn.Module):
    def __init__(self, in_channels, num_res_blocks=3):
        super().__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.conv_block = nn.ModuleList()
        self.output = nn.BatchNorm2d(self.in_channels)

        for _ in range(num_res_blocks):
            self.conv_block.append(nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
                    nn.ReLU(),
                    nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1),
                    nn.ReLU(),
                    nn.BatchNorm2d(self.in_channels)
                    )
            )

        
    # define forward pass
    def forward(self, input_):
        assert self.in_channels == input_.shape[1]

        x = self.bn(input_)

        for conv in self.conv_block:
            x = conv(x) + input_

        x = self.output(x)
        assert x.shape == input_.shape

        return x



class UpsampledViT(nn.Module):
    def __init__(self, in_channels, out_channels, img_size, patch_size, hidden_size, num_heads, spatial_dims):
        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.hidden_size = hidden_size
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.vit = vit.ViT(in_channels=in_channels, 
                           img_size=img_size, 
                           patch_size=patch_size, 
                           hidden_size=hidden_size,
                           num_heads=num_heads, 
                           num_layers=3, 
                           spatial_dims=spatial_dims, 
                           classification=False
                           )
        self.transp1 = nn.ConvTranspose2d(hidden_size, out_channels, kernel_size=2, stride=2)
        self.transp2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = Resblock_torch(out_channels)
        self.conv2 = Resblock_torch(out_channels)
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
        x, _ = self.vit(x)
        x = self.proj_feat(x)
        x = self.transp1(x)
        x =  x.clone() + self.conv1(x)
        x = self.bn1(x)
        x = self.transp2(x)
        x =  x.clone() + self.conv2(x)
        x = self.bn2(x)

        return x



class Unet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = [2**(i+4) for i in range(depth+1)]
        self.extractors = nn.ModuleList()
        self.deconvs = nn.ModuleList()

        if len(self.depth)%2 != 0:
            length = len(self.depth)+1
        else:
            length = len(self.depth)

        for i, feature in enumerate(self.depth):
            if i < length//2:
                self.extractors.append(nn.Sequential(
                        nn.Conv2d(in_channels, feature, (2, 2), stride=2),
                        Resblock_torch(feature)
                    )
                )
            else:
                self.extractors.append(nn.Sequential(
                        nn.Conv2d(in_channels, feature, (2, 2), stride=2),
                        UpsampledViT(feature, feature, (self.depth[-i-1], self.depth[-i-1]), (4,4), 256, 4, 2)
                    )
                )

            in_channels = feature

        self.bottleneck = nn.Sequential(
            Resblock_torch(self.depth[-1]),
            Resblock_torch(self.depth[-1])
        )

        for feature in reversed(self.depth):
            self.deconvs.append(nn.Sequential(
                                    nn.ConvTranspose2d(feature, int(feature//2), (2, 2), stride = 2),
                                    Resblock_torch(int(feature//2)),
                                            )
                                )

        self.output = nn.Sequential(
            nn.Conv2d(int(self.depth[0]//2), int(self.depth[0]//2), (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(int(self.depth[0]//2)),
            nn.Conv2d(int(self.depth[0]//2), int(self.depth[0]//2), (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(int(self.depth[0]//2)),
            nn.Conv2d(int(self.depth[0]//2), self.out_channels, (1, 1), padding=0),
            nn.Softmax(dim=1)
        )
    
    # define forward pass
    def forward(self, input_):
        assert input_.shape[1] == self.in_channels

        skip_connections = []
        x = input_

        for extractor in self.extractors:
            x = extractor(x)
            skip_connections.append(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1][1:]

        for idx in range(len(self.deconvs)-1):
            x = self.deconvs[idx](x)
            skip_connection = skip_connections[idx]
            x = skip_connection + x

        x = self.deconvs[-1](x)
        
        return self.output(x)
