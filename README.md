# AI701

(description)

## Table of Contents

- [Introduction](#introduction)
- [Residual Block Class](#residual-block-class)
- [Upsampled Vision Transformer Class](#upsampled-vision-transformer-class)
- [UNet Class](#unet-class)
- [Usage](#usage)
- [Dataset Class](#dataset-class)
- [Training Script](#training-script)

## Introduction


## Residual Block Class

### `Resblock_torch`

The `Resblock_torch` class defines a residual block for the UNet architecture. It includes a customizable number of residual blocks within the module.

- **Parameters:**
  - `in_channels` (int): Number of input channels.
  - `num_res_blocks` (int, optional): Number of residual blocks within the module.

- **Attributes:**
  - `in_channels` (int): Number of input channels.
  - `bn` (nn.BatchNorm2d): Batch normalization layer.
  - `conv_block` (nn.ModuleList): List of convolutional blocks.
  - `output` (nn.BatchNorm2d): Batch normalization layer for the output.

- **Methods:**
  - `forward(input_)`: Defines the forward pass through the residual block.

## Upsampled Vision Transformer Class

### `UpsampledViT`

The `UpsampledViT` class represents an Upsampled Vision Transformer block.

- **Parameters:**
  - `in_channels` (int): Number of input channels.
  - `out_channels` (int): Number of output channels.
  - `img_size` (tuple): Size of the input feature map.
  - `patch_size` (tuple): Size of the feature extractor window.
  - `hidden_size` (int): Size of the perceptron layer output.
  - `num_heads` (int): Number of regions to divide the input for feature extraction.
  - `spatial_dims` (int): Dimensionality of the prediction.

- **Attributes:**
  - Various attributes for layers and projections.

- **Methods:**
  - `proj_feat(x)`: Performs feature projection after the transformer into the defined shape.
  - `forward(x)`: Defines the forward pass through the UpsampledViT block.

## UNet Class

### `Unet`

The `Unet` class represents the UNet network.

- **Parameters:**
  - `in_channels` (int): Number of input channels.
  - `out_channels` (int): Number of predicted classes.
  - `depth` (int, optional): Parameter that regulates feature maps' dimensionality.

- **Attributes:**
  - Various attributes for extractors, deconvolutions, and final output.

- **Methods:**
  - `forward(input_)`: Defines the forward pass through the UNet network.

## Usage



## Dataset Class



## Training Script

