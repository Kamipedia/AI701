# AI701

 A novel CNN-transformer-based architecture Faster-UNETR that combines the efficiency of CNNs and transformers

 ![faster_unetr_diagram](https://github.com/Kamipedia/AI701/assets/91109627/c4ffcf84-d64c-464e-aa48-16520b9fec30)


- [Introduction](#introduction)
- [Residual Block Class](#residual-block-class)
- [Upsampled Vision Transformer Class](#upsampled-vision-transformer-class)
- [UNet Class](#unet-class)
- [Dataset Class](#dataset-class)
- [Usage](#usage)

## Introduction

Medical image segmentation is one of the significant challenges in the field of computer vision. Being a main tool in computer-aided diagnosis (CAD), it offers a visual representation of the body's interior for clinical analysis. This method can assist doctors in detecting and localizing the boundaries of organ anomalies, increasing the accuracy and speed of the diagnosis process. For this reason, high precision and robustness of segmentation models are of vital importance.

Modern medical image segmentation techniques are typically symmetrical encoder-decoder-based Convolutional Neural Networks (CNNs). They first encode the image into the latent space and then learn to decode the position of the regions of interest back into images. Incorporation of skip connections between the encoder and decoder part results in U-Net, the original version and modifications of which tend to be the most popular modern model architectures for medical image segmentation. These skip connections are used to share features from different levels, improving the model's ability to perceive fine details and solving the problem of information loss during down-sampling and up-sampling.  

Despite the success of CNN models, they have limited receptive field and inductive bias towards the locality of features in data, which prevents them from modeling global relationships within the data. To alleviate the locality problem of CNNs, transformer architectures have been adapted from the field of Natural language Processing into Computer Vision tasks. Recent architectural developments like ViT have outperformed CNNs in benchmark imaging tasks. Usually, ViT-like models divide the images into non-overlapping patches and add spatial positioning to them using positional encoding. Standard transformer layers then process this enhanced representation to capture long-range dependencies in the data. 

Considering the impressive abilities of CNNs and transformers to capture local and global context, a network that combines their strong sides should show even better performance than purely convolutional or transformer-based models. Recently proposed models for medical image segmentation such as HiFormer, and Fully Convolutional Transformer  demonstrated state-of-the-art performance on the 3D segmentation task on MRI datasets. Inspired by such development, we decided to implement a model incorporating both convolutional and transformer blocks, but for a 2D segmentation task. 

In this paper, we propose a novel CNN-transformer-based architecture Faster-UNETR that combines the efficiency of CNNs and transformers in capturing local feature representations and global long-distance dependencies respectively. Particularly, we adopted the structure of Faster Vit and incorporated it into the UNETR model. The main alterations happen in the downsample pass, where the input image is firstly processed by purely convolutional layers for local feature extraction. To address the absence of global representation, the Upsampled ViT module is utilized atop the deep features of CNNs to grasp long-range dependencies. The resultant model is then utilized for semantic segmentation of coronary vessels in angiography images from ARCADE dataset.

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


## Dataset Class

Custom dataset class for image segmentation.

 - **Parameters:**
    - `data` (str): Path pattern for input image files.
    - `target` (str): Path pattern for target mask files.
    - `transforms` (callable, optional): Optional image transformations.

  - **Attributes:**
    - `data` (list): List of file paths for input images.
    - `target` (list): List of file paths for target masks.
    - `transforms` (callable, optional): Optional image transformations.

  - **Methods:**
    - `__len__`: Returns the total number of samples in the dataset.
    - `__getitem__`: Retrieves and preprocesses the image and mask at the specified index.
   
## Usage

### `model.py`

Contains model initializator.

### `dataset.py`

Contains dataset class initializator to operate images.

### `train_loop.py`

Contains train and validation loop functions.

### `test.py`

Contains code to test model.

### `main.py`

Initializer of all the classes and starts training model.


### Get started

For retraining model from scratch drop the `best_fivt_unet.pth` file and start the `main.py` file. To test current model start the `test.py` file.
