import cv2
import numpy as np
from glob import glob
from torch.utils.data import Dataset

class DatasetClass(Dataset):
    """
    Custom dataset class for image segmentation.

    Parameters:
        data (str): Path pattern for input image files.
        target (str): Path pattern for target mask files.
        transforms (callable, optional): Optional image transformations.

    Attributes:
        data (list): List of file paths for input images.
        target (list): List of file paths for target masks.
        transforms (callable, optional): Optional image transformations.

    Methods:
        __len__: Returns the total number of samples in the dataset.
        __getitem__: Retrieves and preprocesses the image and mask at the specified index.

    """
    def __init__(self, data, target, transforms=None):
        self.data = sorted(list(glob(data)))
        self.target = sorted(list(glob(target)))
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = cv2.imread(self.data[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.target[idx], cv2.IMREAD_GRAYSCALE)

        # Resize images if needed
        if img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        # Normalize and format the mask
        mask = np.stack([1 - mask, mask], axis=0, dtype=np.float32) / 255

        # Expand dimensions for the input image
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255

        if self.transforms:
            return self.transforms(img)

        return img, mask
