import cv2
import numpy as np
from glob import glob


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

        if img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        mask = mask
        mask = np.stack([1-mask, mask], axis=0, dtype=np.float32)/255
        img = np.expand_dims(img, axis=0).astype(np.float32)/255
        if self.transforms:
            return self.transforms(img)
        
        return img, mask
