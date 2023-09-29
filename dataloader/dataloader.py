# dataLoader.py

import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

class AutoencoderDataset(data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            root_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied to the images.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.csv_file.iloc[idx, 0])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image = self.transform(image)

        return image

def get_autoencoder_dataloader(csv_file, root_dir, batch_size, shuffle=True):
    """
    Create a DataLoader for the autoencoder.

    Args:
        csv_file (str): Path to the CSV file with image paths and labels.
        root_dir (str): Directory with all the images.
        batch_size (int): Batch size for DataLoader.
        shuffle (bool, optional): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader for the autoencoder.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add any additional transformations here if needed
    ])

    dataset = AutoencoderDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader
