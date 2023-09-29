# dataLoader.py

import os
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms
import yaml

with open('config/config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Read parameters from the configuration file
train_csv = config["train_csv"]
val_csv = config["train_csv"]
batch_size = config["batch_size"]
num_workers = config["num_workers"]




class AutoencoderDataset(data.Dataset):
    'Generates data for Keras'

    def __init__(self, list_IDs, labels):    #, batch_size=32, dim=(88, 200), n_channels=3, n_classes=1, shuffle=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.labels = labels
        self.transform = transforms.Compose([transforms.ToTensor()
                                           #transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                           ])

        '''self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()'''



    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) ))# / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generation([self.list_IDs[index]], [self.labels[index]])
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, labels):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X_img = np.empty((128, 128, 1))
        y_cmd = np.empty([1])

        # for i, ID in enumerate(list_IDs_temp):
        ID = list_IDs_temp[0]
        img = cv2.imread(str(ID[0]))
        img = cv2.resize(img, (128, 128), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        X_img = img
        #print(X_img.shape)
        X_img = np.reshape(X_img, (128, 128, 1))
        X_img = X_img.transpose(2, 0, 1)
        y_cmd = ID[1]
        return X_img, y_cmd



def get_dataset():
    #-----read-csv-file------------#
    data_df = pd.read_csv(train_csv)
    X_train = data_df[['img', 'label']].values
    y_train = data_df['label'].values

    data_df_val = pd.read_csv(val_csv)
    X_val = data_df_val[['img', 'label']].values
    y_val = data_df_val['label'].values
    #---------------------------------#

    training_set = AutoencoderDataset(X_train, y_train)
    params = {'batch_size': batch_size,
              'sampler': SequentialSampler(training_set),
              'num_workers': num_workers,
              'drop_last': True}

    train_generator = data.DataLoader(training_set, **params)

    validation_set = AutoencoderDataset(X_val, y_val)
    params = {'batch_size': batch_size,
              'sampler': SequentialSampler(validation_set),
              'num_workers': num_workers,
              'drop_last': True}
    validation_generator = data.DataLoader(validation_set, **params)

    return train_generator, validation_generator
