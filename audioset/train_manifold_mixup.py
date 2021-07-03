import pickle
import yaml
import pandas as pd
import numpy as np 
from albumentations import Compose, ShiftScaleRotate, GridDistortion
from albumentations.pytorch import ToTensor
import librosa
import sys
import os
import numpy as np
import pandas as pd
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import optim
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader    
from tqdm import tqdm, trange
import random

from utils import prepare_data, AudioDataset, Task5Model, AudioDataset3
from mixup_model import MixupModel

from augmentation.SpecTransforms import MelSpectrogram, ResizeSpectrogram, TimeMask, FrequencyMask, RandomCycle
from augmentation.AudioTransforms import ResizeWaveform, NormalizeWaveform, TimeStretching, PitchShifting

def run():
    
    fold0 = pd.read_csv('./data/split/fold_0.txt', delimiter=" ", header=None)
    fold1 = pd.read_csv('./data/split/fold_1.txt', delimiter=" ", header=None)
    fold2 = pd.read_csv('./data/split/fold_2.txt', delimiter=" ", header=None)
    fold3 = pd.read_csv('./data/split/fold_3.txt', delimiter=" ", header=None)
    fold4 = pd.read_csv('./data/split/fold_4.txt', delimiter=" ", header=None)
    
    train_df = pd.concat([fold4, fold0, fold1])
    valid_df = fold2
    test_df = fold3

    resize = transforms.Compose([ResizeSpectrogram(frames=636)])

    spec_transforms = transforms.Compose([
        TimeMask(), 
        FrequencyMask(), 
        RandomCycle()
    ])

    albumentations_transform = Compose([
        # ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
        # GridDistortion(),
        ToTensor()
    ])

    # Create the datasets and the dataloaders
    train_dataset = AudioDataset3(train_df, path = './data/logmelspec/', 
        resize = resize,
        image_transform = albumentations_transform)
        # spec_transform = spec_transforms)

    # train_dataset = AudioDataset3(train_df, path = './data/logmelspec/', resize = resize,  spec_transform = spec_transforms)
    valid_dataset = AudioDataset3(valid_df, path = './data/logmelspec/', resize = resize)

    val_loader = DataLoader(valid_dataset, 16, shuffle=False, num_workers =2)
    train_loader = DataLoader(train_dataset, 16, shuffle=True, num_workers = 2)

    # Instantiate the model
    model = MixupModel(9).to(device)

    # Define optimizer, scheduler and loss criteria
    # optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    epochs = 30
    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0
    
    for i in range(epochs):
        print('Epoch: ', i)

        this_epoch_train_loss = 0
        for sample in tqdm(train_loader):

            inputs = sample['data'].to(device)
            label = sample['labels'].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                model = model.train()
                outputs, y_a, y_b, lam = model(inputs, mixup=True, target=label, mixup_alpha=2.0)

                loss = lam*criterion(outputs, y_a) + (1-lam)*criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                this_epoch_train_loss += loss.detach().cpu().numpy()

        this_epoch_valid_loss = 0
        for sample in tqdm(val_loader):
            inputs = sample['data'].to(device)
            labels = sample['labels'].to(device)
            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels) 
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_df)
        this_epoch_valid_loss /= len(valid_df)

        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save(model.state_dict(), './data/model_system1_fold5')
            epochs_without_new_lowest = 0
        else:
            epochs_without_new_lowest += 1

        if epochs_without_new_lowest >= 25:
            break

        print(this_epoch_train_loss, this_epoch_valid_loss)

        scheduler.step(this_epoch_valid_loss)

if __name__ == "__main__":
    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    run()