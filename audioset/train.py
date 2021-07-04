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
import argparse
from utils import AudioDataset, Task5Model, mixup_data

from augmentation.SpecTransforms import ResizeSpectrogram, TimeMask, FrequencyMask, RandomCycle

def run(feature_type, num_frames, perm, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    folds = []
    for i in range(5):
        folds.append(pd.read_csv('./metadata/split/fold_{}.txt'.format(i), delimiter=" ", header=None))
    
    train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
    valid_df = folds[perm[3]]
    test_df = folds[perm[4]]

    spec_transforms = transforms.Compose([
        TimeMask(), 
        FrequencyMask(), 
        RandomCycle()
    ])

    albumentations_transform = Compose([
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
        GridDistortion(),
        ToTensor()
    ])

    # Create the datasets and the dataloaders

    train_dataset = AudioDataset(train_df, feature_type=feature_type,
        perm=perm,
        resize = num_frames,
        image_transform = albumentations_transform,
        spec_transform = spec_transforms)

    valid_dataset = AudioDataset(valid_df, feature_type=feature_type, perm=perm, resize = num_frames)

    val_loader = DataLoader(valid_dataset, 16, shuffle=False, num_workers =2)
    train_loader = DataLoader(train_dataset, 16, shuffle=True, num_workers = 2)

    # Define the device to be used
    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    
    print('Device: ', device)
    # Instantiate the model
    model = Task5Model(10).to(device)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    epochs = 100
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

            inputs, y_a, y_b, lam = mixup_data(inputs, label, 1.0)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(inputs)
                loss = criterion(outputs, label)
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
                loss = lam*criterion(outputs, y_a) + (1-lam)*criterion(outputs, y_b)
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_df)
        this_epoch_valid_loss /= len(valid_df)

        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save(model.state_dict(), './model/model_{}_{}'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
            epochs_without_new_lowest = 0
        else:
            epochs_without_new_lowest += 1

        if epochs_without_new_lowest >= 25:
            break

        print(this_epoch_train_loss, this_epoch_valid_loss)

        scheduler.step(this_epoch_valid_loss)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-n', '--num_frames', type=int, default=200)
    parser.add_argument('-p', '--permutation', type=int, nargs='+', default=[0,1,2,3,4])
    parser.add_argument('-s', '--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.feature_type, args.num_frames,args.permutation, args.seed)
