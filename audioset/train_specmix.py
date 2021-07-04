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
import matplotlib.pyplot as plt
import time
from utils import prepare_data, AudioDataset, Task5Model, AudioDataset3

from augmentation.SpecTransforms import MelSpectrogram, ResizeSpectrogram, TimeMask, FrequencyMask, RandomCycle
from augmentation.AudioTransforms import ResizeWaveform, NormalizeWaveform, TimeStretching, PitchShifting

def specmix(x, y):
        
    num_bins = x.shape[2]
    num_frames = x.shape[3]

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    y_a, y_b = y, y[index]

    time_width = 40
    freq_width = 30

    beta = []

    time_index = random.randint(0, num_frames-time_width)
    freq_index = random.randint(0, num_bins-freq_width)
    
    x[:,:,freq_index:freq_index+freq_width, :] = x[index,:,freq_index:freq_index+freq_width, :]
    x[:,:,:,time_index:time_index+time_width] = x[index,:,:,time_index:time_index+time_width]

    beta = 1 - (time_width*num_bins + freq_width*num_frames - time_width*freq_width)/(num_bins*num_frames)
    
    return x, y_a, y_b, beta

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

    epochs = 50
    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0

    for i in range(epochs):
        print('Epoch: ', i)

        this_epoch_train_loss = 0
        for sample in train_loader:

            inputs, labels = sample['data'].to(device), sample['labels'].to(device)

            # mixup the inputs and labels
            inputs, y_a, y_b, lam = specmix(inputs, labels)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(inputs.to(device))
                # Mixup loss
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
                loss.backward()
                optimizer.step()
                this_epoch_train_loss += loss.detach().cpu().numpy()

        this_epoch_valid_loss = 0
        for sample in tqdm(val_loader):
            inputs = sample['data']
            labels = sample['labels']
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels) 
                loss = loss.sum()
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_df)
        this_epoch_valid_loss /= len(valid_df)

        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save(model.state_dict(),'./model/model_{}_{}'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
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