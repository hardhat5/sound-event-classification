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

# def specmix(input1, input2):
    
#     batch_size = input1.shape[0]
#     num_bins = input1.shape[2]
#     num_frames = input1.shape[3]
    
#     time_width = 40
#     freq_width = 64

#     beta = []
    
#     for i in range(batch_size):
    
#         time_index = random.randint(0, num_frames-time_width)
#         freq_index = random.randint(0, num_bins-freq_width)
        
#         input1[i,:,freq_index:freq_index+freq_width, :] = input2[i,:,freq_index:freq_index+freq_width, :]
#         input1[i,:,:,time_index:time_index+time_width] = input2[i,:,:,time_index:time_index+time_width]

#         beta.append((time_width*num_bins + freq_width*num_frames - time_width*freq_width)/(num_bins*num_frames)) 
#         beta.append(0.5)

#     beta = torch.FloatTensor(beta)
#     return input1, beta

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

    # train_dataset = AudioDataset3(train_df, path = './data/logmelspec/', resize = resize,  spec_transform = spec_transforms)
    valid_dataset = AudioDataset3(valid_df, path = './data/logmelspec/', resize = resize)

    val_loader = DataLoader(valid_dataset, 16, shuffle=False, num_workers = 2)
    train_loader = DataLoader(train_dataset, 16, shuffle=True, num_workers = 2)

    # Instantiate the model
    model = Task5Model(9).to(device)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    epochs = 50
    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0

    # sample1 = next(iter(train_loader_1))
    # sample2 = next(iter(train_loader_2))

    # input1, input2 = sample1['data'], sample2['data']
    # label1, label2 = sample1['labels'].to(device), sample2['labels'].to(device)
    # np.save('first.npy', input1[4,:,:,:].squeeze().numpy())
    # np.save('second.npy', input2[4,:,:,:].squeeze().numpy())


    # plt.figure()
    # # mixup the inputs
    # inputs, beta = specmix(input1, input2)
    # print(beta)
    # print(type(inputs))
    # np.save('mixed.npy', inputs[4,:,:,:].squeeze().numpy())

    for i in range(epochs):
        print('Epoch: ', i)

        this_epoch_train_loss = 0
        for sample in tqdm(train_loader):

            inputs, labels = sample['data'].to(device), sample['labels'].to(device)

            # mixup the inputs and labels
            inputs, y_a, y_b, beta = specmix(inputs, labels)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(inputs.to(device))
                # Mixup loss
                loss = beta * criterion(outputs, y_a) + (1 - beta) * criterion(outputs, y_b)
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