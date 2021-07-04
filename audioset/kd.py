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
from utils import AudioDataset, Task5Model
from rangerlars import RangerLars
from torch.autograd import Variable

from augmentation.SpecTransforms import ResizeSpectrogram, TimeMask, FrequencyMask, RandomCycle

class KDDataset(Dataset):

    def __init__(self, df, df_8k, feature_type="logmelspec", perm=[0,1,2,3,4], spec_transform=None, image_transform=None, resize=None):
        
        self.dataset = AudioDataset(df, "logmelspec", perm, spec_transform, image_transform, 1003)
        self.dataset_8k = AudioDataset(df_8k, "logmelspec_8k", perm, spec_transform, image_transform, 1003)
        print(len(self.dataset), len(self.dataset_8k))    

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        sample = self.dataset[idx]
        sample_8k = self.dataset_8k[idx]
        data = {}
        data['data'] = sample['data']
        data['data_8k'] = sample_8k['data']
        data['labels'] = sample['labels']
        data['file_name'] = sample['file_name']

        return data

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):

    KD_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def run(feature_type, num_frames, perm, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    folds = []
    folds_8k = []
    for i in range(5):
        folds.append(pd.read_csv('./metadata/split_new/fold_{}.txt'.format(i), delimiter=" ", header=None))
        folds_8k.append(pd.read_csv('./metadata/split_8k/fold_{}.txt'.format(i), delimiter=" ", header=None))
    
    train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
    valid_df = folds[perm[3]]
    test_df = folds[perm[4]]

    train_df_8k = pd.concat([folds_8k[perm[0]], folds_8k[perm[1]], folds_8k[perm[2]]])
    valid_df_8k = folds_8k[perm[3]]
    test_df_8k = folds_8k[perm[4]]

    for i in range(5): print(len(folds[i]))
    for i in range(5): print(len(folds_8k[i]))


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
    train_dataset = KDDataset(train_df, train_df_8k, 
        feature_type=feature_type,
        perm=perm,
        resize = num_frames,
        image_transform = albumentations_transform,
        spec_transform=spec_transforms)

    valid_dataset = KDDataset(valid_df, valid_df_8k, feature_type=feature_type, perm=perm, resize = num_frames)

    val_loader = DataLoader(valid_dataset, 64, shuffle=False, num_workers =2)
    train_loader = DataLoader(train_dataset, 64, shuffle=True, num_workers = 2)

    # Define the device to be used
    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    
    print('Device: ', device)
    # Instantiate the model
    teacher = Task5Model(10).to(device)
    student = Task5Model(10).to(device)

    teacher.load_state_dict(torch.load('./model/model_{}_{}'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2]))))
    for param in teacher.parameters():
        param.requires_grad = False

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    #optimizer = RangerLars(params=model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.CrossEntropyLoss()

    epochs = 100
    alpha, T = 0.01, 2
    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0
    
    for i in range(epochs):
        print('Epoch: ', i)

        this_epoch_train_loss = 0
        for sample in train_loader:

            inputs = sample['data'].to(device)
            inputs_8k = sample['data_8k'].to(device)
            label = sample['labels'].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                student = student.train()
                outputs = student(inputs_8k)
                teacher_outputs = teacher(inputs).detach()
                loss = loss_fn_kd(outputs, label, teacher_outputs, 0.01, T)
                loss.backward()
                optimizer.step()
                this_epoch_train_loss += loss.detach().cpu().numpy()

        this_epoch_valid_loss = 0
        for sample in val_loader:
            inputs = sample['data_8k'].to(device)
            labels = sample['labels'].to(device)
            with torch.set_grad_enabled(False):
                student = student.eval()
                outputs = student(inputs)
                loss = criterion(outputs, labels) 
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_df)
        this_epoch_valid_loss /= len(valid_df)

        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save(student.state_dict(), './model/model_kd_{}_8k_{}'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
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
    parser.add_argument('-n', '--num_frames', type=int, default=1003)
    parser.add_argument('-p', '--permutation', type=int, nargs='+', default=[1,2,3,4,0])
    parser.add_argument('-s', '--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.feature_type, args.num_frames,args.permutation, args.seed)
