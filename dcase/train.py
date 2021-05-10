import pickle
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

from albumentations import Compose, ShiftScaleRotate, GridDistortion
from albumentations.pytorch import ToTensor
import random
from utils import AudioDataset, Task5Model , mixup_data

from augmentation.SpecTransforms import TimeMask, FrequencyMask

def run(feature_type, num_frames, seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load and prepare data
    with open('./metadata/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    train_df = metadata['coarse_train']
    valid_df = metadata['coarse_test']

    # Define the data augmentation transformations
    albumentations_transform = Compose([
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
        GridDistortion(),
        ToTensor()
    ])

    spec_transforms = transforms.Compose([
        TimeMask(), 
        FrequencyMask(), 
    ])

    train_dataset = AudioDataset(train_df, feature_type=feature_type, spec_transform=spec_transforms, image_transform=albumentations_transform, resize=num_frames)
    valid_dataset = AudioDataset(valid_df, feature_type=feature_type, resize=num_frames)

    val_loader = DataLoader(valid_dataset, 8, shuffle=False)
    train_loader = DataLoader(train_dataset, 8, shuffle=True)

    # Define the device to be used
    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)

    # Instantiate the model
    model = Task5Model(8).to(device)
    print(model)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()

    epochs = 100
    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0

    for i in range(epochs):
        print('Epoch: ', i)

        this_epoch_train_loss = 0
        for sample in train_loader:

            data, labels = sample['data'].to(device), sample['labels'].to(device)
            data, y_a, y_b, lam = mixup_data(data, labels, 1.0)
            labels = lam*y_a + (1-lam)*y_b

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(data)
                # calculate loss for each set of annotations
                # loss = criterion(outputs, labels)
                loss_0 = criterion(outputs, labels[:, 0, :]) 
                loss_1 = criterion(outputs, labels[:, 1, :]) 
                loss_2 = criterion(outputs, labels[:, 2, :]) 
                loss = (loss_0.sum() + loss_1.sum() + loss_2.sum())/3
                loss.backward()
                optimizer.step()
                this_epoch_train_loss += loss.detach().cpu().numpy()

        this_epoch_valid_loss = 0
        for sample in val_loader:
            data, labels = sample['data'].to(device), sample['labels'].to(device)
            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(data)
                # loss = criterion(outputs, labels)
                loss_0 = criterion(outputs, labels[:, 0, :]) 
                loss_1 = criterion(outputs, labels[:, 1, :]) 
                loss_2 = criterion(outputs, labels[:, 2, :]) 
                loss = (loss_0.sum() + loss_1.sum() + loss_2.sum())/3
                this_epoch_valid_loss += loss.detach().cpu().numpy()

        this_epoch_train_loss /= len(train_loader)
        this_epoch_valid_loss /= len(val_loader)

        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss:
            lowest_val_loss = this_epoch_valid_loss
            torch.save(model.state_dict(), './models/model_{}'.format(feature_type))
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
    parser.add_argument('-n', '--num_frames', type=int, default=635)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.feature_type, args.num_frames,args.seed)
