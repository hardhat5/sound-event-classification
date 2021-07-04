import pickle
import pandas as pd
import numpy as np 
from albumentations import Compose, ShiftScaleRotate, GridDistortion
from albumentations.pytorch import ToTensor
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader    
import random
import torchvision
from augmentation.SpecTransforms import ResizeSpectrogram
from augmentation.RandomErasing import RandomErasing

class_mapping = {}
class_mapping['breaking'] = 0
class_mapping['chatter'] = 1
class_mapping['crying_sobbing'] = 2
class_mapping['emergency_vehicle'] = 3
class_mapping['explosion'] = 4
class_mapping['gunshot_gunfire'] = 5
class_mapping['motor_vehicle_road'] = 6
class_mapping['screaming'] = 7
class_mapping['siren'] = 8
class_mapping['others'] = 9

random_erasing = RandomErasing()

class AudioDataset(Dataset):

    def __init__(self, df, feature_type="logmelspec", perm=[0,1,2,3,4], spec_transform=None, image_transform=None, resize=None):
        
        self.df = df
        self.filenames = df[0].unique()
        self.feature_type = feature_type

        self.spec_transform = spec_transform
        self.image_transform = image_transform
        self.resize = ResizeSpectrogram(frames=resize)
        self.pil = transforms.ToPILImage()

        self.channel_means = np.load('./data/statistics/channel_means_{}_{}.npy'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
        self.channel_stds = np.load('./data/statistics/channel_stds_{}_{}.npy'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2])))

        self.channel_means = self.channel_means.reshape(1,-1,1)
        self.channel_stds = self.channel_stds.reshape(1,-1,1)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        
        curr = self.df.iloc[idx, :]
        file_name = curr[0]
        labels = class_mapping[file_name.split('-')[0]]

        sample = np.load('./data/' + self.feature_type + '/' + file_name + '.wav.npy')
        
        if self.resize:
            sample = self.resize(sample)

        sample = (sample-self.channel_means)/self.channel_stds
        sample = torch.Tensor(sample)

        if self.spec_transform:
            sample = self.spec_transform(sample)

        sample = sample.transpose(0,1)
        
        if self.image_transform:
            # min-max transformation
            this_min = sample.min()
            this_max = sample.max()
            sample = (sample - this_min) / (this_max - this_min)

            # apply albumentations transforms
            sample = np.array(self.pil(sample))
            sample = self.image_transform(image=sample)
            sample = sample['image']
            sample = sample[None, :, :].permute(0, 2, 1)

            # apply random erasing
            sample = random_erasing(sample.clone().detach())

            # revert min-max transformation
            sample = (sample * (this_max - this_min)) + this_min
        
        if len(sample.shape)<3:
            sample = torch.unsqueeze(sample, 0)
        
        labels = torch.LongTensor([labels]).squeeze()
        
        data = {}
        data['data'], data['labels'], data['file_name'] = sample, labels, file_name
        return data

class Task5Model(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

        self.final = nn.Sequential(
            nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
            nn.Linear(512, num_classes))

    def forward(self, x):

        x = self.bw2col(x)
        x = self.mv2.features(x)
        x = x.max(dim=-1)[0].max(dim=-1)[0]
        x = self.final(x)
        return x

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
