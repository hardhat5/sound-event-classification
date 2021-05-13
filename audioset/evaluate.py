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
from sklearn.metrics import classification_report
import argparse
from utils import AudioDataset, Task5Model

from augmentation.SpecTransforms import ResizeSpectrogram, TimeMask, FrequencyMask, RandomCycle
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

target_names = ['breaking', 'chatter', 'crying_sobbing', 'emergency_vehicle', 'explosion', 'gunshot_gunfire', 'motor_vehicle_road', 'screaming', 'siren', 'others']

def run(feature_type, num_frames, perm):
    
    folds = []
    for i in range(5):
        folds.append(pd.read_csv('./metadata/split/fold_{}.txt'.format(i), delimiter=" ", header=None))

    train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
    valid_df = folds[perm[3]]
    test_df = folds[perm[4]]

    # Create the datasets and the dataloaders

    test_dataset = AudioDataset(test_df, feature_type=feature_type, perm=perm, resize = num_frames)
    test_loader = DataLoader(test_dataset, 16, shuffle=False, num_workers =2)

    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    
    # Instantiate the model
    model = Task5Model(10).to(device)
    model.load_state_dict(torch.load('./model/model_{}_{}'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2]))))

    y_pred = []
    for sample in test_loader:
        inputs = sample['data'].to(device)
        labels = sample['labels'].to(device)

        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            for i in range(len(outputs)):
                curr = outputs[i]
                arg = torch.argmax(curr)
                y_pred.append(arg.detach().cpu())
    y_true = []

    for index, row in test_df.iterrows():
        class_name = row[0].split('-')[0]
        y_true.append(class_mapping[class_name])

    print(classification_report(y_true, y_pred, digits=4))

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-n', '--num_frames', type=int, default=200)
    parser.add_argument('-p', '--permutation', type=int, nargs='+', default = [0,1,2,3,4])
    args = parser.parse_args()
    run(args.feature_type, args.num_frames,args.permutation)
