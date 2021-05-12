import os
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm, trange
from sklearn import preprocessing
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 

from utils import Task5Model, AudioDataset
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, average_precision_score

class_mapping = {}
class_mapping['breaking'] = 0
class_mapping['chatter'] = 1
class_mapping['crying_sobbing'] = 2
class_mapping['emergency_vehicle'] = 3
class_mapping['explosion'] = 4
class_mapping['gunshot_gunfire'] = 5
class_mapping['motor_vehicle_(road)'] = 6
class_mapping['motor_vehicle_road'] = 6
class_mapping['screaming'] = 7
class_mapping['siren'] = 8

class FeatureFusion(nn.Module):

    def __init__(self, l_weight, c_weight, g_weight, num_classes, perm_name):

        super().__init__()
        
        self.logmel = Task5Model(num_classes)
        self.cqt = Task5Model(num_classes)
        self.gammatone = Task5Model(num_classes)
        self.l_weight = l_weight
        self.c_weight = c_weight
        self.g_weight = g_weight
        self.perm_name = perm_name
        self.init_weights()
        
    def init_weights(self):
        self.logmel.load_state_dict(torch.load('./model/model_logmelspec_' + self.perm_name))
        self.cqt.load_state_dict(torch.load('./model/model_cqt_' + self.perm_name))
        self.gammatone.load_state_dict(torch.load('./model/model_gammatone_' + self.perm_name))
        print('Loaded')
    
    def forward(self, x1, x2, x3):
        x1 = self.logmel(x1)
        x2 = self.cqt(x2)
        x3 = self.gammatone(x3)
        
        x1 = x1 * self.l_weight
        x2 = x2 * self.c_weight
        x3 = x3 * self.g_weight
        
        x = x1 + x2 + x3
  
        return x

def load_dict(name):
    with open('./weights/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_model_features(test_df, feature_type, perm, perm_name, num_frames):

    # Create the datasets and the dataloaders
    test_dataset = AudioDataset(test_df, feature_type, perm=perm, resize=num_frames)
    test_loader = DataLoader(test_dataset, 16, shuffle=False, num_workers=2)

    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    model_name = './model/model_' + feature_type + '_' + perm_name

    model = Task5Model(10).to(device)
    model.load_state_dict(torch.load(model_name))

    # Get features
    print(f'Obtaining {feature_type} features')
    test_features = []
    for sample in tqdm(test_loader):
        inputs = sample['data'].to(device)
        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            test_features.append(outputs.detach().cpu().numpy())
    test_features = np.concatenate(test_features, axis=0)
    
    return test_features

def main(perm):
    
    #os.makedirs('../result', exist_ok=True)
    perm_name = str(perm[0]) + str(perm[1]) + str(perm[2])
    folds = []
    fold_files = sorted(glob('./metadata/split/*.txt'))
    folds = [pd.read_csv(x, delimiter=" ", header=None) for x in fold_files]
    
    print(f'\nEvaluating Fold {perm_name}:')
    print(f'Testing on {perm[-1]}')
    
    test_df = folds[perm[-1]]
    logmel_out = get_model_features(test_df, 'logmelspec', perm, perm_name, 200)
    cqt_out = get_model_features(test_df, 'cqt', perm, perm_name, 431)
    gamma_out = get_model_features(test_df, 'gammatone', perm, perm_name, 496)
    
    l_weight = np.array(list(load_dict('logmelspec_valid_weights_' + perm_name).values()))
    c_weight = np.array(list(load_dict('cqt_valid_weights_' + perm_name).values()))
    g_weight = np.array(list(load_dict('gammatone_valid_weights_' + perm_name).values()))
    
    logmel_out *= l_weight
    cqt_out *= c_weight
    gamma_out *= g_weight
    
    pred = logmel_out + cqt_out + gamma_out
    y_pred = list(np.argmax(pred, axis=1))
    
    y_true = []
    for index, row in test_df.iterrows():
        class_name = row[0].split('-')[0]
        y_true.append(class_mapping[class_name])

    print(classification_report(y_true, y_pred, digits=4))
    
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Feature Combination')
    parser.add_argument('--permutation', '--arg', nargs='+', type=int)
    args = parser.parse_args()
    
    main(args.permutation)

