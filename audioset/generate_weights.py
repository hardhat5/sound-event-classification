import os
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
from sklearn import preprocessing
from collections import defaultdict
from sklearn.metrics import average_precision_score

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from SpecTransforms import ResizeSpectrogram
from utils import Task5Model, prepare_data, AudioDataset

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

def save_dict(obj, name):
    with open('weights/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def get_auprc(feature_type, num_frames, perm):
    perm_name = str(perm[0]) + str(perm[1]) + str(perm[2])
    folds = []
    fold_files = sorted(glob('./metadata/split/*.txt'))
    folds = [pd.read_csv(x, delimiter=" ", header=None) for x in fold_files]
    
    perm_name = str(perm[0]) + str(perm[1]) + str(perm[2])
    print(f'Evaluating Fold {perm_name}:')
    print(f'Validating on {perm[3]}')
    
    test_df = folds[perm[3]]
    test_files = [x + '.wav' for x in test_df[0].tolist()]
    
    test_dataset = AudioDataset(test_df, feature_type, perm=perm, resize=num_frames)
    test_loader = DataLoader(test_dataset, 16, shuffle=False, num_workers=2)
    
    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    
    model = Task5Model(10).to(device)
    model_name = './model/model_' + feature_type + '_'  + perm_name
    model.load_state_dict(torch.load(model_name))
    
    all_preds = []
    for _ in range(10):
        preds = []
        for sample in tqdm(test_loader):
                inputs = sample['data'].to(device)
                with torch.set_grad_enabled(False):
                    model = model.eval()
                    outputs = model(inputs)
                    preds.append(outputs.detach().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        preds = np.array(preds, dtype=np.float128)
        preds = (1 / (1 + np.exp(-preds)))
        all_preds.append(preds)
    tmp = all_preds[0]
    for x in all_preds[1:]:
        tmp += x
    tmp = tmp / 10
    preds = tmp
    
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(class_mapping.values()))
    lb_name_mapping = dict(zip(lb.classes_, lb.transform(lb.classes_)))
    
    gt = [class_mapping[x] for x in test_df[2].tolist()]
    gt = lb.transform(gt)
        
    auprc = average_precision_score(gt, preds, average=None)

    return auprc
    
def main(perm):
    os.makedirs('weights', exist_ok=True)
    logmelspec_auprc = get_auprc('logmelspec', 200, perm)
    cqt_auprc = get_auprc('cqt', 431, perm)
    gammatone_auprc = get_auprc('gammatone', 496, perm)
    
    logmelspec_weights = {}
    cqt_weights = {}
    gammatone_weights = {}
    perm_name = str(perm[0]) + str(perm[1]) + str(perm[2])
    key = 1
    for score_1, score_2, score_3 in zip(logmelspec_auprc, cqt_auprc, gammatone_auprc):
        total = score_1 + score_2 + score_3
        logmelspec_weights[key] = score_1/total
        cqt_weights[key] = score_2/total
        gammatone_weights[key] = score_3/total
        key += 1
    
    save_dict(logmelspec_weights, 'logmelspec_valid_weights_' + perm_name)
    save_dict(cqt_weights, 'cqt_valid_weights_' + perm_name)
    save_dict(gammatone_weights, 'gammatone_valid_weights_' + perm_name)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get AUPRC Weights')
    parser.add_argument('-p', '--permutation', '--arg', nargs='+', type=int, default=[0,1,2,3,4])
    args = parser.parse_args()
    
    main(args.permutation)
