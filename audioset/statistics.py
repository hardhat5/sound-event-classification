import numpy as np 
from tqdm import tqdm
import pickle 
from glob import glob
import argparse
import pandas as pd
import os

def run(feature_type, num_bins, perm):

    # Load and prepare data
    folds = []
    for i in range(5):
        folds.append(pd.read_csv('./metadata/split/fold_{}.txt'.format(i), delimiter=" ", header=None))
    
    train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])    
    file_list = train_df[0].unique()
    
    mean = np.zeros((num_bins,))
    M2 = np.zeros((num_bins,))

    n = 0
    for file_name in tqdm(file_list):
        data = np.load('./data/{}/{}.wav.npy'.format(feature_type, file_name))
        x = data.mean(axis=1)
        n += 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)

    variance = M2/(n - 1)
    stdev = np.sqrt(variance)

    os.makedirs('./data/statistics', exist_ok=True)
    np.save('./data/statistics/channel_means_{}_{}'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2])), mean)
    np.save('./data/statistics/channel_stds_{}_{}'.format(feature_type, str(perm[0])+str(perm[1])+str(perm[2])), stdev)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-n', '--num_bins', type=int, default=128)
    parser.add_argument('-p', '--permutation', type=int, nargs='+', required=True)
    args = parser.parse_args()
    run(args.feature_type, args.num_bins, args.permutation)

