import numpy as np 
from tqdm import tqdm
import pickle 
from glob import glob
import argparse

def run(feature_type, num_bins):

    # Load and prepare data
    with open('./metadata/metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    train_df = metadata['coarse_train']
    file_list = list(set(train_df.index.tolist()))
    
    mean = np.zeros((num_bins,))
    M2 = np.zeros((num_bins,))

    n = 0
    for file_name in tqdm(file_list):
        data = np.load('./data/{}/{}.npy'.format(feature_type, file_name))
        x = data.mean(axis=1)
        n += 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)

    variance = M2/(n - 1)
    stdev = np.sqrt(variance)

    np.save('./data/statistics/channel_means_{}'.format(feature_type), mean)
    np.save('./data/statistics/channel_stds_{}'.format(feature_type), stdev)

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-n', '--num_bins', type=int, default=128)
    args = parser.parse_args()
    run(args.feature_type, args.num_bins)

