import os
import sys
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from glob import glob
from tqdm import tqdm
import argparse

num_cores = 9

def compute_cqt(file_name, outdir, label=''):
   
    audio, sample_rate = librosa.load(file_name)
    try:
        cqt = librosa.cqt(y=audio, sr=22050, n_bins=80)
        
        cqt = librosa.core.power_to_db(np.abs(cqt)**2)
        np.save(outdir + label + os.path.basename(file_name) + '.npy', cqt)
    except ValueError:
        print(file_name)

# Training and validation data
def main(input_path, output_path):
    file_list = glob(input_path + '/*.wav')
    os.makedirs(output_path, exist_ok=True)
    _ = Parallel(n_jobs=num_cores)(
            delayed(lambda x: compute_cqt(x, output_path +'/'))(x)
            for x in tqdm(file_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input and Output Paths')
    parser.add_argument('input_path', type=str, help="Specifies directory of audio files")
    parser.add_argument('output_path', type=str, help="Specifies directory for generated spectrograms")
    args = parser.parse_args()
    
    main(args.input_path, args.output_path)