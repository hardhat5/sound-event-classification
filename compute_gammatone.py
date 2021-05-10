from __future__ import division
import os.path

import scipy.constants
import scipy.io.wavfile
import matplotlib.pyplot
import matplotlib.ticker

import os
import re
import sys
import librosa
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

from gammatone.filters import erb_point
import gammatone.gtgram
import gammatone.fftweight

num_cores = 9

class ERBFormatter(matplotlib.ticker.EngFormatter):
    def __init__(self, low_freq, high_freq, *args, **kwargs):
        self.low_freq = low_freq
        self.high_freq = high_freq
        super().__init__(*args, **kwargs)

    def _erb_axis_scale(self, fraction):
        return erb_point(self.low_freq, self.high_freq, fraction)

    def __call__(self, val, pos=None):
        newval = self._erb_axis_scale(val)
        return super().__call__(newval, pos)

def gtgram_plot(save_dir, wav_file, gtgram_function, 
                x, fs, window_time, hop_time,
                channels, f_min, label, imshow_args=None):
    
    m = re.search('/[a-zA-Z](.+?).wav', wav_file)
    if m:
      file_name = m.group(0)
    file_name = file_name.split('/')[2]
    save_path = save_dir + label + file_name + '.npy'
  
    try:
        gtg = gtgram_function(x, fs, window_time, hop_time, channels, f_min)
        gtg = librosa.core.power_to_db(gtg)
        np.save(save_path, gtg)
    except ValueError:
        print(f'Unable to render {file_name}')
    
def compute_gammatone(wav_file, save_dir, label='', function=gammatone.fftweight.fft_gtgram):
    data, samplerate = librosa.load(wav_file, sr=22050)
    signal = data

    # Default gammatone-based spectrogram parameters
    twin = 0.04
    thop = twin / 2
    channels = 64
    fmin = 20

    gtgram_plot(save_dir, wav_file, function,
                signal, samplerate,
                twin, thop, channels, fmin, label)

def main():
    file_list = glob('../audio/*.wav')
    os.makedirs('../gammatone', exist_ok=True)
    _ = Parallel(n_jobs=num_cores)(
            delayed(lambda x: compute_gammatone(x, '../gammatone/'))(x)
            for x in tqdm(file_list))

if __name__ == '__main__':
    main()