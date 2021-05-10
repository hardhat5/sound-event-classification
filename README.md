# Sound Event Classification

This repository contains code for various sound event classification systems implemented on two datasets - (a) DCASE 2019 Task 5 and (b) a subset of Audioset.

## Organization
Each dataset has its own folder and contains the following subfolders:
1. augmentation: package which contains code for various data augmentations
2. data: contains raw audio files and generated features
3. metadata: contains metadata required to parse the dataset
4. models: contains saved weights
Other than these subfolders, each dataset folder contains a file `statistics.py` to evaluate the channel wise mean and standard deviation for the various features. 

## Datasets

## Reproducing the results
1. Clone the repository
2. Generate the required feature using the following<br/>
`python compute_logmel.py ./dcase/data/train ./dcase/data/logmelspec`<br/>
 `python compute_gammatone.py ./dcase/data/train ./dcase/data/gammatone`<br/>
`python compute_cqt.py ./dcase/data/train ./dcase/data/cqt`<br/>
For audioset, replace "dcase" in the above commands with "audioset". The audio files can be stored anywhere but the features must be stored in the subfolders mentioned above.
3. Evaluate the channel wise mean and standard deviation for the features using `statistics.py`. For DCASE, only two arguments are required - feature type and number of frequency bins. <br/>
`python statistics.py -f logmelspec -n 64` <br/>
For Audioset, an additional parameter is required to specify the training, validation and testing folds. For training on folds 0,1 and 2, validating on 3 and testing on 4, run <br/>
`python statistics.py -f logmelspec -n 64 -p 0 1 2 3 4 <br/>
 
