# Sound Event Classification

This repository contains code for various sound event classification systems implemented on two datasets - (a) DCASE 2019 Task 5 and (b) a subset of Audioset.

## Organization
Each dataset has its own folder and contains the following subfolders:
1. augmentation: package which contains code for various data augmentations
2. data: contains raw audio files and generated features
3. metadata: contains metadata required to parse the dataset
4. models: contains saved weights
Other than these subfolders, each dataset folder contains a script `statistics.py` to evaluate the channel wise mean and standard deviation for the various features. 

## Datasets
### A. DCASE
This is the Urban Sound Tagging dataset from DCASE 2019 Task 5. It contains one training split and one validation split. 
### B. Audioset
This is a subset of Audioset containing 10 classes. It is split into 5 different folds for 5-fold cross validation. 

## Reproducing the results
To reproduce the results, first clone this repository. Then, follow the steps below. 
### 1. Generating the features
Generate the required type of feature using the following <br/>
`python compute_<feature_type>.py <input_path> <output_path>
Replace `<feature_type>` with one of `logmelspec`, `cqt`, `gammatone`. The output path has to be `./<dataset>/data/<feature_type>` where `<dataset>` is one of `dcase` or `audioset`. 

### 2. Evaluating channel wise mean and standard deviation
Evaluate the channel wise mean and standard deviation for the features using `statistics.py`. 
#### DCASE
Only two arguments are required: feature type and number of frequency bins. <br/>
`python statistics.py -f logmelspec -n 64` <br/>
#### Audioset
An additional parameter is required to specify the training, validation and testing folds. For training on folds 0, 1 and 2, validating on 3 and testing on 4, run <br/>
`python statistics.py -f logmelspec -n 64 -p 0 1 2 3 4` <br/>
### 3. Parse the data
To parse the DCASE data, run `parse_dcase.py`. No such step is required for Audioset.
### 4. Training
#### DCASE
The `train.py` file for DCASE takes in 3 arguments: feature type, number of time frames and random seed. To train logmel, run <br/>
`python train.py -f logmelspec -n 635 --seed 42` <br/>
#### Audioset
Other than the three arguments above, the `train.py` file for Audioset takes in an additional argument to specify the training, validation and testing folds. For training on folds 0, 1 and 2, validating on 3 and testing on 4, run <br/>
`python train.py -f logmelspec -n 635 -p 0 1 2 3 4` <br/>
### 5. Validating
For validation, run the same commands as above but without the random seed argument. 
