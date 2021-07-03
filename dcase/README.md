# DCASE 2015 Task 5
This folder contains code to reproduce the DCASE 2015 Task 5 best performing system. This differs from the official code in the following ways:
1. The system can be evaluated on 3 different features (log mel, gammatone and CQT).
2. SpecAugment is used instead of random erasing.
3. A feature combination system is added to aggregate the predictions from the different features.

## Reproducing the results
To reproduce the results, first clone this repository. Then, follow the steps below. 
### 1. Generating the features
Generate the required type of feature using the scripts in the parent folder <br/>
`python compute_<feature_type>.py <input_path> <output_path>`<br/>
Replace `<feature_type>` with one of `logmelspec`, `cqt`, `gammatone`. Make sure the output path is  `./dcase/data/<feature_type>`.

### 2. Evaluating channel wise mean and standard deviation
Evaluate the channel wise mean and standard deviation for the features using `statistics.py`. Two arguments are required: feature type and number of frequency bins. <br/>
`python statistics.py -f <feature_type> -n <num_bins>` <br/>

### 3. Parse the data
Run `parse_dcase.py`. 

### 4. Training
The `train.py` file takes in 2 arguments: feature type and number of time frames. Time frames refers to the length of the time axis of the logmel features. Random seed can be set using the optional argument. For example, to train logmel with 635 time frames, run <br/>
`python train.py -f logmelspec -n 635 --seed 42` <br/>

### 5. Validating
For validation, run `evaluate.py` with the same arguments as above but without the random seed argument.
