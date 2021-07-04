# Audioset

This folder contains code to implement several sound event classification systems on a subset of Audioset and the corresponding telephony data. It contains the following implementations:
1. DCASE 2019 Task 5 best performing system
2. Feature combination
3. Temporal-spectral attention
4. Knowledge distillation for telephony data

## Reproducing the results
To reproduce the results, first clone this repository. Then, follow the steps below. 
### 1. Generating the features
Generate the required type of feature using the scripts in the parent folder: <br/>
`python compute_<feature_type>.py <input_path> <output_path>`<br/>
Replace `<feature_type>` with one of `logmelspec`, `cqt`, `gammatone`. Make sure the output path is  `./audioset/data/<feature_type>`. 

### 2. Evaluating channel wise mean and standard deviation
Evaluate the channel wise mean and standard deviation for the features using `statistics.py`. Three parameters are required for this script - feature type, number of frequency bins and training permutation. For example, for training on folds 0, 1 and 2, validating on 3 and testing on 4 with logmel features and 64 frequency bins, run <br/>
`python statistics.py -f logmelspec -n 64 -p 0 1 2 3 4` <br/>

### 3. Training

The `train.py` file takes in 3 arguments: feature type, number of time frames and training permutation. To train logmel with 635 time frames and training on folds 0, 1 and 2, validating on 3 and testing on 4, run <br/>
`python train.py -f logmelspec -n 635 -p 0 1 2 3 4` <br/>
Optionally, the random seed can be set using the seed argument (eg. `--seed 42`).  

### 5. Validating
For validation, run `evaluate.py` with the same arguments as above but without the random seed argument.

### 6. Feature Fusion
In order to perform feature fusion, refer to the first point to generate  `logmelspec`, `cqt` and  `gammatone` features and then train their respective models. Next, to generate the weights of each feature, run <br/>
`python generate_weights.py -p 0 1 2 3 4` <br/>

Finally, run <br/>
`python feature_fusion.py -p 0 1 2 3 4` <br/>

### 7. Mixup variants
The code for two mixup variants (specmix and manifold mixup) is included in this folder. To evaluate specmix and manifold mixup, run `train_specmix.py` and `train_manifold_mixup.py`. The specmix function can be found in the `train_specmix.py` file. The modified MobileNetV2 model for manifold mixup can be found in `mixup_model.py`. The arguments to be passed are the same as step 3. 

### 8. Telephony data
To train on 8kHz telephony data, follow the same steps as above with the following differences:
(a) The output folders for telephony features should be 
`./audioset/data/<feature_type>_8k`
(b) Simiarly, replace `<feature_type>` in the arguments passed for evaluating mean and stdev and training and testing with `<feature_type>_8k`.
(c) Run `train_telephony.py` and `evaluate_telephony.py` instead of `train.py` and `evaluate.py`.

### 9. Knowledge distillation
 
