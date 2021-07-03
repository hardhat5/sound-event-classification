# Sound Event Classification

This repository contains code for various sound event classification systems implemented on two datasets - (a) DCASE 2019 Task 5 and (b) a subset of Audioset.

## Problem statement
Label the presence or absence of certain sound events in a given audio clip using weakly labeled training data. Only the presence/absence of a sound event needs to be labeled, the onset and offset times of the events are not considered. As the final system would be deployed on telephony audio, low quality 8kHz data is simulated to test the approaches on telephony data.

## Datasets
The following datasets have been used for this project. Instructions to reproduce the results for each dataset are given in the README files in their respective folders.
### A. DCASE
This is the Urban Sound Tagging dataset from DCASE 2019 Task 5. It contains one training split and one validation split.
### B. Audioset
This is a subset of Audioset containing 10 classes. It is split into 5 different folds for 5-fold cross validation.

## Implemented Approaches
1. DCASE 2015 Task 5 best performing system
2. Feature combination
3. Temporal Spectral Attention
4. Knowledge distillation for better performance on telephony data 


