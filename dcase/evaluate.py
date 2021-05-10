import pickle
import numpy as np
import pandas as pd
from metrics import evaluate, micro_averaged_auprc, macro_averaged_auprc

import torch
from torch.utils.data import Dataset, DataLoader
from albumentations import Compose
from albumentations.pytorch import ToTensor
from utils import Task5Model, AudioDataset

import argparse

with open('./metadata/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

def run(feature_type, num_frames):

    validate_files = []
    valid_df = metadata["coarse_test"]

    valid_dataset = AudioDataset(valid_df, 'logmelspec', resize=num_frames)
    valid_loader = DataLoader(valid_dataset, 8, shuffle=False)

    cuda = True
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)

    model = Task5Model(8).to(device)
    model.load_state_dict(torch.load('./models/model_{}'.format(feature_type)))

    preds = []
    for sample in valid_loader:
            inputs = sample['data'].to(device)
            file_name = sample['file_name']

            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)
                for j in range(outputs.shape[0]):
                    preds.append(outputs[j,:].detach().cpu().numpy())
                    validate_files.append(file_name[j])

    preds = np.array(preds)
    output_df = pd.DataFrame(
        preds, columns=[
            '1_engine', '2_machinery-impact', '3_non-machinery-impact',
            '4_powered-saw', '5_alert-signal', '6_music', '7_human-voice', '8_dog'])

    output_df['audio_filename'] = pd.Series(
        validate_files,
        index=output_df.index)

    output_df.to_csv('./models/pred.csv', index=False)

    mode = "coarse"
    df_dict = evaluate('./models/pred.csv',
                       './metadata/annotations-dev.csv',
                       './metadata/dcase-ust-taxonomy.yaml',
                       "coarse")

    micro_auprc, eval_df = micro_averaged_auprc(df_dict, return_df=True)
    macro_auprc, class_auprc = macro_averaged_auprc(df_dict, return_classwise=True)
    # Get index of first threshold that is at least 0.5
    thresh_0pt5_idx = (eval_df['threshold'] >= 0.5).nonzero()[0][0]
    print("{} level evaluation:".format(mode.capitalize()))
    print("======================")
    print(" * Micro AUPRC:           {}".format(micro_auprc))
    print(" * Micro F1-score (@0.5): {}".format(eval_df["F"][thresh_0pt5_idx]))
    print(" * Macro AUPRC:           {}".format(macro_auprc))
    print(" * Coarse Tag AUPRC:")
    
    for coarse_id, auprc in class_auprc.items():
        print("      - {}: {}".format(coarse_id, auprc))

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-f', '--feature_type', type=str, default='logmelspec')
    parser.add_argument('-n', '--num_frames', type=int, default=635)
    args = parser.parse_args()
    run(args.feature_type, args.num_frames)
