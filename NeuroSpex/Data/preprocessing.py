"""
Author: Anurag Kumar
Created on: 2023-11-15 12:00:00
"""

from tqdm import tqdm
import os
import numpy as np
import pickle
import pandas as pd
import argparse
from mat4py import loadmat

def main(ARGS):
    #root = "/fs/scratch/PAS2301/kumar1109/KUL-Mix"
    NUM_SUBJECTS = 16
    NUM_TRIALS = 20
    eeg_save = os.path.join(ARGS.root, 'eeg')

    os.makedirs(eeg_save, exist_ok=True)

    trials = {f'S{i+1}':{} for i in range(NUM_SUBJECTS)}
    for i in tqdm(range(NUM_SUBJECTS)):
        subj = f"S{i+1}"
        matfile = f"{subj}.mat"
        data = loadmat(os.path.join(ARGS.root, matfile))
        for k, trial in enumerate(data['trials']):
            if k+1 not in trials[subj]:
                trials[subj][k+1] = {}
            
            eeg_signal = trial['RawData']['EegData']
            e_id = f"{subj}_{k+1}.pkl"
            eeg_save_path = os.path.join(eeg_save, e_id)
            with open(eeg_save_path, 'wb') as f:
                pickle.dump(eeg_signal, f)
                
            trials[subj][k+1]['eeg'] = eeg_save_path
            trials[subj][k+1]['direction'] = trial['attended_ear']
            trials[subj][k+1]['experiment'] = trial['experiment']
            trials[subj][k+1]['attended_track'] = trial['attended_track']
            trials[subj][k+1]['stimuli'] = f"{data['trials'][0]['stimuli'][0][0]} {data['trials'][0]['stimuli'][1][0]}"
            trials[subj][k+1]['subject'] = subj

    #Save csv file
    csv_path = os.path.join(ARGS.root, 'trials.csv')
    data_dict = {
        'subject':[],
        'trial':[],
        'eeg_path':[],
        'direction':[],
        'attended_track':[],
        'experiment':[],
        'stimuli':[],
    }
    for subj in trials:
        for t_id in range(NUM_TRIALS):
            data_dict['subject'].append(subj)
            data_dict['trial'].append(t_id+1)
            data_dict['eeg_path'].append(trials[subj][t_id+1]['eeg'])
            data_dict['direction'].append(trials[subj][t_id+1]['direction'])
            data_dict['attended_track'].append(trials[subj][t_id+1]['attended_track'])
            data_dict['experiment'].append(trials[subj][t_id+1]['experiment'])
            data_dict['stimuli'].append(trials[subj][t_id+1]['stimuli'])

    df = pd.DataFrame(data_dict)
    df.to_csv(csv_path, index=False)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--root', type=str, required=True,
                        help='Path to KUL-Mix dataset root directory')
    return parser

if __name__ == "__main__":
    ARGS = args().parse_args()
    main(ARGS)

