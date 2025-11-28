"""
Author: Anurag Kumar
Created on: 2025-09-07
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

    trials = {f'S{i+1}':{} for i in range(NUM_SUBJECTS)}
    for i in tqdm(range(NUM_SUBJECTS)):
        subj = f"S{i+1}"
        matfile = f"{subj}.mat"
        data = loadmat(os.path.join(ARGS.root, matfile))
        for k, trial in enumerate(data['trials']):
            if k+1 not in trials[subj]:
                trials[subj][k+1] = {}
            
            trials[subj][k+1]['direction'] = trial['attended_ear']
            trials[subj][k+1]['experiment'] = trial['experiment']
            trials[subj][k+1]['attended_track'] = trial['attended_track']
            trials[subj][k+1]['stimuli'] = trial['stimuli']
            trials[subj][k+1]['subject'] = subj

    #Save csv file
    csv_path = os.path.join(ARGS.root, 'trials.csv')
    data_dict = {
        'subject':[],
        'trial':[],
        'direction':[],
        'attended_track':[],
        'experiment':[],
        'stimuli':[],
    }
    for subj in trials:
        for t_id in range(NUM_TRIALS):
            data_dict['subject'].append(subj)
            data_dict['trial'].append(t_id+1)
            data_dict['direction'].append(trials[subj][t_id+1]['direction'])
            data_dict['attended_track'].append(trials[subj][t_id+1]['attended_track'])
            data_dict['experiment'].append(trials[subj][t_id+1]['experiment'])
            files = []
            for fname in trials[subj][t_id+1]['stimuli']:
                files.append(fname[0])
            files = " ".join(files)
            data_dict['stimuli'].append(files)
    

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

