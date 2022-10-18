#!/usr/bin/env python
import os
import sys
from time import time
import argparse

import numpy as np
import pandas as pd
from davis2017.evaluation import DAVISEvaluation
import yaml
from utils import plot_verb_chart
default_EPIC_path = '../val_data'

time_start = time()
parser = argparse.ArgumentParser()
parser.add_argument('--EPIC_path', type=str, help='Path to the EPIC folder containing the JPEGImages, Annotations, '
                                                    'ImageSets, Annotations_unsupervised folders',
                    required=False, default=default_EPIC_path)
parser.add_argument('--yaml_root', type=str, 
                    required=False, default=f'{default_EPIC_path}/EPIC100_state_positive_val.yaml')
parser.add_argument('--set', type=str, help='Subset to evaluate the results', default='val')
parser.add_argument('--task', type=str, help='Task to evaluate the results', default='semi-supervised',
                    choices=['semi-supervised', 'unsupervised'])
parser.add_argument('--XMem_results_path', type=str, help='Path to the folder containing the sequences folders',
                    default='/home/venom/projects/XMem_evaluation/XMem_output/Sep04_09.49.56_test_0904_not_freeze_epic_25000') 
parser.add_argument('--Ours_results_path', type=str, help='Path to the folder containing the sequences folders',
                    default='/home/venom/projects/XMem_evaluation/XMem_output/Sep04_09.49.56_test_0904_not_freeze_epic_25000') 
parser.add_argument('--sequence_type', type=str, help='compute for all images or only for the second half images',
                    default='all', choices=['all', 'second_half'])
args, _ = parser.parse_known_args()
csv_name_global = f'{args.sequence_type}_global_results-{args.set}.csv'
csv_name_per_sequence = f'{args.sequence_type}_per-sequence_results-{args.set}.csv'

# Check if the method has been evaluated before, if so read the results, otherwise compute the results
csv_name_global_path = os.path.join('./', csv_name_global)

csv_name_per_sequence_path = os.path.join('./', csv_name_per_sequence)
# csv_name_per_sequence_path = '/home/venom/projects/XMem/output/shit'
# csv_name_global_path = '/home/venom/projects/XMem/output/shit/global.csv'

print(f'Evaluating sequences for the {args.task} task...')
# Create dataset and evaluate
dataset_eval = DAVISEvaluation(davis_root=args.EPIC_path, yaml_root=args.yaml_root, task=args.task, gt_set=args.set, sequences=args.sequence_type)
XMem_metrics_res = dataset_eval.evaluate(args.XMem_results_path)
XMem_J, XMem_F = XMem_metrics_res['J'], XMem_metrics_res['F']

Ours_metrics_res = dataset_eval.evaluate(args.Ours_results_path)
Ours_J, Ours_F = Ours_metrics_res['J'], Ours_metrics_res['F']

# Generate a dataframe for the per sequence results
with open(args.yaml_root, 'r') as f:
    yaml_data = yaml.safe_load(f)

seq_names = list(XMem_J['M_per_object'].keys())

narrations = []
for name in seq_names:
    name = '_'.join(name.split('_')[:-1])
    narration = yaml_data[name]['narration']
    narrations.append(narration)
seq_measures = ['narration', 'Sequence', 'J-Mean', 'F-Mean']

J_ensemble_measures = []
F_ensemble_measures = []
for name in seq_names:
    tmp_J = []
    tmp_F = []
    print(XMem_J['M_per_frame'][name])
    for frame_idx in range(len(XMem_J['M_per_frame'][name])):
        frame_J_value = np.max([XMem_J['M_per_frame'][name][frame_idx], Ours_J['M_per_frame'][name][frame_idx]])
        frame_F_value = np.max([XMem_F['M_per_frame'][name][frame_idx], Ours_F['M_per_frame'][name][frame_idx]])
        tmp_J.append(frame_J_value)
        tmp_F.append(frame_F_value)
    J_ensemble_measures.append(np.mean(tmp_J))
    F_ensemble_measures.append(np.mean(tmp_F))

table_seq = pd.DataFrame(data=list(zip(seq_names, narrations, J_ensemble_measures, F_ensemble_measures)), columns=seq_measures)

with open(csv_name_per_sequence_path, 'w') as f:
    table_seq.to_csv(f, index=False, float_format="%.3f")
print(f'Per-sequence results saved in {csv_name_per_sequence_path}')


# Generate dataframe for the general results
ensemble_g_measures = ['J&F-Mean', 'J-Mean', 'F-Mean', ]
final_mean = (np.mean(J_ensemble_measures) + np.mean(F_ensemble_measures)) / 2.
ensemble_g_res = np.array([final_mean, np.mean(J_ensemble_measures), np.mean(F_ensemble_measures)])
ensemble_g_res = np.reshape(ensemble_g_res, [1, len(ensemble_g_res)])
ensemble_table_g = pd.DataFrame(data=ensemble_g_res, columns=ensemble_g_measures)

with open(csv_name_global_path, 'w') as f:
    ensemble_table_g.to_csv(f, index=False, float_format="%.3f")
print(f'Global results saved in {csv_name_global_path}')

# Print the results
sys.stdout.write(f"--------------------------- Global results for {args.set} ---------------------------\n")
print(ensemble_table_g.to_string(index=False))
sys.stdout.write(f"\n---------- Per sequence results for {args.set} ----------\n")
print(table_seq.to_string(index=False))
total_time = time() - time_start
sys.stdout.write('\nTotal time:' + str(total_time))
