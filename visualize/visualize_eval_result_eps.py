import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_eval_result(run_dir):
    # run_dir = '/u/ryanxli/.exp/0925_state_change_segm/0925_nf=8_bs=8/D0002_lr=1e-5,start_warm=2500,end_warm=15000'
    JF_list = []
    J_list = []
    F_list = []
    iters = []
    for dir_name in os.listdir(run_dir):
        if dir_name.startswith('eval_'):
            eval_dir = os.path.join(run_dir, dir_name)
        else:
            continue

        csv_path = os.path.join(eval_dir,'global_results-val.csv')
        df = pd.read_csv(csv_path)      
        iters.append(int(dir_name.split('_')[-1]))
        JF_list.append(df['J&F-Mean'][0])
        J_list.append(df['J-Mean'][0])
        F_list.append(df['F-Mean'][0])
    
    idx = np.argsort(iters)
    # print(idx)
    iters = np.array(iters)[idx]
    
    JF_list = np.array(JF_list)[idx]
    J_list = np.array(J_list)[idx]
    F_list = np.array(F_list)[idx]
    
    plt.plot(iters, JF_list, label='J&F-Mean')
    exp_name = run_dir.split('/')[-2] + '--' + run_dir.split('/')[-1]
    plt.title(exp_name)
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'JF_plot_diff_eps.png'))
    
    return iters, JF_list, J_list, F_list