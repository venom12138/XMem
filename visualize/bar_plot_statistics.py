import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
csv_names = ['/home/venom/projects/XMem/output/1006_new_valdata_normal_train/D0080_no_align,use_text=0,use_flow=1,network_12500/per-sequence_results-val.csv',
            '/home/venom/projects/XMem/output/1006_new_valdata_normal_train/D0080_no_align,use_text=0,use_flow=1,network_15000/per-sequence_results-val.csv',
            '/home/venom/projects/XMem/output/1007_new_valdata_normal_train_CLIPL/D0082_no_align,use_text=1,use_flow=1,network_11500/per-sequence_results-val.csv',
            '/home/venom/projects/XMem/output/1007_new_valdata_normal_train_CLIPL/D0082_no_align,use_text=1,use_flow=1,network_15000/per-sequence_results-val.csv',
            '/home/venom/projects/XMem/output/1010_ema_ts_all_align_loss=0/D0110_teacher_warmup=100,teacher_loss_weight=0.01,use_text=0,use_flow=1,network_6000/per-sequence_results-val.csv',
            '/home/venom/projects/XMem/output/1010_ema_ts_all_align_loss=0/D0110_teacher_warmup=100,teacher_loss_weight=0.01,use_text=0,use_flow=1,network_10000/per-sequence_results-val.csv',
            '/home/venom/projects/XMem/output/1010_ema_ts_all_align_loss=0/D0113_teacher_warmup=100,teacher_loss_weight=0.01,use_text=1,use_flow=1,network_2000/per-sequence_results-val.csv',
            '/home/venom/projects/XMem/output/1010_ema_ts_all_align_loss=0/D0113_teacher_warmup=100,teacher_loss_weight=0.01,use_text=1,use_flow=1,network_10000/per-sequence_results-val.csv',]

csv_name1 = '/home/venom/.exp/XMem/1003XMem/per-sequence_results-val.csv'

csv1 = pd.read_csv(csv_name1)

counts_dict = {}
for csv_name in csv_names:
    csv2 = pd.read_csv(csv_name)
    # df = df.sort_values(by='J-Mean', ascending=False)
    data_merge = pd.merge(csv1, csv2, on='narration', how='outer')
    data_merge = data_merge.drop(columns='Sequence_y')
    data_merge.fillna(0, inplace=True)
    data_merge['J_diff'] = data_merge['J-Mean_y'] - data_merge['J-Mean_x'] # csv2-csv1
    data_merge['F_diff'] = data_merge['F-Mean_y'] - data_merge['F-Mean_x'] # csv2-csv1
    data_merge = data_merge.sort_values(by='J_diff', ascending=False)

    data_merge[['J-Mean_x', 'J-Mean_y', 'F-Mean_x', 'F-Mean_y', 'J_diff', 'F_diff']] = data_merge[['J-Mean_x', 'J-Mean_y', 'F-Mean_x', 'F-Mean_y', 'J_diff', 'F_diff']].astype('float16')
    
    for idx, item in enumerate(data_merge['narration']):
        item = '_'.join(item.split('_')[:-1])
        if item in counts_dict.keys():
            counts_dict[item]['cnt'] += 1
            counts_dict[item]['num'].append(data_merge.iloc[idx]['J_diff'])
        else:
            counts_dict.update({item: {'narration':data_merge.iloc[idx]['Sequence_x'],'cnt': 1, 'num': [data_merge.iloc[idx]['J_diff']]}})

key_list = []
mean_list = []
std_list = []
for key, value in counts_dict.items():
    key_list.append(key)
    mean_list.append(np.mean(value['num']))
    std_list.append(np.std(value['num']))

idx = np.argsort(mean_list)
key_list = np.array(key_list)[idx]
mean_list = np.array(mean_list)[idx]
std_list = np.array(std_list)[idx]

plt.bar(range(len(key_list)), mean_list, yerr=std_list, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.title('ours J metric - XMem J metric')
plt.legend()
plt.show()

# per category
category_count_dict = {}
with open('../val_data/EPIC100_state_positive_val.yaml', 'r') as f:
    yaml_data = yaml.safe_load(f)
    verb_class_df = pd.read_csv('../XMem_evaluation/EPIC_100_verb_classes.csv').drop(columns=['id', 'category'])
    verb_class = {}
    for index, row in verb_class_df.iterrows():        
        verb_class.update({row['key']: row['instances'].strip('"').strip('[').strip(']').replace("'",'').replace(' ','').split(',')})
    
for key, value in counts_dict.items():
    verb_name = yaml_data[key]['verb']
    if verb_name in list(category_count_dict.keys()):
        category_count_dict[verb_name]['count'] += 1
        category_count_dict[verb_name]['num'].extend(value['num'])
    else:
        category_count_dict.update({verb_name: {'count': 1, 'num': counts_dict[key]['num']}})

key_list = []
mean_list = []
std_list = []
for key, value in category_count_dict.items():
    key_list.append(key)
    mean_list.append(np.mean(value['num']))
    std_list.append(np.std(value['num']))

idx = np.argsort(mean_list)
key_list = np.array(key_list)[idx]
mean_list = np.array(mean_list)[idx]
std_list = np.array(std_list)[idx]

plt.bar(key_list, mean_list, yerr=std_list, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.title('category ours J metric - XMem J metric')
plt.legend()
plt.show()