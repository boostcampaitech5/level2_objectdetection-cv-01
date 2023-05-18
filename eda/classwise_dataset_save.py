from simple_loader import *
from eda_fns import *
import eda_fns
import matplotlib.pyplot as plt

target = 'split_val'

data_df, ann_df = json_to_df(f'../../dataset/{target}.json')
with open(f'../../dataset/{target}.json','r') as f:
    save_json = json.load(f)

for idx in [[3,8],[4,5]]:
    split_ann_df = ann_df[ann_df['category_id'].isin(idx)]    
    split_data_df = data_df[data_df['id'].isin(split_ann_df['image_id'].unique())]
    save_json['images'] = split_data_df.to_dict('records')
    save_json['annotations'] = split_ann_df.to_dict('records')
    print(f'idx:{idx}, split_data:{len(split_data_df)},split_ann:{len(split_ann_df)}')
    with open(f'./classwise_dataset/{target}_{",".join(map(str,idx))}.json','w') as f:
        json.dump(save_json,f,indent=2)


    