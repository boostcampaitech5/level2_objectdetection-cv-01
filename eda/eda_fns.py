
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simple_loader import json_to_df

data_df, ann_df = json_to_df()
class_names = ['General\ntrash', 'Paper', 'Paper\npack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plastic\nbag', 'Battery', 'Clothing']


        

#bar chart에 수치 표기
def bar_score_show(bar,score,ax):
    idx = 0
    for b,s in zip(bar,score):
        h = b.get_height()
        ax.text(b.get_x()+(b.get_width()/2),h,f'{s}',ha='center',size=10)
        idx += 1



#class 개수 분포
def eda_class_disribution():
    fig,ax = plt.subplots()
    num_of_class = ann_df['category_id'].value_counts().sort_index()
    bar = ax.bar(class_names,num_of_class)
    bar_score_show(bar,num_of_class,ax)
    return [fig,ax,num_of_class]

#이미지당 오브젝트 개수 분포
def eda_one_image_object_distribution():
    fig,ax = plt.subplots()
    image_class_num = ann_df.groupby('image_id')['category_id'].count().value_counts()
    bar = ax.bar(image_class_num.index,image_class_num) 
    #bar_score_show(bar,image_class_num)
    return [fig,ax,image_class_num]

#이미지당 포함된 클래스 개수 분포
def eda_one_image_class_distribution():
    fig,ax = plt.subplots()
    image_class_num = ann_df.groupby('image_id')['category_id'].nunique().value_counts()
    bar = ax.bar(image_class_num.index,image_class_num) 
    bar_score_show(bar,image_class_num,ax)
    return [fig,ax,image_class_num]

#박스의 중심점 위치 분포
def eda_box_position_distribution():
    box_pos = []
    boxes = ann_df['bbox']
    for box in boxes:
        x=box[0]+(box[2]/2)
        y=box[1]+(box[3]/2)
        box_pos.append([x,y])
    box_pos = np.array(box_pos)
    fig,ax = plt.subplots()
    ax.scatter(box_pos[:,0],box_pos[:,1],alpha=0.1,s=1)
    return [fig,ax]

#box의 크기 분포: ~100, 100~500, 500~1000, 1000~5000, 5000~
def eda_box_area_distribution():
    fig,ax = plt.subplots()
    bins = [0,1024,1024*4,1024*16,1024*64,1024*256,1024*1024]
    area_df = pd.cut(ann_df['area'],bins)
    area_df = area_df.value_counts().sort_index()
    area_df.index = ['~1024','1024~\n4096','4096~\n16384','16384~\n65536','65536~\n262144','262144~']
    bar=ax.bar(area_df.index, area_df)
    bar_score_show(bar,area_df,ax)
    fig.show()
    return [fig,ax]

#box의 w/h 비율 분포
def eda_box_wh_ratio_distribution():
    fig,ax = plt.subplots()
    bins = [0,0.1,0.3,0.5,1,2,3,10,30]
    w_h_df = ann_df['bbox'].tolist()
    w_h_np = np.array(w_h_df)[:,2:]

    w_h_ratio = w_h_np[:,0]/w_h_np[:,1]
    w_h_ratio_df = pd.DataFrame(w_h_ratio)
    cutted_ratio = pd.cut(w_h_ratio_df[0],bins).value_counts().sort_index()
    cutted_ratio.index = ['~0.1','0.1~0.3','0.3~0.5','0.5~1','1~2','2~3','3~10','10~']
    bar = ax.bar(cutted_ratio.index, cutted_ratio)
    bar_score_show(bar,cutted_ratio,ax)
    return [fig,ax]
    

