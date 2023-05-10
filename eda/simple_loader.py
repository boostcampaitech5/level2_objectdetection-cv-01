
#%%
from PIL import Image, ImageDraw
import json
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
class_names = ['Generaltrash', 'Paper', 'Paperpack', 'Metal', 'Glass', 'Plastic', 'Styrofoam', 'Plasticbag', 'Battery', 'Clothing']
class_color = ['r','g','b','c','m','y','navy','k','brown','gray']
def json_to_df(json_file='/opt/ml/dataset/train.json'):
    with open(json_file,'r') as f:
        json_data = json.load(f)
        json_img = json_data['images']
        json_box = json_data['annotations']
    img_df = pd.DataFrame(json_img)
    box_df = pd.DataFrame(json_box)

    return img_df,box_df

def get_img(img_config):
    path = img_config['file_name']
    img_path = os.path.join('/opt/ml/dataset/',path)
    img_pil = Image.open(img_path)
    return img_pil
def get_img_with_bbox(img_config,anns,vis_anns = None):
    if vis_anns == None:
        vis_anns = [True]*len(anns)
    path = img_config['file_name']
    img_path = os.path.join('/opt/ml/dataset/',path)
    img_pil = Image.open(img_path)
    
    fig,ax = plt.subplots()
    ax.imshow(img_pil)
    for i in range(len(anns)):
        if vis_anns[i]==False:
            continue
        ann = anns.iloc[i]
        bbox = ann['bbox']
        annotation = class_names[ann['category_id']]
        patch = Rectangle(bbox[:2],bbox[2],bbox[3],edgecolor=class_color[ann['category_id']],facecolor='none')
        ax.add_patch(patch)
        ax.text(patch.get_x(),patch.get_y(),f'{i}.{annotation}',color='k',bbox={"boxstyle":'square','facecolor':"w",'alpha':0.3})
    return fig
def save_classify_json(high,mid,low):
    dir = "/opt/ml/saved_json"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    high_json = {'images':high}
    mid_json = {'images':mid}
    low_json = {'images':low}
    with open(dir+'/high.json','w') as f:
        json.dump(high_json,f)
    with open(dir+'/mid.json','w') as f:
        json.dump(mid_json,f)
    with open(dir+'/low.json','w') as f:
        json.dump(low_json,f)
if __name__ == '__main__':
    data,box=json_to_df()
    # #get_img(data.loc[0])
    # get_img_with_bbox(data.loc[0],[[50,50,200,200],[100,100,50,50]])
    # with open('../../backup/high.json','r') as f:
    #     data = json.load(f)
    # pd_data = pd.DataFrame(data['images'])
    # print(pd_data)
# %%
