from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import argparse
import json
#from mmcv.runner import load_checkpoint
import os
#from mmengine.visualization import Visualizer

def train(config):
    classes = config['classes']

    # config file 들고오기
    cfg = Config.fromfile(config['base_model'])

    root=config['data_dir']

    #wandb 세팅
    if config['wandb_logger']:
        cfg.log_config.hooks.append(dict(type = "MMDetWandbHook",**config['wandb_args']))
        

    ### model 종속    
    #bbox_head 수정
    cfg.model.bbox_head.num_classes = config['num_classes']
    # dataset config 수정
    cfg.data.train.dataset.classes = classes
    cfg.data.train.dataset.img_prefix = root
    cfg.data.train.dataset.ann_file = root + config['train_json'] # train json 정보
   # cfg.data.train.pipeline[3]['policies'][0][0]['img_scale'] = tuple(config['resize']) # Resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + config['test_json'] # test json 정보
   # cfg.data.test.pipeline[1]['img_scale'] = tuple(config['resize']) # Resize

    #valid set이 존재할 경우
    if "val_json" in config:
        cfg.data.val.classes = classes
        cfg.data.val.img_prefix = root
        cfg.data.val.ann_file = root + config['val_json'] # test json 정보
        #cfg.workflow.append(('val',1))
    cfg.data.samples_per_gpu = 4

    cfg.seed = config['seed']
    cfg.gpu_ids = [0]
    #cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
    cfg.work_dir = config['work_dir']

    #cfg.model.roi_head.bbox_head.num_classes = 10
    cfg.runner['max_epochs'] = config['max_epochs']
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    #pretrained model loa
    if 'load_from' in config:
        cfg.load_from = os.path.join(cfg.work_dir,config['load_from'])
    elif 'resume_from' in config:
        cfg.resume_from = os.path.join(cfg.work_dir,config['resume_from'])
    #checkpoint_path =  os.path.join(cfg.work_dir, 'deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth')
    #checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    
    
    #datasets = [build_dataset(cfg.data.train),build_dataset(cfg.data.val)]
    #cfg.workflow=[('val',1)]

    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
    if 'load_config' not in config:
        model.init_weights()
    train_detector(model, datasets, cfg, distributed=False, validate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'MMdetectionWithConfig',
                                     description='')
    parser.add_argument('-c','--config',default='./config.json',type=str,help='학습에 사용할 config파일의 path')

    args = parser.parse_args()
    with open(args.config,'r') as w:
        config = json.load(w)
    train(config)