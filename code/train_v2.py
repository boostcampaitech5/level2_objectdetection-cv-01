from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import argparse
import json
import wandb
#from mmengine.visualization import Visualizer

def train(config):
    classes = config['classes']

    # config file 들고오기
    cfg = Config.fromfile('./configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py')

    root=config['data_dir']


    #visualize setting
    vis_backend = [dict(type= 'LocalVisBackend'),
                   dict(type= "WandbVisBackend")]
    #wandb 세팅
    if config['wandb_logger']:
        cfg.log_config.hooks.append(dict(type = "MMDetWandbHook",**config['wandb_args']))
        #cfg.log_config.hooks.append(dict(type = "DetVisualizationHook",interval = 50,draw=True, test_out_dir='./work_dirs/deformable_detr'))
        # visualizer = Visualizer.build(config['Visualizer'])
        #cfg.custom_hooks.append(dict(type='DetLocalVisualizer',vis_backend=[dict(type='LocalVisBackend')],name='visualizer'))
    #bbox_head 수정
    cfg.model.bbox_head.num_classes = config['num_classes']
    # dataset config 수정
    cfg.data.train.classes = classes
    #이미지 file_name = '/train/imagename'
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + config['val_json'] # train json 정보
    cfg.data.train.pipeline[3]['policies'][0][0]['img_scale'] = tuple(config['resize']) # Resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + config['test_json'] # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = tuple(config['resize']) # Resize

    #valid set이 존재할 경우
    if "val_json" in config:
        cfg.data.val.classes = classes
        cfg.data.val.img_prefix = root
        cfg.data.val.ann_file = root + config['val_json'] # test json 정보
        cfg.data.train.pipeline[3]['policies'][0][0]['img_scale'] = tuple(config['resize']) # Resize
        #cfg.workflow.append(('val',1))
    cfg.data.samples_per_gpu = 4

    cfg.seed = config['seed']
    cfg.gpu_ids = [0]
    #cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'
    cfg.work_dir = './work_dirs/deformable_detr'

    #cfg.model.roi_head.bbox_head.num_classes = 10
    cfg.runner['max_epochs'] = config['max_epochs']
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    #cfg.fp16 = dict(loss_scale=512.)

    #datasets = [build_dataset(cfg.data.train),build_dataset(cfg.data.val)]
    #cfg.workflow=[('val',1)]
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model)
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