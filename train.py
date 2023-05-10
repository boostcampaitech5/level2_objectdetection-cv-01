from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
from mmdet.datasets import coco

from datetime import datetime
import wandb
import os

def set_wandb(cfg:Config):
    """
        wandb 세팅

        Args: 
            cfg (Config) : mmdetection Config 객체
    """
    init_kwargs = {
        'project': 'level2) Object Detection',
        'tags' : ['mmdetection','faster_rcnn_r50'],
        'entity' : 'janghyeji0828',
        'name' : cfg.work_dir.split('/')[-1],
    }

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
            init_kwargs=init_kwargs,
            interval=300,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100,
            bbox_score_thr=0.3)
    ]

def train(cfg_path, epochs):
    # load config json 

    # load mmdetection config file 
    cfg = Config.fromfile(cfg_path)

    # set config
    cfg.seed = 2023
    cfg.gpu_ids = [0]
    cfg.device = get_device()
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=epochs)

    now = datetime.now().strftime('%y%m%d_%H%M_')
    dir_name = now + cfg.model['type'] + '_epochs' + str(epochs)
    cfg.work_dir = os.path.join('./work_dirs/', dir_name)

    cfg.evaluation = dict(
        interval = 2,
        metric = 'mAP',
        save_best = 'mAP',
        rule = 'greater',
        by_epoch = False,
        iou_thrs=[0.5],
        )

    # set wandb
    set_wandb(cfg)

    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)

if __name__ == '__main__':
    train('./configs/yolo/myYolov3.py', 2)
    # train('./configs/ssd/mySSD300.py', 3)