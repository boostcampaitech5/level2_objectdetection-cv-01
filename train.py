from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
from mmdet.datasets import coco
from mmcv.runner import load_checkpoint

from datetime import datetime
import os
import wandb

def set_wandb(cfg:Config):
    """
        wandb 세팅

        Args: 
            cfg (Config) : mmdetection Config 객체
    """
    init_kwargs = {
        'project': 'level2) Object Detection',
        'tags' : ['mmdetection', ],
        'entity' : 'janghyeji0828',
        'name' : cfg.work_dir.split('/')[-1],
    }

    cfg.log_config.hooks = [
        dict(type='TextLoggerHook', interval=100),
        dict(type='MMDetWandbHook',
            init_kwargs=init_kwargs,
            interval=100,
            # log_checkpoint=True,
            # log_checkpoint_metadata=True,
            # num_eval_images=100,
            # bbox_score_thr=0.3)
        )
    ]

def set_config(cfg:Config, epochs):
    """ config setting

    Args:
        cfg (Config): config to customize
        epochs (int): train epoch
    """

    # cfg.data.samples_per_gpu = 4

    cfg.seed = 2023
    cfg.gpu_ids = [0]
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    # evaluation = dict(interval=1, classwise=True, metric="bbox")

    cfg.runner = dict(type='EpochBasedRunner', max_epochs=epochs)

    # optimizer
    # cfg.optimizer=dict(type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05)
    # cfg.optimizer=dict(type='Adam', lr=0.0003, weight_decay=0.0001)

    # fp16
    # cfg.fp16 = dict(loss_scale=512.0)

    # soft nms
    # cfg.model.test_config.nms=dict(type="soft_nms", iou_threshold=0.7)

    # cosine annealing
    # cfg.lr_config=dict(policy='CosineAnnealing', min_lr_ratio=0.001, by_epoch = False)
    # cosine restart
    # cfg.lr_config=dict(policy="CosineRestart", warmup="linear", warmup_iters=1000,
    #                     warmup_ratio=0.001, periods=[1, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    #                     restart_weights=[1, 1, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125],
    #                     min_lr=1e-5,)
    
    # resume_from
    # cfg.load_from = 'path'
    # cfg.resume_from = 'path'


def train(cfg_path, epochs, checkpoint_path=None):
    """ train model

    Args:
        cfg_path (Config): cfg.py path
        epochs (int): train epochs
        checkpoint_path (str) : path for checkpoint to resume train
    """

    # load mmdetection config file 
    cfg = Config.fromfile(cfg_path)

    now = datetime.now().strftime('%y%m%d_%H%M_')
    dir_name = now + cfg.model['type'] + '_epochs' + str(epochs)
    # if checkpoint_path != None:
    #     dir_name += '_resume'
    cfg.work_dir = os.path.join('./work_dirs/', dir_name)

    # set config   
    set_config(cfg, epochs)

    # set wandb
    set_wandb(cfg)

    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    if(checkpoint_path == None):
        model.init_weights()
    else:
        checkpoint = load_checkpoint(model, checkpoint_path)

    # 모델 학습
    train_detector(model, datasets, cfg, distributed=False, validate=True, meta=dict())

if __name__ == '__main__':

    # SSD 300 / pre-trained
    # train('./configs/ssd/mySSD300.py', 60,
    #         './models/ssd300_coco_20210803_015428-d231a06e.pth')
    
    # SSD 512 / pre-trained
    # train('./configs/ssd/mySSD512.py', 60,
    #         './models/ssd512_coco_20210803_022849-0a47a1ca.pth')

    # YOLO v3 / pre-trained
    train('./configs/yolo/myYolov3.py', 30, 
          './models/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth')
    #         './work_dirs/230511_1016_YOLOV3_epochs60_pretrained/latest.pth')