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
    # cfg.data.train.classes = classes
    # cfg.data.train.img_prefix = data_root
    # cfg.data.train.ann_file = data_root + 'split_train.json'
    # cfg.data.train.pipeline[4]["img_scale"] = (512, 512)

    # cfg.data.test.classes = classes
    # cfg.data.test.img_prefix = data_root
    # cfg.data.test.ann_file = data_root + 'test.json'
    # cfg.data.test.pipeline[1]["img_scale"] = (512,512)

    # cfg.data.val.classes = classes
    # cfg.data.val.img_prefix = data_root
    # cfg.data.val.ann_file = data_root + 'split_val.json'
    # cfg.data.val.pipeline = cfg.data.test.pipeline

    # cfg.model.bbox_head['num_classes'] = 10

    # cfg.data.samples_per_gpu = 4

    cfg.seed = 2023
    cfg.gpu_ids = [0]
    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    cfg.runner = dict(type='EpochBasedRunner', max_epochs=epochs)

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
    if checkpoint_path != None:
        dir_name += '_resume'
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
    train('./configs/yolo/myYolov3.py', 2)
    # train('./configs/yolo/yolov3_d53_320_273e_coco.py', 2)
    # train('./configs/yolo/yolov3_d53_320_273e_coco.py', 60,
    #       './work_dirs/230510_0951_YOLOV3_epochs2/latest.pth')
    # train('./configs/ssd/mySSD300.py', 3)