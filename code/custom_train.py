# 모듈 import

from mmcv import Config
import mmdet
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import get_device

import wandb
import os
import argparse
import json

# config_file = "mmdetection/configs/path/to/config.py"


# classes = (
#     "General trash",
#     "Paper",
#     "Paper pack",
#     "Metal",
#     "Glass",
#     "Plastic",
#     "Styrofoam",
#     "Plastic bag",
#     "Battery",
#     "Clothing",
# )


def train(config):
    # config file 들고오기
    cfg = Config.fromfile(config["config_file"])

    # TODO 이 부분을 arg parser로 들고오는 것도 좋을 것 같다.
    root = "/opt/ml/dataset"
    # print(cfg.data.train.pipeline)
    # raise
    # root = config["data_dir"]

    # dataset config 수정
    cfg.data.train.classes = config["classes"]
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = os.path.join(root, "split_train.json")  # train json 정보
    cfg.data.train.pipeline[2]["img_scale"] = tuple(config["img_scale"])  # Resize

    # validation dataset
    cfg.data.val.classes = config["classes"]
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = os.path.join(root, "split_val.json")  # val json 정보
    cfg.data.val.pipeline[1]["img_scale"] = tuple(config["img_scale"])  # Resize

    cfg.data.test.classes = config["classes"]
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = os.path.join(root, "test.json")  # test json 정보
    cfg.data.test.pipeline[1]["img_scale"] = tuple(config["img_scale"])  # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = 2022
    cfg.gpu_ids = [0]
    cfg.work_dir = os.path.join(
        config["work_dir"], os.path.split(config["config_file"])[-1]
    )

    # cfg.model.roi_head.bbox_head[0].num_classes = 10
    # cfg.model.roi_head.bbox_head[1].num_classes = 10
    # cfg.model.roi_head.bbox_head[2].num_classes = 10

    cfg.log_config.hooks = [
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            init_kwargs={
                "project": "level2-Object_detection",
                "tags": ["mmdetection", "faster_rcnn_r50"],
                "entity": "boost_camp",
                "name": os.path.split(cfg.work_dir)[-1],
            },
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            # # num_eval_images=100,
            # bbox_score_thr=0.3,
        ),
    ]
    # train시키기 위해 고군분투..
    for i in cfg.model.roi_head.bbox_head:
        i.num_classes = 10

    # print(cfg.pretty_text)

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()

    
    # 동규가 준 코드
    # # build_dataset
    # cfg.workflow.append(("val", 1))
    datasets = [build_dataset(cfg.data.train)]  # , build_dataset(cfg.data.val)]

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="MMdetectionWithConfig", description="")
    parser.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help="학습에 사용할 config파일의 path",
    )

    args = parser.parse_args()
    with open(args.config, "r") as w:
        config = json.load(w)
    train(config)
