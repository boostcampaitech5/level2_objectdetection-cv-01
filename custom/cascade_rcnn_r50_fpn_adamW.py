_base_ = [
    "./model/cascade_rcnn_r50_fpn_model.py",
    "../mmdetection/configs/_base_/datasets/coco_detection.py",
    "./scheduler/schedule_30e_adamW.py",
    "../mmdetection/configs/_base_/default_runtime.py",
]
