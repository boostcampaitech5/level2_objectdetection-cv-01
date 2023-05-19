_base_ = [
    "./model/faster_rcnn_r50_fpn_model.py",
    "../mmdetection/configs/_base_/datasets/coco_detection.py",
    "./scheduler/schedule_30e_adam.py",
    "../mmdetection/configs/_base_/default_runtime.py",
]

# fp16 settings
fp16 = dict(loss_scale=512.0)