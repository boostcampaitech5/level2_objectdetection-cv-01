_base_ = "./swin-t-p4-w7_cascade_rcnn_fpn_adamW.py"

# pretrained = https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth

model = dict(backbone=dict(depths=[2, 2, 18, 2]))
