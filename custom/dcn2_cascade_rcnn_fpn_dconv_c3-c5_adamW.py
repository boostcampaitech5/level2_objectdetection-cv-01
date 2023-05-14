_base_ = "./cascade_rcnn_r50_fpn_adamW.py"
pretrained = "https://download.openmmlab.com/mmdetection/v2.0/fp16/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco/mask_rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco_20210520_180434-cf8fefa5.pth"
model = dict(
    backbone=dict(
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    )
)
