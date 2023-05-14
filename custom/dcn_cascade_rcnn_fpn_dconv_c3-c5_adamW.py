_base_ = "./cascade_rcnn_r50_fpn_adamW.py"
pretrained = "https://download.openmmlab.com/mmdetection/v2.0/dcn/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco/cascade_rcnn_r50_fpn_dconv_c3-c5_1x_coco_20200130-2f1fca44.pth"
model = dict(
    backbone=dict(
        dcn=dict(type="DCN", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    )
)
