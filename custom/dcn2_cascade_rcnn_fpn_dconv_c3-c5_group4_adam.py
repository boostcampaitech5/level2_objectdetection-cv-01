_base_ = "./cascade_rcnn_r50_fpn_adam.py"
model = dict(
    backbone=dict(
        dcn=dict(type="DCNv2", deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    )
)
