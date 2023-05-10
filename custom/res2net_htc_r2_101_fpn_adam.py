_base_ = "./htc_without_semantic_r50_fpn_adam.py"
model = dict(
    backbone=dict(
        type="Res2Net",
        depth=101,
        scales=4,
        base_width=26,
        init_cfg=dict(
            type="Pretrained", checkpoint="open-mmlab://res2net101_v1d_26w_4s"
        ),
    )
)
