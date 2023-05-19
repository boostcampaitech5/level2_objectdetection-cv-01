_base_ = [
    "./model/cascade_rcnn_r50_fpn_focalLoss_model.py",
    "./dataset/coco_detection_mstrain.py",
    "./scheduler/schedule_30e_CosRes_adamW.py",
    "../mmdetection/configs/_base_/default_runtime.py",
]

# pretrained = https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22kto1k.pth

model = dict(
    backbone=dict(
        _delete_=True,
        type="SwinTransformer",
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
    ),
    neck=dict(in_channels=[192, 384, 768, 1536]),
)

# fp16 settings
fp16 = dict(loss_scale=512.0)

optim_wrapper = dict(
    type="OptimWrapper",
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        }
    ),
    optimizer=dict(
        _delete_=True, type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05
    ),
)