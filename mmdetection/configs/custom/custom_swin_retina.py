_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768], start_level=0, num_outs=5))
#model.bbox_head.init_cfg=dict(type='Pretrained',checkpoint=pretrained_retina)
# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='Mosaic',img_scale=(512,512),prob=0.3),
    #dict(type='MixUp',img_scale=(1024,1024),ratio_range=(0.8,1.2)),
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(512,512)],
                # img_scale=(1024,1024),
                multiscale_mode='range',
                keep_ratio=True)
        ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024,1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
train_dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file='',
            img_prefix='',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            filter_empty_gt=False,
        ),
        pipeline=train_pipeline)

data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=train_dataset,
   # train=dict(type='CocoDataset',pipeline=train_pipeline),
    val=dict(type='CocoDataset',pipeline=test_pipeline),
    test=dict(type='CocoDataset',pipeline=test_pipeline))
