_base_ = ['../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py']
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=4, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)))
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)



classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass", 
    "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]
train_pipeline = [
    dict(type='Mosaic',img_scale=(1024,1024),prob=0.3),
    #dict(type='MixUp',img_scale=(1024,1024),ratio_range=(0.8,1.2)),
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(1024,1024),(512,512)],
                # img_scale=(1024,1024),
                multiscale_mode='range',
                keep_ratio=True)
        ]]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

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
            classes = classes,
            type='CocoDataset',
            ann_file='opt/ml/dataset/classwise_dataset/split_train_0.json',
            img_prefix='opt/ml/dataset',
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
    # train=train_dataset,
    train=train_dataset,
    val=dict(type='CocoDataset',pipeline=test_pipeline,ann_file='opt/ml/dataset/classwise_dataset/split_train_0.json',
            img_prefix='opt/ml/dataset',classes = classes),
    test=dict(type='CocoDataset',pipeline=test_pipeline,ann_file='opt/ml/dataset/classwise_dataset/split_train_0.json',
            img_prefix='opt/ml/dataset',classes=classes))



