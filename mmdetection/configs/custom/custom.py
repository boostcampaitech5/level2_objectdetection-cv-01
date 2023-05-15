from mmdet.datasets.pipelines.transforms import Albu
_base_ = [
   '../_base_/default_runtime.py'
]

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'
yolo_pretrained = "./work_dirs/swin/yolov3_d53_320_273e_coco-421362b6.pth"
model = dict(
    type='YOLOV3',
    backbone=dict(
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
        out_indices=(3,2,1),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[768,384,192],
        out_channels=[512, 256, 128],
        init_cfg = dict(type='Pretrained', checkpoint=yolo_pretrained)),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=10,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum'),
        init_cfg = dict(type='Pretrained', checkpoint=yolo_pretrained)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='Resize', img_scale=[ (512,512)], keep_ratio=True),
    dict(type='Mosaic',img_scale=(256,256),prob=0.2),
    dict(type='MixUp',img_scale=(512,512),ratio_range=(0.5,1.5),flip_ratio=0.5),
    #dict(type='MixupOrMosaic',ratio=[0.3,0.3,0.4],img_scale=(1024,1024)),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type = 'RandomChoice',
    #      transforms= [
    #         dict(type='Mosaic',img_scale=(1024,1024),prob=0.7),
    #         dict(type='Mixup',img_scale=(1024,1024),ratio_range=(0.5,1.5),flip_ratio=0.5),
    #         dict(type='None')
    # ]),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
train_dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
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
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=train_dataset,
    # val=dict(
    #     type='MultiImageMixDataset',
    #     dataset=dict(
    #         type=dataset_type,
    #         ann_file='',
    #         img_prefix='',
    #         pipeline=[
    #             dict(type='LoadImageFromFile'),
    #             dict(type='LoadAnnotations', with_bbox=True)
    #         ],
    #         filter_empty_gt=False,
    #     ),
    #     pipeline=test_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
    )
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-3,
    weight_decay=0.005,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)
        })
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=273)
evaluation = dict(interval=1, metric=['bbox'])
val_dataloader = dict(shuffle=True)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
