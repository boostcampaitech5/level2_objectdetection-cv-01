_base_ = [
    '../_base_/models/ssd300.py', '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

# dataset settings
# dataset_type = 'CocoDataset'
data_root = '../../dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") 

# workflow = [('train', 1), ('val', 1)]

input_size = 512
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='SSDVGG',
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://vgg16_caffe')),
    neck=dict(
        type='SSDNeck',
        in_channels=(512, 1024),
        out_channels=(512, 1024, 512, 256, 256, 256),
        level_strides=(2, 2, 1, 1),
        level_paddings=(1, 1, 0, 0),
        l2_norm_scale=20),
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=10,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])),

        # focal loss
        # loss_cls=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
        # loss_conf=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),
        # loss_xy=dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0),

    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,
        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        nms=dict(type='nms', iou_threshold=0.45),
        # soft nms
        # nms=dict(type="soft_nms", iou_threshold=0.7),
        min_bbox_size=0,
        score_thr=0.02,
        max_per_img=200))
cudnn_benchmark = True

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),

    # # mosaic
    # dict(type='Mosaic', img_scale=(512, 512), pad_val=114.0),
    
    # # mixup
    # dict(
    #     type='MixUp',
    #     img_scale=(512, 512),
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),

    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # multiscale
    # dict(type='Resize', img_scale=[(1024, 1024), (512, 512)], multiscale_mode = "range", keep_ratio=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
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
data = dict(
    samples_per_gpu=8,
    # samples_per_gpu=16,
    # workers_per_gpu=4,
    train=dict(
        # type=dataset_type,
        classes=classes,
        ann_file=data_root + 'split_train.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        # type=dataset_type,
        classes=classes,
        ann_file=data_root + 'split_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        # type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

# cosine annealing
# lr_config=dict(polyicy='CosineAnnealing', min_lr_ratio=0.001, by_epoch = False)
# cosine restart
# lr_config=dict(polyicy="CosineResatrt", min_lr_ratio=0.001, by_epoch = False)
