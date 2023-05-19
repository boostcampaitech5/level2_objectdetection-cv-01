_base_ = './yolov3_d53_mstrain-608_273e_coco.py'

# dataset settings
dataset_type = 'CocoDataset'
data_root = '../../dataset/'

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing") 

# workflow = [('train', 1), ('val', 1)]

model = dict(
    type='YOLOV3',
    backbone=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53')),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
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

        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
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
        # nms=dict(type='nms', iou_threshold=0.45),
        # soft nms
        nms=dict(type="soft_nms", iou_threshold=0.7),
        max_per_img=100))

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    # # mosaic
    # dict(type='Mosaic', img_scale=(512, 512), pad_val=114.0),
    # # # mixup
    # dict(
    #     type='MixUp',
    #     img_scale=(512, 512),
    #     ratio_range=(0.8, 1.6),
    #     pad_val=114.0),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    # multi scale
    # dict(type='Resize', img_scale=[(1024, 1024), (512, 512)], multiscale_mode = "range", keep_ratio=True), 
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

# train_dataset = dict(
#     type='MultiImageMixDataset',
#     dataset=dict(
#         type=dataset_type,
#         ann_file=data_root + 'split_train.json',
#         img_prefix=data_root,
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         filter_empty_gt=False,
#     ),
#     pipeline=train_pipeline)

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

