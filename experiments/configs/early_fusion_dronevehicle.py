# Early-fusion S2ANet training config for DroneVehicle dataset.
#
# Architecture: ResNet-50 with 4-channel first conv (3 RGB + 1 TIR grayscale
# concatenated before the backbone).  Detection head identical to C2Former
# (S2ANet / FAM + ODM) for a fair comparison.
#
# Run from project root:
#   PYTHONPATH=/home/s3165582/thesis/drone-multimodal-robustness \
#   python models/c2former/tools/train.py \
#       experiments/configs/early_fusion_dronevehicle.py \
#       --work-dir work_dirs/early_fusion

custom_imports = dict(
    imports=['src.models.early_fusion'],
    allow_failed_imports=False)

angle_version = 'le135'

model = dict(
    type='S2ANet',
    backbone=dict(
        type='EarlyFusionResNet',
        in_channels=4,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        pretrained='/home/s3165582/thesis/drone-multimodal-robustness/'
                   'pretrain_weights/resnet50.pth'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    fam_head=dict(
        type='RotatedRetinaHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            scales=[4],
            ratios=[1.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    align_cfgs=dict(
        type='AlignConv',
        kernel_size=3,
        channels=256,
        featmap_strides=[8, 16, 32, 64, 128]),
    odm_head=dict(
        type='ODMRefineHead',
        num_classes=5,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='PseudoAnchorGenerator', strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=1,
            edge_swap=False,
            proj_xy=True,
            target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)),
    train_cfg=dict(
        fam_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        odm_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBboxOverlaps2D')),
            allowed_border=-1,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# ── Dataset ────────────────────────────────────────────────────────────────────
dataset_type = 'DroneVehicleDataset'
data_root = '/home/s3165582/thesis/drone-multimodal-robustness/data/DroneVehicle/'

# 4-channel normalization: ImageNet RGB stats + TIR channel (mean of RGB means)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53, 114.50],
    std=[58.395, 57.12, 57.375, 57.63],
    to_rgb=False)

train_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='ConcatRGBTIR'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(512, 640)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadPairedImageFromFile'),
    dict(type='ConcatRGBTIR'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 640),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ]),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/trainMatchedLabel',
        img_prefix=data_root + 'train/trainMatchedImg',
        pipeline=train_pipeline,
        version=angle_version),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val/valMatchedLabel',
        img_prefix=data_root + 'val/valMatchedImg',
        pipeline=test_pipeline,
        version=angle_version),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/testMatchedLabel',
        img_prefix=data_root + 'test/testMatchedImg',
        pipeline=test_pipeline,
        version=angle_version))

# ── Schedule ───────────────────────────────────────────────────────────────────
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1)

# ── Runtime ────────────────────────────────────────────────────────────────────
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
