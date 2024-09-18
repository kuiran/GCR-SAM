_base_ = [
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmtrack.models.sam'], allow_failed_imports=False)

backbone_norm_cfg = dict(type='LN', eps=1e-6)

vit_embed_dims = 1280
prompt_embed_dims = 256
image_size = 1024
vit_patch_size = 16
image_embed_size = image_size // vit_patch_size

model = dict(
    type='SingleStageDetector',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ViTSAM',
        arch='huge',
        img_size=image_size,
        patch_size=vit_patch_size,
        window_size=14,
        qkv_bias=True,
        use_rel_pos=True,
        norm_cfg=backbone_norm_cfg),
    neck=[
        dict(type='SAMNeck', 
             in_channels=vit_embed_dims,
             out_channels=prompt_embed_dims,
             freeze=True)
    ],
    bbox_head=dict(
        type='SAMHead',
        points_per_side=32,
        points_per_batch=64,
        prompt_encoder=dict(
            type='PromptEncoder',
            embed_dims=prompt_embed_dims,
            image_embed_size=image_embed_size,
            image_size=image_size,
            mask_channels=16),
        multimask_output=False,
        add_scores=True,
        mask_decoder=dict(
            type='MaskDecoder',
            num_multimask_outputs=3,
            transformer=dict(
                type='TwoWayTransformer',
                embed_dims=prompt_embed_dims,
                depth=2,
                num_heads=8,
                feedforward_channels=2048),
            transformer_dims=prompt_embed_dims,
            iou_head_depth=3,
            iou_head_hidden_dim=256)),
    test_cfg=dict(
        with_mask=True,
        with_bbox=True,
        mask_threshold=0.,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.7,
        stability_score_offset=1.0,
        nms=dict(type='nms', iou_threshold=0.85),
        max_per_img=1000))

# load_from = 'data/pretrained/sam/sam_vit_h_4b8939.pth'

load_from = '/home/ubuntu/hxm/ovd/mmdetection/data/pretrained/sam/sam_vit_h_4b8939.pth'

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsWithPointAndText', with_bbox=True),
    dict(type='ResizeImgBoxAndPoint', scale=(image_size, image_size), keep_ratio=True),
    dict(type='Pad', size=(image_size, image_size), pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

# coco dataset

train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type='mmtrack.Coco_add_points_text',
        data_root='data/coco',
        # ann_file='annotations/instances_train2017_add-points_size-range0.5.json',
        ann_file='annotations/instances_train2017_size-range0.25_with-mask.json',
        data_prefix=dict(img='train2017'),
        filter_cfg=dict(filter_empty_gt=True, min_size=-1),
        pipeline=train_pipeline
    )
)

data_root = 'data/coco/'

# dataset_type = 'LVISV1Dataset'
# data_root = 'data/lvis_v1/'

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsWithPointAndText', with_bbox=True, with_mask=False),
    dict(type='ResizeImgAndPoint', scale=(image_size, image_size), keep_ratio=True),
    dict(type='PadImgOnly', size=(image_size, image_size), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='mmtrack.LaSOTDataset1stFrame',
        data_root='data/',
        ann_file='lasot/annotations/lasot_test_1st_frame.json',
        data_prefix=dict(img='lasot/LaSOTBenchmark'),
        test_mode=True,
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

default_hooks = dict(logger=dict(interval=100))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=99)
val_cfg = dict(type='ValLoop')  
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/lasot/annotations/lasot_test_1st_frame.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='save_path'
    )
test_evaluator = val_evaluator

