_base_ = [
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['mmtrack.models.sam.sam'], allow_failed_imports=False)

backbone_norm_cfg = dict(type='LN', eps=1e-6)

vit_embed_dims = 768
prompt_embed_dims = 256
image_size = 1024
vit_patch_size = 16
image_embed_size = image_size // vit_patch_size

model = dict(
    type='GCR',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ViTSAM',
        arch='base',
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
    rpn_head=dict(
        type='GCR_proposal_gnerator',
        clip_model_name='ViT-B/32',
        clip_feature_path="clip_feature_path"),
    roi_head=dict(
        type='GCR_box_decoder',
        mask_head=dict(
            type='SAMHead',
            points_per_side=32,
            points_per_batch=64,
            prompt_encoder=dict(
                type='PromptEncoder',
                embed_dims=prompt_embed_dims,
                image_embed_size=image_embed_size,
                image_size=image_size,
                mask_channels=16),
            mask_decoder=dict(
                type='MaskDecoder',
                num_multimask_outputs=3,
                transformer=dict(
                    type='TwoWayTransformer',
                    depth=2,
                    embed_dims=prompt_embed_dims,
                    feedforward_channels=2048,
                    num_heads=8),
                transformer_dims=prompt_embed_dims,
                iou_head_depth=3,
                iou_head_hidden_dim=256),
            multimask_output=False,
            test_cfg=dict(
                with_mask=True,
                with_bbox=False,
                mask_threshold=.0,
                pred_iou_thresh=-99.0,
                stability_score_thresh=-99.0,
                stability_score_offset=1.0,
                nms=dict(type='nms', iou_threshold=0.7),
                max_per_img=1000))),
    train_cfg=dict(
            rpn=None,
            rcnn=None),
    test_cfg=dict(
        rpn=None,
        rcnn=None))

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsWithPointAndText', with_bbox=True),
    dict(type='ResizeImgBoxAndPoint', scale=(image_size, image_size), keep_ratio=True),
    dict(type='Pad', size=(image_size, image_size), pad_val=dict(img=(114, 114, 114))),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

load_from = 'exp/model/sam/sam_vit_b_01ec64_mmdet.pth'

data_root = 'data/'
train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='mmdet.AspectRatioBatchSampler'),
    dataset=dict(
        type='mmdet.Objects365V2Dataset',
        data_root='data/Objects365',
        # ann_file='Obj365_v2/annotations/zhiyuan_objv2_train_fixname_fixmiss_add-points_size-range0.5.json',
        ann_file='Obj365_v2/annotations/zhiyuan_objv2_train_fixname_fixmiss_add-points_size-range0.5.json',
        data_prefix=dict(img='Obj365_v2/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotationsWithPointAndText', with_bbox=True, with_mask=True),
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
        type='mmdet.Objects365V2Dataset',
        data_root='data/Objects365',
        # ann_file='Obj365_v2/annotations/zhiyuan_objv2_train_fixname_fixmiss_add-points_size-range0.5.json',
        ann_file='Obj365_v2/annotations/zhiyuan_objv2_train_subset_k500_size-range0.5.json',
        data_prefix=dict(img='Obj365_v2/train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=test_pipeline))
test_dataloader = val_dataloader
default_hooks = dict(logger=dict(interval=100))
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=99)
val_cfg = dict(type='ValLoop')  
test_cfg = dict(type='TestLoop')

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017_add-points_size-range0.5.json',
    metric='bbox',
    format_only=True,
    outfile_prefix=None
    )
test_evaluator = val_evaluator

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1, norm_type=2))