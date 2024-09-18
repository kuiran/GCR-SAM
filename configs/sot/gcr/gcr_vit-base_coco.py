_base_ = [
    './gcr_vit-base.py'
]

find_unused_parameters = True

model = dict(
    rpn_head=dict(
        _delete_=True,
        type='GCR_proposal_gnerator',
        clip_model_name='ViT-B/32',
        clip_feature_path="clip_feature_path"),
)
train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

# coco dataset
data_root = 'data/coco/'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='mmtrack.Coco_add_points_text',
        data_root='data/coco',
        # ann_file='annotations/instances_train2017_add-points_size-range0.5.json',
        ann_file='annotations/instances_train2017_size-range0.25_with-mask.json',
        data_prefix=dict(img='train2017'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline
    )
)



val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='mmtrack.Coco_add_points_text',
        data_root='data/coco',
        ann_file='annotations/instances_val2017_add-points_size-range0.5.json',
        data_prefix=dict(img='val2017'),
        filter_cfg=dict(filter_empty_gt=True, min_size=-1),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017_add-points_size-range0.5.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='save_path'
    )
test_evaluator = val_evaluator