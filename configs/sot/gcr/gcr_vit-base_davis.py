_base_ = [
    './gcr_vit-base_coco.py'
]

find_unused_parameters = True

model = dict(
    rpn_head=dict(
        _delete_=True,
        type='GCR_proposal_gnerator',
        clip_model_name='ViT-B/32',
        clip_feature_path="clip_feature_path"
    )
)

test_pipeline = {{_base_.test_pipeline}}
data_root = 'data/DAVIS/'
val_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='mmtrack.Davis_add_points_text',
        data_root='data/DAVIS/',
        ann_file='2017_val_first_frame_add_class_add-points_size-range0.25_with-mask.json',
        data_prefix=dict(img='JPEGImages/480p'),
        filter_cfg=dict(filter_empty_gt=True, min_size=-1),
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='CocoMetric',
    ann_file=data_root + '2017_val_first_frame_add_class_add-points_size-range0.25_with-mask.json',
    metric='bbox',
    format_only=True,
    outfile_prefix='save_path'
)
test_evaluator = val_evaluator
