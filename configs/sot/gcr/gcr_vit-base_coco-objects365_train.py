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
test_pipeline = None

# coco dataset
data_root = 'data/coco/'

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        ignore_keys=['classes', 'palette'],
        datasets=[
            dict(
                type='mmtrack.Coco_add_points_text1',
                data_root='data/coco',
                # ann_file='annotations/instances_train2017_add-points_size-range0.5.json',
                ann_file='annotations/instances_train2017_size-range0.25_with-mask.json',
                data_prefix=dict(img='train2017'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline
            ),
            dict(
                type='mmtrack.Objects365V2Dataset1',
                data_root='data/Objects365',
                # ann_file='Obj365_v2/annotations/zhiyuan_objv2_train_fixname_fixmiss_add-points_size-range0.5.json',
                ann_file='Obj365_v2/annotations/zhiyuan_objv2_train_subset_k500_size-range0.5.json',
                data_prefix=dict(img='Obj365_v2/train/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline
            )
        ]
    )
)

val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = val_evaluator
val_cfg=None
test_cfg=None


