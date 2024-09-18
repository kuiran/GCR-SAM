## Installation
We use conda to mange enviroment, gcr-sam is based on mmtracking-dev-1.x[link](https://github.com/open-mmlab/mmtracking/tree/dev-1.x). You can refer [installation](https://github.com/open-mmlab/mmtracking/blob/dev-1.x/docs/en/get_started.md) to create environment.
## Train
### Preparation
You can refer [generate_clip_feature](https://github.com/kuiran/GCR-SAM/blob/main/utils/dump_clip_features.py) and [generate_point_annotation](https://github.com/kuiran/GCR-SAM/blob/main/utils/huicv/coarse_utils/noise_data_mask_utils1.py) to generate text feature and point annotation

bash ./tools/dist_train.sh \
    {config_path} 8 \
    --work-dir={work_dir}
## Inference
  bash ./tools/dist_test.sh \
      {config_path} \
      8 \
      --checkpoint {mmodel_path} 
