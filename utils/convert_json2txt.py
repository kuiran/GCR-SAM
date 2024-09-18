import os
import argparse
import json
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='pkl2txt')
    parser.add_argument('--json_path')
    parser.add_argument('--txt_save_path')
    parser.add_argument('--txt_name')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    f = open(args.json_path, 'rb')
    a = json.load(f)
    bboxes = []
    for ann in a:
        ann_bbox = np.array(ann['bbox'])
        ann_bbox[2:] = ann_bbox[:2] + ann_bbox[2:]
        bboxes.append(ann_bbox.astype(np.int16).tolist())
    # ori_pred_bboxes = a['ori_pred_bbox']
    # ori_pred_bboxes_1 = [_.int().tolist() for _ in ori_pred_bboxes]
    txt_file = args.txt_save_path
    file_name = args.txt_name
    if not os.path.isdir(txt_file):
        os.makedirs(txt_file)
    with open(os.path.join(txt_file, file_name), 'w') as f:
        for bbox in bboxes:
            # bbox = bbox[0]
            for i in range(4):
                if i < 3:
                    f.write(str(bbox[i]) + ',')
                else:
                    f.write(str(bbox[i]) + '\n')
    f.close()


if __name__ == '__main__':
    main()

# python utils/convert_json2txt.py \
# --json_path lasot_frame1st_results.bbox.json \
# --txt_save_path /sam_vit_base_baseline_1/lasot/ \
# --txt_name lasot_sam.txt
