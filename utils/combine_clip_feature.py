# combine coco and objects365 text features
import torch
coco_prompt_file_path = 'clip_vit-b-32_coco_prompt-a.pt'
objects365_file_path = 'clip_vit-b-32_objects365_prompt-a_lower.pt'
output_file_path = 'clip_vit-b-32_coco_objects365_prompt-a.pt'

coco_feature = torch.load(coco_prompt_file_path)
objects365_file = torch.load(objects365_file_path)
objects365_keys = objects365_file.keys()
for key in coco_feature.keys():
    if key not in objects365_keys:
        print(f'{key} complete!')
        objects365_file[key] = coco_feature[key]

torch.save(objects365_file, output_file_path)
print('complete!')