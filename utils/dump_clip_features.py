# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import json
import torch
import numpy as np
import itertools
# from nltk.corpus import wordnet
import sys

# python utils/dump_clip_features.py --ann data/coco/annotations/instances_train2017_add-points_size-range0.5.json --out_path clip_features/clip_vit-b-32_coco_prompt-a.pt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann', default='datasets/lvis/lvis_v1_val.json')
    parser.add_argument('--out_path', default='')
    parser.add_argument('--prompt', default='a')
    parser.add_argument('--model', default='clip')
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--fix_space', action='store_true')
    parser.add_argument('--use_underscore', action='store_true')
    parser.add_argument('--process_objects365', action='store_true')
    args = parser.parse_args()

    print('Loading', args.ann)
    data = json.load(open(args.ann, 'r'))
    cat_names = [x['name'] for x in \
        sorted(data['categories'], key=lambda x: x['id'])]
    if args.fix_space:
        cat_names = [x.replace('_', ' ') for x in cat_names]
    if args.use_underscore:
        cat_names = [x.strip().replace('/ ', '/').replace(' ', '_') for x in cat_names]
    if args.process_objects365:
        cat_names1 = []
        for x in cat_names:
            if '/' in x:
                cat_names1.append((x.split('/')[0]).lower())
            else:
                cat_names1.append(x.lower())
        cat_names = cat_names1
    print('cat_names', cat_names)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.prompt == 'a':
        sentences = ['a ' + x for x in cat_names]
        # sentences_synonyms = [['a ' + xx for xx in x] for x in synonyms]
    if args.prompt == 'none':
        sentences = [x for x in cat_names]
        # sentences_synonyms = [[xx for xx in x] for x in synonyms]
    elif args.prompt == 'photo':
        sentences = ['a photo of a {}'.format(x) for x in cat_names]
        # sentences_synonyms = [['a photo of a {}'.format(xx) for xx in x] \
        #     for x in synonyms]
    elif args.prompt == 'scene':
        sentences = ['a photo of a {} in the scene'.format(x) for x in cat_names]
        # sentences_synonyms = [['a photo of a {} in the scene'.format(xx) for xx in x] \
        #     for x in synonyms]

    # print('sentences_synonyms', len(sentences_synonyms), \
    #     sum(len(x) for x in sentences_synonyms))
    if args.model == 'clip':
        import clip
        print('Loading CLIP')
        model, preprocess = clip.load(args.clip_model, device=device)
        # if args.avg_synonyms:
        #     sentences = list(itertools.chain.from_iterable(sentences_synonyms))
        #     print('flattened_sentences', len(sentences))
        save_feature = dict()
        for sentence, cat_name in zip(sentences, cat_names):
            text = clip.tokenize([sentence]).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
            text_features = text_features.cpu()
            save_feature[cat_name] = text_features
            print('text_features.shape', text_features.shape)
    else:
        assert 0, args.model
    if args.out_path != '':
        print('saveing to', args.out_path)
        torch.save(save_feature, args.out_path)