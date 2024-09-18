from mmengine.model import BaseModule
from mmtrack.registry import MODELS
from typing import Dict, List, Tuple, Union

from torch import Tensor, nn
import torch
from mmdet.models.layers import Transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding

from einops import rearrange, reduce, repeat


@MODELS.register_module()
class ClickTrackFormer(Transformer):
    def __init__(self, 
                 encoder=None, 
                 decoder=None, 
                 init_cfg=None):
        super().__init__(encoder=encoder, decoder=decoder, init_cfg=init_cfg)
    
    def forward(self, x: Tensor, mask: Tensor, query_embed: Tensor, pos_embed: Tensor) -> Tuple[Tensor, Tensor]:
        """
        z: template img feature tensor (b, c, h, w)
        x: search img feature (b, c, h, w)
        text: text feature (b, c)
        """
        _, bs, _ = x.shape
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        enc_mem = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=mask)
        target = torch.zeros_like(query_embed)
        # out_dec: [num_dec_layers, num_query, bs, embed_dims]
        out_dec = self.decoder(
            query=target,
            key=enc_mem,
            value=enc_mem,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask
        )
        out_dec = out_dec.transpos(1, 2)
        return out_dec, enc_mem

@MODELS.register_module()
class ClickTrackHead(BaseModule):
    def __init__(self,
                 num_query=1,
                 transformer=None,
                 positional_encoding=dict(
                    type='SinePositionalEncoding',
                    num_feats=128,
                    normalize=True),
                 bbox_head=None,
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 frozen_modules=None,
                 **kwargs):
        super(ClickTrackHead).__init__(init_cfg=init_cfg)
        self.transformer = MODELS.build(transformer)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        assert bbox_head is not None
        self.bbox_head = MODELS.build(bbox_head)
