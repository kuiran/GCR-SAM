from mmdet.registry import MODELS
from mmdet.models.detectors.base import BaseDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig, InstanceList
from mmdet.models.detectors.two_stage import TwoStageDetector
from mmdet.structures import SampleList, OptSampleList, DetDataSample

# sam
from segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.modeling.sam import Sam
from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.prompt_encoder import PromptEncoder

# box_decoder head
from mmengine.model.base_module import BaseModule, ModuleList
from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
from typing import List, Tuple, Any, Dict, Union
from torch import Tensor
from mmdet.models.utils.misc import unpack_gt_instances, empty_instances
import torch
from mmdet.structures.bbox import bbox2roi
from mmengine.structures import InstanceData

from einops import rearrange, reduce, repeat

# text guided bbox_head
from mmdet.models.roi_heads.bbox_heads.dii_head import DIIHead
from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.cnn import build_activation_layer, build_norm_layer, Linear
import torch.nn as nn
from mmengine.model import bias_init_with_prob
from mmdet.structures.bbox import bbox_overlaps

# rpn
import open_clip
import torch.nn.functional as F
from mmdet.registry import TASK_UTILS
from mmtrack.models.sam.sam.mask_decoder import MLP

ForwardResults = Union[Dict[str, torch.Tensor], List[DetDataSample],
                       Tuple[torch.Tensor], torch.Tensor]


@MODELS.register_module(name='TextGuidedBBoxHead', force=True)
class TextGuidedBBoxHead(BBoxHead):
    def __init__(self,
                 num_ffn_fcs: int = 2,
                 num_heads: int = 8,
                 num_pred_iou_fcs: int = 3,
                 num_reg_fcs: int = 3,
                 feedforward_channels: int = 2048,
                 in_channels: int = 256,
                 dropout: float = 0.0,
                 ffn_act_cfg: ConfigType = dict(type='ReLU', inplace=True),
                 guide_conv_cfg: ConfigType = dict(
                    type='GuidedConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')
                 ),
                 loss_pred_iou: ConfigType = dict(type='SmoothL1Loss'),
                 loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
                 with_cls: bool = False, 
                 reg_decoded_bbox: bool = True, 
                 init_cfg: OptConfigType = None,
                 **kwargs) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(with_cls = with_cls,
                         reg_decoded_bbox = reg_decoded_bbox,
                         reg_class_agnostic=True,
                         init_cfg = init_cfg,
                         **kwargs)
        self.loss_pred_iou = MODELS.build(loss_pred_iou)
        self.loss_iou = MODELS.build(loss_iou)
        self.in_channels = in_channels
        self.fp16_enabled = False
        self.attention = MultiheadAttention(in_channels, num_heads, dropout)
        self.attention_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.instance_interactive_conv = MODELS.build(guide_conv_cfg)
        self.instance_interactive_conv_dropout = nn.Dropout(dropout)
        self.instance_interactive_conv_norm = build_norm_layer(
            dict(type='LN'), in_channels
        )[1]

        self.ffn = FFN(
            in_channels,
            feedforward_channels,
            num_ffn_fcs,
            act_cfg=ffn_act_cfg,
            dropout=dropout
        )

        self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.iou_pred_fcs = nn.ModuleList()
        for _ in range(num_pred_iou_fcs):
            self.iou_pred_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False)
            )
            self.iou_pred_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1]
            )
            self.iou_pred_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True))
            )

        # pred iou with Linear
        self.fc_iou = nn.Linear(in_channels, 1)
        self.fc_iou_sigmoid = nn.Sigmoid()

        # pred iou with mlp

        self.reg_fcs = nn.ModuleList()
        for _ in range(num_reg_fcs):
            self.reg_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False)
            )
            self.reg_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1]
            )
            self.reg_fcs.append(
                build_activation_layer(dict(type='ReLU', inplace=True))
            )
        self.fc_reg = nn.Linear(in_channels, 4)

    def init_weights(self) -> None:
        super().init_weights()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            # else:
            #     pass
        # if self.loss_pred_iou.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     nn.init.constant_(self.fc_cls.bias, bias_init)
    
    def forward(self, roi_feat: Tensor, text_feat: Tensor, to_pred_iou: bool=False) -> tuple:
        N, num_proposals = text_feat.shape[:2]

        # self attention
        text_feat = rearrange(text_feat, 'b num_proposal c -> num_proposal b c')
        text_feat = self.attention_norm(self.attention(text_feat))
        attn_text_feats = rearrange(text_feat, 'num_proposal b c -> b num_proposal c')

        # instance interactive
        text_feat = rearrange(attn_text_feats, 'b num_proposal c -> (b num_proposal) c')
        proposal_feat_iic = self.instance_interactive_conv(text_feat, roi_feat)
        proposal_feat = text_feat + self.instance_interactive_conv_dropout(proposal_feat_iic)
        obj_feat = self.instance_interactive_conv_norm(proposal_feat)
        # FFN
        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        if not to_pred_iou:
            reg_feat = obj_feat
            for reg_layer in self.reg_fcs:
                reg_feat = reg_layer(reg_feat)
            # use u mean the dim unchange u=4 or 1
            bbox_delta = rearrange(self.fc_reg(reg_feat), '(b num_proposal) u -> b num_proposal u', b=N)
            return bbox_delta, rearrange(obj_feat, '(b num_proposal) c -> b num_proposal c', b=N), attn_text_feats
        else:
            iou_pred_feat = obj_feat
            for iou_pred_layer in self.iou_pred_fcs:
                iou_pred_feat = iou_pred_layer(iou_pred_feat)
            pred_iou = rearrange(self.fc_iou_sigmoid(self.fc_iou(iou_pred_feat)), '(b num_proposal) u -> b num_proposal u', b=N)
            return pred_iou, rearrange(obj_feat, '(b num_proposal) c -> b num_proposal c', b=N), attn_text_feats

    def refine_bboxes(self,
                      bbox_results: dict,
                      batch_img_metas: List[dict]) -> InstanceList:
        rois = bbox_results['rois']
        bbox_preds = bbox_results['bbox_pred']

        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() <= len(batch_img_metas)

        results_list = []
        for i in range(len(batch_img_metas)):
            inds = torch.nonzero(rois[:, 0] == i, as_tuple=False)
            inds = rearrange(inds, 'num_proposal () -> num_proposal')
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            bbox_preds_ = bbox_preds[inds]
            img_meta_ = batch_img_metas[i]
            
            bboxes = self.regress_single_image(bboxes_, bbox_preds_, img_meta_)

            results = InstanceData(bboxes=bboxes)
            results_list.append(results)
        
        return results_list
    
    def regress_single_image(self, priors: Tensor, bbox_pred: Tensor, img_meta:dict) -> Tensor:
        max_shape = img_meta['img_shape']
        regressed_bboxes = self.bbox_coder.decode(
            priors, bbox_pred, max_shape=max_shape
        )
        return regressed_bboxes

    def loss_and_target(self,
                        iou_pred: Tensor,
                        bbox_pred: Tensor,
                        batch_gt_instances: InstanceList,
                        imgs_whwh: Tensor,) -> dict:
        losses = dict()
        iou_dict = dict()
        avg_factor = bbox_pred.size(0)
        batch_size = len(batch_gt_instances)
        single_pos_num_proposals = bbox_pred.size(0) // batch_size // batch_gt_instances[0].bboxes.size(0)
        # [(b num_proposal) 4]
        gt_bboxes = torch.cat([batch_gt_instance.bboxes for batch_gt_instance in batch_gt_instances])
        if gt_bboxes.size(0) != bbox_pred.size(0):
            gt_bboxes = torch.repeat_interleave(gt_bboxes, dim=0, repeats=single_pos_num_proposals)
        if iou_pred is not None:
            gt_ious = bbox_overlaps(bbox_pred, gt_bboxes, is_aligned=True)
            # bn means batch_size * num_proposal
            gt_ious = rearrange(gt_ious, 'bn -> bn ()')
            split_gt_ious = rearrange(gt_ious, '(b instance_num_proposal) u -> b (instance_num_proposal) u', b=batch_size)
            split_gt_ious = rearrange(split_gt_ious, 'b (instance num_proposal) u -> b instance num_proposal u', num_proposal=single_pos_num_proposals)
            max_gt_iou = reduce(split_gt_ious, 'b instance num_proposal u -> b instance u', 'max')

            split_pred_ious = rearrange(iou_pred, '(b instance_num_proposal) u -> b (instance_num_proposal) u', b=batch_size)
            split_pred_ious = rearrange(split_pred_ious, 'b (instance num_proposal) u -> b instance num_proposal u', num_proposal=single_pos_num_proposals)
            max_iou_pred = reduce(split_pred_ious, 'b instance num_proposal u -> b instance u', 'max')

            # max_gt_iou = reduce(rearrange(gt_ious, '(b num_proposal) u -> b num_proposal u', b=batch_size), 'b num_proposal u -> b u', 'max')
            # max_iou_pred = reduce(rearrange(iou_pred, '(b num_proposal) u -> b num_proposal u', b=batch_size), 'b num_proposal u -> b u', 'max')
            losses['loss_pred_iou'] = self.loss_pred_iou(
                iou_pred,
                gt_ious,
                avg_factor=avg_factor
            )
            iou_dict['max_gt_iou'] = max_gt_iou
            iou_dict['max_iou_pred'] = max_iou_pred
        else:
            imgs_whwh = rearrange(imgs_whwh, 'b num_proposal u -> (b num_proposal) u')
            losses['loss_bbox'] = self.loss_bbox(
                bbox_pred / imgs_whwh,
                gt_bboxes / imgs_whwh,
                avg_factor=avg_factor
            )
            losses['loss_iou'] = self.loss_iou(
                bbox_pred,
                gt_bboxes,
                avg_factor=avg_factor
            )

        return dict(loss_bbox=losses, record_ious=iou_dict)


@MODELS.register_module(name='GCR_box_decoder', force=True)
class GCR_box_decoder(CascadeRoIHead):
    def __init__(self,
                 num_stages: int = 6,
                 stage_loss_weights: Tuple[float]=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel: int = 256,
                 bbox_roi_extractor: ConfigType = dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(
                        type='RoIAlign', output_size=7, sampling_ratio=2
                    ),
                    out_channels=256,
                    featmap_strides=[16]
                 ),
                 bbox_head: ConfigType = dict(
                    type='TextGuidedBBoxHead'   
                 ),
                 mask_head: ConfigType = dict(
                            type='SAMHead',
                            points_per_side=32,
                            points_per_batch=64,
                            prompt_encoder=dict(
                                type='PromptEncoder',
                                embed_dims=256,
                                image_embed_size=64,
                                image_size=1024,
                                mask_channels=16),
                            mask_decoder=dict(
                                type='MaskDecoder',
                                num_multimask_outputs=3,
                                transformer=dict(
                                    type='TwoWayTransformer',
                                    depth=2,
                                    embed_dims=256,
                                    feedforward_channels=2048,
                                    num_heads=8),
                                transformer_dims=256,
                                iou_head_depth=3,
                                iou_head_hidden_dim=256)
                 ),
                 pred_mask = True,
                 fix_text_feature=False,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None
                 ) -> None:
        assert bbox_roi_extractor is not None
        assert bbox_head is not None
        assert len(stage_loss_weights) == num_stages
        super().__init__(
            num_stages=num_stages,
            stage_loss_weights=stage_loss_weights,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg
        )
        self.num_stages = num_stages
        self.proposal_feature_channel = proposal_feature_channel

        self.pred_mask = pred_mask
        if self.pred_mask:
            self.mask_head = MODELS.build(mask_head)
        self.fix_text_feature = fix_text_feature
    
    def bbox_loss(self, 
                  stage: int, 
                  x: Tuple[Tensor],
                  results_list: InstanceList,
                  text_features: Tensor,
                  batch_img_metas: List[dict],
                  batch_gt_instances: InstanceList) -> dict:
        proposal_list = [res.bboxes for res in results_list]
        rois = bbox2roi(proposal_list)
        bbox_results = self._bbox_forward(stage, x, rois, results_list, proposal_list, text_features, batch_img_metas)
        
        imgs_whwh = torch.cat(
            [res.imgs_whwh[None, ...] for res in results_list]
        )

        bbox_head = self.bbox_head[stage]

        iou_pred = bbox_results['iou_pred']
        decoded_bboxes = bbox_results['decoded_bboxes']
        iou_pred = rearrange(iou_pred, 'b num_proposal u -> (b num_proposal) u') if iou_pred is not None else None

        bbox_loss_and_target = bbox_head.loss_and_target(
            iou_pred,
            decoded_bboxes,
            batch_gt_instances,
            imgs_whwh
        )

        bbox_results.update(bbox_loss_and_target)

        # propose for the new proposal_list
        proposal_list = []
        for idx in range(len(batch_img_metas)):
            results = InstanceData()
            if stage == 1:
                if idx == 0:
                    batch_size = len(results_list)
                    single_pos_num_proposals = bbox_results['detached_proposals'][0].size(0) // batch_gt_instances[0].bboxes.size(0)
                    
                    # select max iou pred proposal
                    split_iou_pred = rearrange(iou_pred, '(batch instances_num_proposal) u -> batch instances_num_proposal u', batch=batch_size)
                    split_iou_pred = rearrange(split_iou_pred, 'batch (instances num_proposal) u -> batch instances num_proposal u', num_proposal=single_pos_num_proposals)
                    # split_iou_pred = rearrange(split_iou_pred, 'batch instances num_proposal () -> batch instances num_proposal')
                max_proposal_ind = torch.argmax(split_iou_pred[idx, ...], dim=1).tolist()
                results.imgs_whwh = repeat(results_list[idx].imgs_whwh[0, :].unsqueeze(0), 'one four -> (repeat one) four', repeat=batch_gt_instances[idx].bboxes.size(0))
                split_bboxes = rearrange(bbox_results['detached_proposals'][idx], '(instance num_proposal) u -> instance num_proposal u', num_proposal=single_pos_num_proposals)
                results.bboxes = rearrange([split_bboxes[i, index, :] for i, index in enumerate(max_proposal_ind)], 'instance () u -> instance u')
            else:
                results.imgs_whwh = results_list[idx].imgs_whwh
                results.bboxes = bbox_results['detached_proposals'][idx]
            proposal_list.append(results)
        bbox_results.update(results_list=proposal_list)
        return bbox_results
        
    def _bbox_forward(self, 
                      stage: int, 
                      x: Tuple[Tensor], 
                      rois: Tensor,
                      results_list: List[InstanceData],
                      proposal_list: List[Tensor],
                      text_feats: Tensor,
                      batch_img_metas: List[dict]) -> dict:
        num_imgs = len(batch_img_metas)
        # stage 0 and 1 mean prototype selection stage0: reg stage1: pred_iou
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(x[:bbox_roi_extractor.num_inputs], rois)
        """
        iou_pred: (b num_proposal 1)
        bbox_pred: (b num_proposal 4)
        text_feats: (b num_proposal c)
        """
        if stage == 1:
            iou_pred, img_text_feats, attn_feats = bbox_head(bbox_feats, text_feats, to_pred_iou=True)
            # select proposal corronsping with max iou_pred 
            # max_iou_pred = reduce(iou_pred, 'b num_proposal 1 -> b 1', 'max')
            # max_proposal_ind = rearrange(torch.argmax(iou_pred, dim=1), 'b () -> b')     
            # proposal_list = [proposal_list[i][max_proposal_ind[i].item(), :].unsqueeze(0) for i in range(num_imgs)]
        else:
            max_iou_pred = None
            iou_pred = None
            bbox_pred, img_text_feats, attn_feats = bbox_head(bbox_feats, text_feats)
            fake_bbox_results = dict(
                rois=rois,
                bbox_pred=rearrange(bbox_pred, 'b num_proposal u -> (b num_proposal) u')
            )
            results_list = bbox_head.refine_bboxes(
                bbox_results=fake_bbox_results,
                batch_img_metas=batch_img_metas
            )
        
            proposal_list = [res.bboxes for res in results_list]

        bbox_results = dict(
            # max_iou_pred=max_iou_pred,
            iou_pred=iou_pred,
            decoded_bboxes=torch.cat(proposal_list),
            img_text_feats=img_text_feats,
            attn_feats=attn_feats,
            detached_iou_pred=None if iou_pred is None else [
                iou_pred[i].detach() for i in range(num_imgs)
            ],
            detached_proposals=[item.detach() for item in proposal_list])

        return bbox_results
    
    def loss(self, x: Tuple[Tensor], rpn_results_list: InstanceList, batch_data_samples: SampleList) -> dict:
        losses = {}

        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs

        # num_instances = [batch_gt_instance.bboxes.size(0) for batch_gt_instance in batch_gt_instances]
        # batch_size = len(batch_img_metas)
        # single_pos_num_proposals = rpn_results_list[0].bboxes.size(0) // num_instances[0]
        reduce_text_feats = rpn_results_list[-1].text_features
        for i in range(len(reduce_text_feats)):
            reduce_text_feats[i] = reduce_text_feats[i].unsqueeze(0)
        reduce_text_feats = torch.cat(reduce_text_feats)
        rpn_results_list = rpn_results_list[:-1]
        text_feats = torch.cat([res.pop('text_features')[None, ...] for res in rpn_results_list])

        reduce_text_feats = reduce_text_feats.detach()
        text_feats = text_feats.detach()

        results_list = rpn_results_list

        # prototype selection
        for stage in range(self.num_stages):
            stage_loss_weight = self.stage_loss_weights[stage]
            if stage < 2:
                input_text_feats = text_feats
            elif stage == 2:
                input_text_feats = reduce_text_feats
            else:
                if not self.fix_text_feature:
                    input_text_feats = img_text_feats
                else:
                    input_text_feats = reduce_text_feats
            bbox_results = self.bbox_loss(
                stage=stage,
                x=x,
                results_list=results_list,
                text_features=input_text_feats,
                batch_img_metas=batch_img_metas,
                batch_gt_instances=batch_gt_instances
            )

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{stage}.{name}'] = (
                    value * stage_loss_weight if 'loss' in name else value
                )
            if stage >= 2:
                img_text_feats = bbox_results['img_text_feats']
            results_list = bbox_results['results_list']
        return losses
    
    def predict(self,
                x: Tuple[Tensor],
                rpn_results_list: InstanceList,
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = outputs
        
        reduce_text_feats = rpn_results_list[-1].text_features
        for i in range(len(reduce_text_feats)):
            reduce_text_feats[i] = reduce_text_feats[i].unsqueeze(0)
        reduce_text_feats = torch.cat(reduce_text_feats)
        rpn_results_list = rpn_results_list[:-1]
        text_feats = torch.cat([res.pop('text_features')[None, ...] for res in rpn_results_list])

        results_list = rpn_results_list

        # prototype selection
        for stage in range(self.num_stages):
            proposal_list = [res.bboxes for res in results_list]
            rois = bbox2roi(proposal_list)
            if stage < 2:
                input_text_feats = text_feats
            elif stage == 2:
                input_text_feats = reduce_text_feats
            else:
                if not self.fix_text_feature:
                    input_text_feats = img_text_feats
                else:
                    input_text_feats = reduce_text_feats
            
            bbox_results = self._bbox_forward(
                stage=stage,
                x=x,
                rois=rois,
                results_list=results_list,
                proposal_list=proposal_list,
                text_feats=input_text_feats,
                batch_img_metas=batch_img_metas
            )

            results = InstanceData()
            idx = 0
            batch_size = 1
            if stage == 1:
                # batch_size = 1
                single_pos_num_proposals = bbox_results['detached_proposals'][0].size(0) // batch_gt_instances[0].bboxes.size(0)

                iou_pred = bbox_results['iou_pred']
                iou_pred = rearrange(iou_pred, 'b num_proposal u -> (b num_proposal) u')
                # select max iou pred proposal
                split_iou_pred = rearrange(iou_pred, '(batch instances_num_proposal) u -> batch instances_num_proposal u', batch=batch_size)
                split_iou_pred = rearrange(split_iou_pred, 'batch (instances num_proposal) u -> batch instances num_proposal u', num_proposal=single_pos_num_proposals)
                # idx = 0
                max_proposal_ind = torch.argmax(split_iou_pred[idx, ...], dim=1).tolist()
                results.imgs_whwh = repeat(results_list[idx].imgs_whwh[0, :].unsqueeze(0), 'one four -> (repeat one) four', repeat=batch_gt_instances[idx].bboxes.size(0))
                split_bboxes = rearrange(bbox_results['detached_proposals'][idx], '(instance num_proposal) u -> instance num_proposal u', num_proposal=single_pos_num_proposals)
                results.bboxes = rearrange([split_bboxes[i, index, :] for i, index in enumerate(max_proposal_ind)], 'instance () u -> instance u').detach()
            else:
                results.imgs_whwh = results_list[idx].imgs_whwh
                results.bboxes = bbox_results['detached_proposals'][idx]
            if stage >= 2:
                img_text_feats = bbox_results['img_text_feats']
            results_list = [results]
            # results_list = bbox_results['results_list']
        
        for img_id in range(len(batch_img_metas)):
            bboxes_per_img = proposal_list[img_id]
            if rescale and bboxes_per_img.size(0) > 0:
                assert batch_img_metas[img_id].get('scale_factor') is not None
                scale_factor = bboxes_per_img.new_tensor(
                    batch_img_metas[img_id]['scale_factor']).repeat((1, 2))
                rescale_bboxes_per_img = (
                    bboxes_per_img.view(bboxes_per_img.size(0), -1, 4) /
                    scale_factor).view(bboxes_per_img.size()[0], -1)

        # rescale gt_points
        scale_factor = batch_gt_instances[0].points.new_tensor(batch_img_metas[0]['scale_factor'])
        batch_gt_instances[0].points = batch_gt_instances[0].points / scale_factor.unsqueeze(0)

        # generate mask
        if self.with_mask:
            img_feat = x[0]

            batch_data_samples[0].box_inputs = bboxes_per_img
            results_list = self.mask_head.predict(img_feat,
                                                batch_data_samples,
                                                rescale=True)
            
            results_list[0].bboxes = rescale_bboxes_per_img
            results_list[0].scores = torch.ones(results_list[0].bboxes.size(0)).cuda()
            results_list[0].labels = batch_gt_instances[0].labels
            results_list[0].points = batch_gt_instances[0].points
        else:
            results_list = []
            results = InstanceData()
            results.bboxes = rescale_bboxes_per_img
            results.scores = torch.ones(results.bboxes.size(0)).cuda()
            results.labels = batch_gt_instances[0].labels
            results.points = batch_gt_instances[0].points
            results_list.append(results)

        # debug
        # batch_data_samples[0].box_inputs = bboxes_per_img[0].unsqueeze(0)
        # results_list = self.mask_head.predict(img_feat,
        #                                       batch_data_samples,
        #                                       rescale=True)
        
        # results_list[0].bboxes = rescale_bboxes_per_img[0].unsqueeze(0)
        # results_list[0].scores = torch.ones(results_list[0].bboxes.size(0)).cuda()
        # results_list[0].labels = batch_gt_instances[0].labels[0].unsqueeze(0)
        # calculate iou

        return results_list  

        
    def forward(self, 
                x: Tensor, 
                rpn_results_list: InstanceList, 
                batch_data_samples: SampleList) -> tuple:
        outputs = unpack_gt_instances
        (batch_gt_instaces, batch_gt_instances_ignore, batch_img_metas) = outputs

        all_stage_bbox_results = []
        object_feats = torch.cat(
            [res.pop('features')[None, ...] for res in rpn_results_list]
        )
        results_list = rpn_results_list

        if self.with_bbox:
            for stage in range(self.num_stages):
                bbox_results = self.bbox_loss(
                    stage=stage,
                    x=x,
                    results_list=results_list,
                    object_feats=object_feats,
                    batch_img_metas=batch_img_metas,
                    batch_gt_instaces=batch_gt_instaces
                )
                bbox_results.pop('loss_bbox')
                all_stage_bbox_results.append((bbox_results, ))
        return tuple(all_stage_bbox_results)


class MLP(BaseModule):
    """Very simple multi-layer perceptron (also called FFN) with relu. Mostly
    used in DETR series detectors.

    Args:
        input_dim (int): Feature dim of the input tensor.
        hidden_dim (int): Feature dim of the hidden layer.
        output_dim (int): Feature dim of the output tensor.
        num_layers (int): Number of FFN layers. As the last
            layer of MLP only contains FFN (Linear).
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = ModuleList(
            Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of MLP.

        Args:
            x (Tensor): The input feature, has shape
                (num_queries, bs, input_dim).
        Returns:
            Tensor: The output feature, has shape
                (num_queries, bs, output_dim).
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

ref_dict={
    'RN50': 1024,
    'ViT-B/32': 512
}


import clip


@MODELS.register_module(name='GCR_proposal_gnerator', force=True)
class GCR_proposal_gnerator(BaseModule):
    def __init__(self,
                 clip_model_name: str,
                 clip_feature_path: str,
                 num_classes: int = 1,
                 anchor_generator_cfg: ConfigType = dict(
                     type='AnchorGenerator',
                     strides=[16],
                     ratios=[0.5, 1.0, 2.0],
                     scales=[1, 4, 8]
                 ),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None
                 ):
        super().__init__(init_cfg=init_cfg)
        assert clip_feature_path is not None
        assert clip_model_name is not None
        self.clip_model_name = clip_model_name
        self.clip_feature_path = clip_feature_path
        self.clip_feature_file = torch.load(self.clip_feature_path)
        self.prior_generator = TASK_UTILS.build(anchor_generator_cfg)
        self.num_proposals = self.prior_generator.num_base_priors[0]
        self.clip_dim_adapter = MLP(input_dim=ref_dict[self.clip_model_name], hidden_dim=256, output_dim=256, num_layers=4)

        self.load_clip_complete = False
        
        # not use
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.num_classes = num_classes
    
    def _init_proposals_and_text_feats(self, x: Tuple[Tensor], batch_data_samples: SampleList, test_mode: bool):
        base_proposals = self.prior_generator.gen_base_anchors()[0].to(x[0].device)

        proposals = []
        for i in range(len(batch_data_samples)):
            points = batch_data_samples[i].gt_instances.points
            num_items = points.size(0)
            repeat_base_proposals = repeat(base_proposals, 'n u -> (repeat n) u', repeat=num_items)
            repeat_points = torch.repeat_interleave(points, dim=0, repeats=self.num_proposals)
            repeat_points = repeat(repeat_points, 'n u -> n (repeat u)', repeat=2)
            proposals.append(repeat_base_proposals + repeat_points)
        
        # imgs_whwh
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
        imgs_whwh = []
        for meta in batch_img_metas:
            h, w = meta['img_shape'][:2]
            imgs_whwh.append(x[0].new_tensor([[w, h, w, h]]))
        imgs_whwh = torch.cat(imgs_whwh, dim=0)
        imgs_whwh = imgs_whwh[:, None, :]

        # text
        # train loading exist text feature
        if not test_mode:
            text_lists = []
            for bds in batch_data_samples:
                texts = bds.gt_instances.text.tolist()
                text_list = []
                for t in texts:
                    text_list.append(self.clip_dim_adapter(self.clip_feature_file[t].float().cuda()))
                text_lists.append(rearrange(text_list, 'num_instance 1 c -> (num_instance 1) c'))
        else:
            device = 'cuda'
            if not self.load_clip_complete:
                print('loading clip model')
                self.clip_model, self.clip_preprocess = clip.load(self.clip_model_name, device=device)
                self.load_clip_complete = True
            
            text_lists = []
            for bds in batch_data_samples:
                texts = bds.gt_instances.text.tolist()
                text_list = []
                for t in texts:
                    # change text to others
                    # if t == 'licenseplate':
                    #     t = 'car'
                    _t = clip.tokenize([f'a {t}']).to(device)
                    # _t = clip.tokenize([f'{t}']).to(device)
                    with torch.no_grad():
                        text_feature = self.clip_model.encode_text(_t)
                    text_list.append(self.clip_dim_adapter(
                        text_feature.float()
                    ))
                text_lists.append(rearrange(text_list, 'num_instance 1 c -> (num_instance 1) c'))

        rpn_results_list = []
        for idx in range(len(batch_data_samples)):
            rpn_results = InstanceData()
            rpn_results.bboxes = proposals[idx]
            rpn_results.imgs_whwh = imgs_whwh[idx].repeat(rpn_results.bboxes.size(0), 1)
            rpn_results.text_features = torch.repeat_interleave(text_lists[idx], repeats=self.num_proposals, dim=0)
            # rpn_results.text_features = repeat(text_lists[idx], 'b c -> (repeat b) c', repeat=self.num_proposals)
            rpn_results_list.append(rpn_results)
        origin_text_feat = InstanceData()
        origin_text_feat.text_features = text_lists
        rpn_results_list.append(origin_text_feat)

        return rpn_results_list

    def predict(self, x: Tuple[Tensor], batch_data_samples: SampleList, **kwargs) -> InstanceList:
        return self._init_proposals_and_text_feats(x , batch_data_samples, test_mode=True)
        
    def loss_and_predict(self, x: Tuple[Tensor], batch_data_samples: SampleList, **kwargs) -> tuple:
        rpn_results_list = self._init_proposals_and_text_feats(x, batch_data_samples, test_mode=False)
        return dict(), rpn_results_list


@MODELS.register_module(name='GCR', force=True)
class GCR(TwoStageDetector):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = dict(
                    type='GCR_proposal_gnerator',
                    clip_model_name='RN50',
                    clip_pretrained='openai'
                 ),
                 roi_head: OptConfigType = dict(
                    type='GCR_box_decoder'
                 ),
                 frozen_modules_list=['backbone', 'neck', 'roi_head.mask_head'],
                #  bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor, 
            init_cfg=init_cfg)
        self.frozen_modules_list = frozen_modules_list
        self.frozen_modules()
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        # self.bbox_head = MODELS.build(bbox_head)
    
    def frozen_modules(self):
        # frozen_modules = ['backbone', 'neck', 'roi_head.mask_head']
        for module in self.frozen_modules_list:
            if '.' in module:
                m = getattr(getattr(self, module.split('.')[0]), module.split('.')[1])
            else:
                m = getattr(self, module)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
    
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        with torch.no_grad():
            x = self.backbone(batch_inputs)
            if self.with_neck:
                x = self.neck(x)
            x = tuple(rearrange(x, 'b c h w -> () b c h w'))
        return x
    

import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_activation_layer, build_norm_layer)
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER



@TRANSFORMER.register_module()
class GuidedConv(BaseModule):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=7,
                 with_proj=True,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(GuidedConv, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.input_feat_shape = input_feat_shape
        self.with_proj = with_proj
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.in_channels * self.feat_channels
        self.num_params_out = self.out_channels * self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)

        self.norm_in = build_norm_layer(norm_cfg, self.feat_channels)[1]
        self.norm_out = build_norm_layer(norm_cfg, self.out_channels)[1]

        self.activation = build_activation_layer(act_cfg)

        num_output = self.out_channels * input_feat_shape**2
        if self.with_proj:
            self.fc_layer = nn.Linear(num_output, self.out_channels)
            self.fc_norm = build_norm_layer(norm_cfg, self.out_channels)[1]

    def forward(self, param_feature, input_feature):
        input_feature = input_feature.flatten(2).permute(2, 0, 1)

        input_feature = input_feature.permute(1, 0, 2)
        parameters = self.dynamic_layer(param_feature)

        param_in = parameters[:, :self.num_params_in].view(
            -1, self.in_channels, self.feat_channels)
        param_out = parameters[:, -self.num_params_out:].view(
            -1, self.feat_channels, self.out_channels)

        # input_feature has shape (num_all_proposals, H*W, in_channels)
        # param_in has shape (num_all_proposals, in_channels, feat_channels)
        # feature has shape (num_all_proposals, H*W, feat_channels)
        features = torch.bmm(input_feature, param_in)
        features = self.norm_in(features)
        features = self.activation(features)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = torch.bmm(features, param_out)
        features = self.norm_out(features)
        features = self.activation(features)

        filter_roi_features = features
        if self.with_proj:
            features = features.flatten(1)
            features = self.fc_layer(features)
            features = self.fc_norm(features)
            features = self.activation(features)

        return features, filter_roi_features