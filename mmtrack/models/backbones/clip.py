from typing import Optional

from mmtrack.registry import MODELS
from mmengine.model import BaseModule
import torch
import open_clip
from open_clip import tokenizer


"""
[('RN50', 'openai'), ('RN50', 'yfcc15m'), ('RN50', 'cc12m'), ('RN50-quickgelu', 'openai'), ('RN50-quickgelu', 'yfcc15m'), 
('RN50-quickgelu', 'cc12m'), ('RN101', 'openai'), ('RN101', 'yfcc15m'), ('RN101-quickgelu', 'openai'), 
('RN101-quickgelu', 'yfcc15m'), ('RN50x4', 'openai'), ('RN50x16', 'openai'), ('RN50x64', 'openai'), ('ViT-B-32', 'openai'), 
('ViT-B-32', 'laion400m_e31'), ('ViT-B-32', 'laion400m_e32'), ('ViT-B-32', 'laion2b_e16'), ('ViT-B-32', 'laion2b_s34b_b79k'),
('ViT-B-32-quickgelu', 'openai'), ('ViT-B-32-quickgelu', 'laion400m_e31'), ('ViT-B-32-quickgelu', 'laion400m_e32'), 
('ViT-B-16', 'openai'), ('ViT-B-16', 'laion400m_e31'), ('ViT-B-16', 'laion400m_e32'), ('ViT-B-16', 'laion2b_s34b_b88k'), 
('ViT-B-16-plus-240', 'laion400m_e31'), ('ViT-B-16-plus-240', 'laion400m_e32'), ('ViT-L-14', 'openai'), 
('ViT-L-14', 'laion400m_e31'), ('ViT-L-14', 'laion400m_e32'), ('ViT-L-14', 'laion2b_s32b_b82k'), ('ViT-L-14-336', 'openai'), 
('ViT-H-14', 'laion2b_s32b_b79k'), ('ViT-g-14', 'laion2b_s12b_b42k'), ('ViT-bigG-14', 'laion2b_s39b_b160k'), 
('roberta-ViT-B-32', 'laion2b_s12b_b32k'), ('xlm-roberta-base-ViT-B-32', 'laion5b_s13b_b90k'), 
('xlm-roberta-large-ViT-H-14', 'frozen_laion5b_s13b_b90k'), ('convnext_base', 'laion400m_s13b_b51k'), 
('convnext_base_w', 'laion2b_s13b_b82k'), ('convnext_base_w', 'laion2b_s13b_b82k_augreg'), 
('convnext_base_w', 'laion_aesthetic_s13b_b82k'), ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k'), 
('convnext_base_w_320', 'laion_aesthetic_s13b_b82k_augreg'), ('convnext_large_d', 'laion2b_s26b_b102k_augreg'), 
('convnext_large_d_320', 'laion2b_s29b_b131k_ft'), ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup'), 
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg'), ('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_rewind'), 
('convnext_xxlarge', 'laion2b_s34b_b82k_augreg_soup'), ('coca_ViT-B-32', 'laion2b_s13b_b90k'), 
('coca_ViT-B-32', 'mscoco_finetuned_laion2b_s13b_b90k'), ('coca_ViT-L-14', 'laion2b_s13b_b90k'), 
('coca_ViT-L-14', 'mscoco_finetuned_laion2b_s13b_b90k')]
"""
@MODELS.register_module()
class ClipForClickTrack(BaseModule):
    def __init__(self, 
                 model: str, 
                 pretrained: str,
                 device: Optional[str] = 'cuda', 
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model)
        self.device = device
    
    def forward(self, image, text):
        # replace in SeqCropClickTrack and TrackDataPreprocessor
        # image = self.preprocess(image)
        text = self.tokenizer(text)
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)
        return image_features, text_features