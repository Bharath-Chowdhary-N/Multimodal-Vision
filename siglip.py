#SigLip --> Sigmoid Loss for Language Image Pre-Training

from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768, #size of embedding vector of this vision transformer
        intermediate_size=3072, #size of linear layer in the feed forward network
        num_hidden_layers=12, # no. of layers in vision transformers
        num_attention_heads=12,
        num_channels=3, #RGB
        image_size=224, #224,448,896 size of poligemma --> supports size of 224
        patch_size=16,  # each image is divided into patches of size 16x16
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None, #ouput image embedding size
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = SiglipVisionPooler(config)

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config) #initialize a vision transformer with this config
    
    def forward(self, pixel_values) -> Tuple:
        # (Batch, Channel, Height, Width) : --> (Batch, Num_paches, embdeing_size) since (H,W,C => N , P^2*C)
        return self.vision_model(pixel_values) #forward pass
