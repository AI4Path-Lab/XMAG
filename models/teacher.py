# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

# models/teacher.py
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange

from torchvision import transforms
import timm
from huggingface_hub import login, hf_hub_download


# login("***REMOVED***")  # login with your User Access Token, found at https://huggingface.co/settings/tokens


class TeacherWrapper(nn.Module):
    """Wrapper for the UNI teacher model."""
    def __init__(self, checkpoint_path, patch_size=224, device='cuda'):
        super().__init__()

        timm_kwargs = {
                    'model_name': 'vit_giant_patch14_224',
                    'img_size': 224, 
                    'patch_size': 14, 
                    'depth': 24,
                    'num_heads': 24,
                    'init_values': 1e-5, 
                    'embed_dim': 1536,
                    'mlp_ratio': 2.66667*2,
                    'num_classes': 0, 
                    'no_embed_class': True,
                    'mlp_layer': timm.layers.SwiGLUPacked, 
                    'act_layer': torch.nn.SiLU, 
                    'reg_tokens': 8, 
                    'dynamic_img_size': True
                }
        
        self.model = timm.create_model(
            pretrained=False, **timm_kwargs
        )

        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
        
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model.eval()

        self.patch_size = patch_size
    
    def extract_patches(self, x):
        # Extract patches from the input image
        B, C, H, W = x.shape
        assert H == 896 and W == 896, "Input image size must be 896x896"

        # Reshape to extract 4x4 patches of 224x224 each
        patches = rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', 
                        h=4, w=4, p1=self.patch_size, p2=self.patch_size)
        
        return patches

    def forward(self, x):
        with torch.inference_mode():
            # input shape: (B, 3, 896, 896)
            bsize = x.shape[0]

            # Extract 4x4 patches
            x = self.extract_patches(x)  # Shape: (B*16, 3, 224, 224)

            x = self.transform(x)  # Normalize the patches
            
            feats = self.model(x)  # Shape: (B*16, 1536)

            # Reshape to separate batch and patch dimensions
            feats = feats.view(bsize, 16, -1)  # Shape: (B, 16, 1536)

            return feats

        