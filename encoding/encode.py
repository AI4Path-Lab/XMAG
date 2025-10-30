# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

import torch
import torch.nn as nn
import traceback
from typing import Optional
from torchvision import transforms
from PIL import Image


class UNIv2DistillSmallInferenceEncoder(nn.Module):
    def __init__(self, weights_path: Optional[str] = None):
        super().__init__()
        self.weights_path = weights_path
        self.model, self.eval_transform, self.precision = self._build()

    def _build(self):
        # Build model
        model = StudentEMA(model_name='dinov2_vitb14', embed_dim=768, pretrained=True)

        # Load weights if provided
        if self.weights_path:
            print(f"Loading model from {self.weights_path}")
            try:
                ckpt = torch.load(self.weights_path, map_location='cpu')
                model.load_state_dict(ckpt['state_dict'], strict=True)
                print("✓ Model loaded successfully.")
            except Exception:
                traceback.print_exc()
                raise Exception(f"Failed to load checkpoint from '{self.weights_path}'.")

        # Define preprocessing
        eval_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
        return model, eval_transform, torch.float32

    @torch.inference_mode()
    def encode(self, image_path: str):
        """Encode an image to a 768-D feature."""
        image = Image.open(image_path).convert('RGB')
        x = self.eval_transform(image).unsqueeze(0)
        feats = self.model(x)
        return feats.squeeze(0)


class StudentEMA(nn.Module):
    """Wrapper around DINOv2 Vision Transformer."""
    def __init__(self, model_name='dinov2_vitb14', embed_dim=768, pretrained=True):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=pretrained)

    def forward(self, x):
        out = self.model.forward_features(x)
        return out['x_norm_clstoken']  # (B, C)


if __name__ == "__main__":
    encoder = UNIv2DistillSmallInferenceEncoder(weights_path="path/to/weights.pth")
    feat = encoder.encode("example.jpg")
    print(feat.shape)  # torch.Size([768])
