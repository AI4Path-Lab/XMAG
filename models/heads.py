# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

# models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """DINO projection head for pathology distillation."""
    def __init__(
        self, 
        in_dim, 
        hidden_dim=2048,  # Larger for more expressive power
        bottleneck_dim=256,
        out_dim=65536, 
        use_bn=False,
        norm_last_layer=True
    ):
        super().__init__()
        
        # Fixed 3-layer MLP structure
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Use weight normalization for the last layer
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle both single vectors and token sequences
        if len(x.shape) == 3:  # [B, num_tokens, C]
            batch_size, num_tokens, hidden_dim = x.shape
            x_flat = x.reshape(-1, hidden_dim)
            
            # Apply MLP with potential batch norm
            x_flat = self.mlp(x_flat)
            x_flat = nn.functional.normalize(x_flat, dim=-1, p=2)
            x_flat = self.last_layer(x_flat)
            
            return x_flat.reshape(batch_size, num_tokens, -1)
        else:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
            return x


class iBOTHead(nn.Module):
    """iBOT projection head for patch tokens."""
    def __init__(
        self, 
        in_dim, 
        hidden_dim=1024,  # Increased from 384
        bottleneck_dim=256, 
        out_dim=8192,
        use_bn=False,
        norm_last_layer=True
    ):
        super().__init__()
        
        # Create MLP with optional batch normalization
        layers = [nn.Linear(in_dim, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Use weight normalization
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False
            
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle different input shapes (single tokens or sequences)
        if len(x.shape) == 3:  # [B, num_tokens, C]
            batch_size, num_tokens, dim = x.shape
            x_flat = x.reshape(-1, dim)
            
            # Handle batch norm correctly
            x_flat = self.mlp(x_flat)
            x_flat = nn.functional.normalize(x_flat, dim=-1, p=2)
            x_flat = self.last_layer(x_flat)
            
            # Reshape back to original format
            return x_flat.reshape(batch_size, num_tokens, -1)
        else:
            x = self.mlp(x)
            x = nn.functional.normalize(x, dim=-1, p=2)
            x = self.last_layer(x)
            return x

class IdentityHead(nn.Module):
    """Identity head for teacher (if need)."""
    def __init__(self):
        super().__init__()
        self.projector = nn.Identity()

    def forward(self, x):
        return self.projector(x)


class linearhead2(nn.Module):
    """2 layer linear head for student to match the dimension."""
    def __init__(self, in_dim=768, hidden_dim=1536, out_dim=1536):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # Direct expansion
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)  # Identity-like transformation
        )

    def forward(self, x):
        # Handle different input shapes (single tokens or sequences)
        if len(x.shape) == 3:  # [B, num_tokens, C]
            batch_size, num_tokens, dim = x.shape
            x_flat = x.reshape(-1, dim)
            
            # Handle batch norm correctly
            x_flat = self.projector(x_flat)
            
            # Reshape back to original format
            return x_flat.reshape(batch_size, num_tokens, -1)
        else:
            return self.projector(x)

class linearhead1(nn.Module):
    def __init__(self, in_dim=768, out_dim=1536):
        super().__init__()
        self.projector = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        # Handle different input shapes (single tokens or sequences)
        if len(x.shape) == 3:  # [B, num_tokens, C]
            batch_size, num_tokens, dim = x.shape
            x_flat = x.reshape(-1, dim)
            
            # Handle batch norm correctly
            x_flat = self.projector(x_flat)
            
            # Reshape back to original format
            return x_flat.reshape(batch_size, num_tokens, -1)
        else:
            return self.projector(x)