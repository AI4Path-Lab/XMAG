# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

# losses/dino_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOLoss(nn.Module):
    """DINO loss without collapse prevention (for distillation)."""
    def __init__(self, student_temp=0.1, teacher_temp=0.07):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        
    def forward(self, student_output, teacher_output):
        # Apply temperature scaling
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)
        teacher_out = F.softmax(teacher_output / self.teacher_temp, dim=-1)
        
        # KL divergence loss
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()
        
        return loss

class iBOTLoss(nn.Module):
    """iBOT loss for local alignment without masking."""
    def __init__(self, student_temp=0.1, teacher_temp=0.07):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        
    def forward(self, student_patches, teacher_patches):
        """
        INPUT SIZE:
        student_patches: (B, N, C) - Student output for each patch
        teacher_patches: (B, N, C) - Teacher output for each patch
        where B is batch size, N is number of patches, and C is number of classes.
        """
        # Ensure same number of patches
        assert student_patches.shape[1] == teacher_patches.shape[1], \
            "Student and teacher must have same number of patches after mapping"
        
        # Apply temperature scaling
        student_out = F.log_softmax(student_patches / self.student_temp, dim=-1)
        teacher_out = F.softmax(teacher_patches / self.teacher_temp, dim=-1)
        
        # Compute loss for each patch
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()
        
        return loss

class negcosinesim(nn.Module):
    """Cosine similarity loss."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, student_feat, teacher_feat):
        """
        INPUT SIZE:
        student_patches: (xx, C) - Student output
        teacher_patches: (xx, C) - Teacher output
        """
        # Normalize both outputs
        loss = -nn.CosineSimilarity(dim=-1, eps=self.eps)(student_feat, teacher_feat)
        # Take mean over all patches
        loss = loss.mean()
        
        return loss

class negcosinesim_smoothL1(nn.Module):
    """Cosine similarity loss with smoothing."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, student_feat, teacher_feat):
        """
        INPUT SIZE:
        student_patches: (xx, C) - Student output
        teacher_patches: (xx, C) - Teacher output
        """

        teacher_feat = teacher_feat.clone()

        loss1 = (1 - nn.CosineSimilarity(dim=-1, eps=self.eps)(student_feat, teacher_feat).mean()) * 0.9
        loss2 = F.smooth_l1_loss(student_feat, teacher_feat) * 0.1

        return loss1 + loss2