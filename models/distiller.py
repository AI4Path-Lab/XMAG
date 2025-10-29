# Copyright (c) 2025 Ziyu Su
# Licensed under the PolyForm Noncommercial License 1.0.0
# See the LICENSE file or https://polyformproject.org/licenses/noncommercial/1.0.0/ for details.

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
# from warmup_scheduler import GradualWarmupScheduler
from einops import rearrange

class PathologyDistiller(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Initialize teacher model
        self.teacher = self._init_teacher()
        
        # Initialize student model
        self.student = self._init_student()
        
        # Initialize EMA student model
        self.student_ema = self._init_student()
        self.ema_decay = config['distillation']['ema_decay']
        
        # Initialize projection heads for DINO and iBOT
        student_global_head = config['model']['student']['global_head']
        teacher_global_head = config['model']['teacher']['global_head']
        student_local_head = config['model']['student']['local_head']
        teacher_local_head = config['model']['teacher']['local_head']
        self.student_global_head = self._init_head(student_global_head, "student")
        self.teacher_global_head = self._init_head(teacher_global_head, "teacher")
        self.student_local_head = self._init_head(student_local_head, "student")
        self.teacher_local_head = self._init_head(teacher_local_head, "teacher")
        
        # Initialize loss functions
        self.global_loss_weight = config['distillation']['global_loss_weight']
        self.local_loss_weight = config['distillation']['local_loss_weight']
        
        # Initialize loss functions
        import losses.losses as loss_module
        self.global_loss_fn = getattr(loss_module, 
            config['distillation']['global_loss']['name'])(config['distillation']['global_loss']['eps'])
        self.local_loss_fn = getattr(loss_module, 
            config['distillation']['local_loss']['name'])(config['distillation']['local_loss']['eps'])
 
        # Freeze teacher model
        if config['model']['teacher']['freeze']:
            for param in self.teacher.parameters():
                param.requires_grad = False
            for param in self.teacher_global_head.parameters():
                param.requires_grad = False
            for param in self.teacher_local_head.parameters():
                param.requires_grad = False
    
    def _init_teacher(self):
        # Import and initialize the teacher model (UNI)
        from models.teacher import TeacherWrapper
        return TeacherWrapper(
            checkpoint_path=self.config['model']['teacher']['checkpoint_path'])
    
    def _init_student(self):
        # Import and initialize the student model (ViT base)
        from models.student import StudentModel
        return StudentModel(
            model_name=self.config['model']['student']['name'],
            embed_dim=self.config['model']['student']['embed_dim'],
            pretrained=self.config['model']['student']['pretrained'],
        )
    
    def _init_head(self, head_type, model_type):
        # Initialize projection head for DINO or iBOT
        import models.heads as heads
    
        if model_type == "teacher":
            in_dim = 1536  # UNI embedding dimension (adjust as needed)
        else:
            in_dim = self.config['model']['student']['embed_dim']
            hidden_dim = self.config['model']['student']['hidden_dim']
            out_dim = self.config['model']['student']['out_dim']
        
        if head_type == "dino":
            return heads.DINOHead(in_dim=in_dim)
        elif head_type == "ibot":
            return heads.iBOTHead(in_dim=in_dim)
        elif head_type == "linear2":
            return heads.linearhead2(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        elif head_type == "identity":
            return heads.IdentityHead()
    
    # def forward(self, x):
    #     # Forward pass through the student model
    #     return self.student(x)
    
    def _update_ema(self):
        # Update the EMA student model
        # Only update EMA on rank 0 to avoid synchronization issues
        # Approach 1: Use PyTorch's built-in distributed utilities (RECOMMENDED)
        if self.trainer.world_size > 1:
            # Only update EMA on rank 0
            if self.trainer.global_rank == 0:
                with torch.no_grad():
                    for ema_param, param in zip(self.student_ema.parameters(), 
                                                self.student.parameters()):
                        ema_param.data = ema_param.data * self.ema_decay + param.data * (1 - self.ema_decay)
            
            # Synchronize EMA parameters across all processes
            for ema_param in self.student_ema.parameters():
                torch.distributed.broadcast(ema_param.data, src=0)
        else:
            # Single GPU case
            with torch.no_grad():
                for ema_param, param in zip(self.student_ema.parameters(), 
                                            self.student.parameters()):
                    ema_param.data = ema_param.data * self.ema_decay + param.data * (1 - self.ema_decay)
    
    def _compute_token_mapping(self, student_tokens):
        # Implement the 4x4 token averaging from student to teacher
        # This will map a 4x4 grid of student tokens to a single teacher token
        B, N, C = student_tokens.shape
        assert N == 256, f"Expected 256 tokens (16x16), got {N}"

        # Reshape to 16x16 grid
        tokens_grid = rearrange(student_tokens, 'b (h w) c -> b h w c', h=16, w=16) # (B, 16, 16, C)

        tokens_block = rearrange(tokens_grid, 'b (h p1) (w p2) c -> b h w p1 p2 c', h=4, w=4, p1=4, p2=4) # (B, 4, 4, 4, 4, C)

        mapped_tokens = tokens_block.mean(dim=(3, 4))  # Average over the 4x4 blocks, (B, 4, 4, C)

        # Reshape to (B, 16, C)
        mapped_tokens = rearrange(mapped_tokens, 'b h w c -> b (h w) c')  # (B, 16, C)
        
        return mapped_tokens 
    
    def training_step(self, batch, batch_idx):
        # Main training step
        high_views, low_views = batch  # Each contains multiple augmented views
        
        # high_views is a list of [batch_size, 3, 896, 896] tensors
        # low_views is a list of [batch_size, 3, 224, 224] tensors
        
        # Process each view
        view1_high, view2_high = high_views[0], high_views[1]
        view1_low, view2_low = low_views[0], low_views[1]
        
        # Teacher processes high-mag views
        with torch.no_grad():
            teacher_features_v1 = self.teacher(view1_high) # (B, 16, 1536)
            teacher_features_v2 = self.teacher(view2_high) # (B, 16, 1536)

        # Global: average the 16 patch features
        teacher_global_v1 = teacher_features_v1.mean(dim=1)  # (B, 1536)
        teacher_global_v2 = teacher_features_v2.mean(dim=1)  # (B, 1536)

        # Project global features
        teacher_global_proj_v1 = self.teacher_global_head(teacher_global_v1) # (B, P)
        teacher_global_proj_v2 = self.teacher_global_head(teacher_global_v2) # (B, P)

        # Project local features
        teacher_local_proj_v1 = self.teacher_local_head(teacher_features_v1) # (B, 16, P')
        teacher_local_proj_v2 = self.teacher_local_head(teacher_features_v2) # (B, 16, P')

        
        # Student processes low-mag views. not using CLS token for now
        student_global_v1, student_tokens_v1 = self.student(view1_low) # (B, 768) [CLS], (B, 16, 768)
        student_global_v2, student_tokens_v2 = self.student(view2_low) # (B, 768) [CLS], (B, 16, 768)
        
        # Project through heads
        # student_global_v1 = student_tokens_v1.mean(dim=1)  # (B, 768)
        # student_global_v2 = student_tokens_v2.mean(dim=1)  # (B, 768)
        student_global_proj_v1 = self.student_global_head(student_global_v1)  # (B, P)
        student_global_proj_v2 = self.student_global_head(student_global_v2)  # (B, P)
        
        # Map student tokens to teacher resolution
        student_local_v1 = self._compute_token_mapping(student_tokens_v1) # (B, 16, 768)
        student_local_v2 = self._compute_token_mapping(student_tokens_v2) # (B, 16, 768)
        student_local_proj_v1 = self.student_local_head(student_local_v1) # (B, 16, P')
        student_local_proj_v2 = self.student_local_head(student_local_v2) # (B, 16, P')
        
        # Compute DINO loss (same-view predictions)
        global_loss = (
            self.global_loss_fn(student_global_proj_v1, teacher_global_proj_v1) +
            self.global_loss_fn(student_global_proj_v2, teacher_global_proj_v2)
        ) / 2
        
        # Compute iBOT loss (same-view predictions as in your plan)
        local_loss = (
            self.local_loss_fn(student_local_proj_v1, teacher_local_proj_v1) +
            self.local_loss_fn(student_local_proj_v2, teacher_local_proj_v2)
        ) / 2
        
        # Total loss
        total_loss = self.global_loss_weight * global_loss + self.local_loss_weight * local_loss
        
        # Update EMA student
        self._update_ema()
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('global_loss', global_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('local_loss', local_loss, on_step=True, on_epoch=True, sync_dist=True)

        # # 添加学习率记录
        # current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        # self.log('learning_rate', current_lr, on_step=True, on_epoch=False, sync_dist=False)
        
        # # 添加批次大小信息（用于验证分布式设置）
        # batch_size = high_views[0].size(0)  # 当前GPU的批大小
        # self.log('batch_size_per_gpu', batch_size, on_step=True, on_epoch=False, sync_dist=False)
        
        # # 添加梯度norm（用于监控训练稳定性）
        # if batch_idx % 10 == 0:  # 每10步记录一次，减少开销
        #     total_norm = 0
        #     for p in self.student.parameters():
        #         if p.grad is not None:
        #             param_norm = p.grad.data.norm(2)
        #             total_norm += param_norm.item() ** 2
        #     total_norm = total_norm ** (1. / 2)
        #     self.log('grad_norm', total_norm, on_step=True, on_epoch=False, sync_dist=True)
        
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        # Similar to training step but using student_ema
        with torch.no_grad():
            high_views, low_views = batch
            view1_high, view2_high = high_views[0], high_views[1]
            view1_low, view2_low = low_views[0], low_views[1]
            
            # Teacher features
            teacher_features_v1 = self.teacher(view1_high) # (B, 16, 1536)
            teacher_features_v2 = self.teacher(view2_high) # (B, 16, 1536)

            # Global: average the 16 patch features
            teacher_global_v1 = teacher_features_v1.mean(dim=1)  # (B, 1536)
            teacher_global_v2 = teacher_features_v2.mean(dim=1)  # (B, 1536)

            # Project global features
            teacher_global_proj_v1 = self.teacher_global_head(teacher_global_v1) # (B, P)
            teacher_global_proj_v2 = self.teacher_global_head(teacher_global_v2) # (B, P)

            # Project local features
            teacher_local_proj_v1 = self.teacher_local_head(teacher_features_v1) # (B, 16, P')
            teacher_local_proj_v2 = self.teacher_local_head(teacher_features_v2) # (B, 16, P')
            
            # Student EMA features
            student_global_v1, student_tokens_v1 = self.student_ema(view1_low) # (B, 768) [CLS], (B, 16, 768)
            student_global_v2, student_tokens_v2 = self.student_ema(view2_low) # (B, 768) [CLS], (B, 16, 768)
            
            # Project through heads
            # student_global_v1 = student_tokens_v1.mean(dim=1)  # (B, 768)
            # student_global_v2 = student_tokens_v2.mean(dim=1)  # (B, 768)
            student_global_proj_v1 = self.student_global_head(student_global_v1)  # (B, P)
            student_global_proj_v2 = self.student_global_head(student_global_v2)  # (B, P)
            
            # Map student tokens to teacher resolution
            student_local_v1 = self._compute_token_mapping(student_tokens_v1) # (B, 16, 768)
            student_local_v2 = self._compute_token_mapping(student_tokens_v2) # (B, 16, 768)
            student_local_proj_v1 = self.student_local_head(student_local_v1) # (B, 16, P')
            student_local_proj_v2 = self.student_local_head(student_local_v2) # (B, 16, P')
            
            # Compute DINO loss (same-view predictions)
            global_loss = (
                self.global_loss_fn(student_global_proj_v1, teacher_global_proj_v1) +
                self.global_loss_fn(student_global_proj_v2, teacher_global_proj_v2)
            ) / 2
            
            # Compute iBOT loss (same-view predictions as in your plan)
            local_loss = (
                self.local_loss_fn(student_local_proj_v1, teacher_local_proj_v1) +
                self.local_loss_fn(student_local_proj_v2, teacher_local_proj_v2)
            ) / 2
            
            # Total loss
            total_loss = self.global_loss_weight * global_loss + self.local_loss_weight * local_loss
            
            self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log('val_global_loss', global_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log('val_local_loss', local_loss, on_step=False, on_epoch=True, sync_dist=True)
            
            return total_loss
    
    def configure_optimizers(self):
        # Only optimize student parameters (teacher is frozen)
        student_params = list(self.student.parameters())
        student_params += list(self.student_global_head.parameters())
        student_params += list(self.student_local_head.parameters())
        
        optimizer = torch.optim.AdamW(
            student_params,
            lr=self.config['optimizer']['lr'],
            weight_decay=self.config['optimizer']['weight_decay']
        )
        
        # Cosine learning rate scheduler with warmup
        scheduler = {
            'scheduler': CosineAnnealingLR(
                optimizer,
                T_max=self.config['scheduler']['max_epochs'],
                eta_min=1e-6
            ),
            'name': 'cosine_annealing',
        }
        
        # Add warmup if specified
        if self.config['scheduler'].get('warmup_epochs', 0) > 0:
            from transformers import get_linear_schedule_with_warmup
            scheduler = {
                'scheduler': get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.config['scheduler']['warmup_epochs'] * 
                                    len(self.trainer.datamodule.train_dataloader()),
                    num_training_steps=self.config['scheduler']['max_epochs'] * 
                                      len(self.trainer.datamodule.train_dataloader())
                ),
                'interval': 'step',
                'frequency': 1,
            }
        
        return [optimizer], [scheduler]