"""
MLP-Sandwich STU block implementation for ablation experiments.

This module implements the MLP→STU→MLP sandwich architecture with
configurable norm and residual placements.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "SandwichSTUBlock",
]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6, device: Optional[torch.device] = None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model, device=device))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Normalized tensor of same shape
        """
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.scale * x * norm
    
    def reset_parameters(self):
        """Reset parameters to default values."""
        nn.init.ones_(self.scale)


class SwiGLU(nn.Module):
    """SwiGLU activation function: SiLU(W1 x) ⊙ (W2 x)."""
    
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hidden, bias=bias, device=device)
        self.w2 = nn.Linear(d_in, d_hidden, bias=bias, device=device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_in)
        
        Returns:
            Activated tensor of shape (..., d_hidden)
        """
        return F.silu(self.w1(x)) * self.w2(x)


class STUWithDualResidual(nn.Module):
    """
    STU wrapper that optionally adds an internal residual connection.
    This is used for the 'dual' residual mode.
    """
    
    def __init__(self, stu_core, dual_residual: bool = False, dropout: float = 0.0):
        super().__init__()
        self.stu_core = stu_core
        self.dual_residual = dual_residual
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor
        
        Returns:
            Output with optional internal skip connection
        """
        out = self.stu_core(x)
        if self.drop is not None:
            out = self.drop(out)
        if self.dual_residual:
            return x + out
        return out


class SandwichSTUBlock(nn.Module):
    """
    MLP-Sandwich STU block with configurable norm and residual placement.
    
    Architecture: MLP_in → STU → MLP_out
    
    Supports different ablation modes:
    - Pre-Norm vs Post-Norm
    - Outer / Dual / Inner-MLP / Gated residuals
    """
    
    def __init__(
        self,
        layer_id: int,
        config: ModelConfig,
        phi: torch.Tensor,
        n: int,
    ):
        """
        Args:
            layer_id: Layer index
            config: Model configuration
            phi: Spectral filters for STU
            n: Padded sequence length for STU FFT
        """
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.d_model = config.d_model
        
        # Determine hidden size for the sandwich
        # Use mlp_ratio from config (user spec says mlp_ratio=8 for hidden size)
        if config.stu_mlp_hidden_size is not None:
            self.d_hidden = config.stu_mlp_hidden_size
        elif config.mlp_hidden_size is not None:
            self.d_hidden = config.mlp_hidden_size
        else:
            self.d_hidden = config.mlp_ratio * config.d_model
        
        # Ablation settings
        self.prenorm = config.stu_sandwich_prenorm
        self.residual_mode = config.stu_sandwich_residual_mode
        dropout = config.stu_sandwich_dropout
        
        # Build normalization layer
        if config.stu_sandwich_norm_type == "rms":
            self.norm = RMSNorm(config.d_model, device=config.init_device)
        else:  # layernorm
            self.norm = nn.LayerNorm(config.d_model, device=config.init_device)
        
        # MLP_in: d_model → d_hidden (with SwiGLU activation)
        self.mlp_in = SwiGLU(
            config.d_model,
            self.d_hidden,
            bias=config.include_bias,
            device=config.init_device,
        )
        
        # Inner-MLP residual adapter (optional, for residual_mode="inner_mlp")
        self.use_inner_mlp = (self.residual_mode == "inner_mlp")
        if self.use_inner_mlp:
            self.adapter = nn.Linear(
                self.d_hidden,
                config.d_model,
                bias=False,
                device=config.init_device,
            )
        
        # STU core operating in hidden space
        from .stu import STU
        
        dual_residual = (self.residual_mode == "dual")
        self.stu = STUWithDualResidual(
            STU(config, phi, n, feature_dim=self.d_hidden),
            dual_residual=dual_residual,
            dropout=dropout,
        )
        
        # MLP_out: d_hidden → d_model
        self.mlp_out = nn.Linear(
            self.d_hidden,
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.mlp_out._is_residual = True  # type: ignore
        
        # Dropout for outer residual path
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else None
        
        # Gated residual (optional, for residual_mode="gated_outer")
        self.use_gate = (self.residual_mode == "gated_outer")
        if self.use_gate:
            self.gate = nn.Parameter(torch.zeros(1, device=config.init_device))
        
        self._activation_checkpoint_fn: Optional[callable] = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass matching OLMoBlock interface.
        
        Note: Sandwich STU blocks don't use attention_bias, layer_past, or cache,
        but we keep the signature for compatibility.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of (output, None) where output has same shape as input
        """
        residual = x
        
        # Apply norm (pre-norm or pass-through for post-norm)
        if self.prenorm:
            h = self.norm(x)
        else:
            h = x
        
        # MLP_in (with SwiGLU activation)
        h = self.mlp_in(h)
        
        # Optional inner-MLP residual (projects back to model dim)
        if self.use_inner_mlp:
            h_proj = self.adapter(h)
            # Add residual in model space, then continue in hidden space
            # This creates a "highway" around the MLP projection
            residual = residual + h_proj
        
        # STU core (operates in hidden space)
        h = self.stu(h)
        
        # MLP_out (back to model dim)
        h = self.mlp_out(h)
        
        # Apply dropout
        if self.drop is not None:
            h = self.drop(h)
        
        # Apply outer residual connection (with optional gating)
        if self.use_gate:
            y = residual + torch.sigmoid(self.gate) * h
        else:
            y = residual + h
        
        # Post-norm (if selected)
        if not self.prenorm:
            y = self.norm(y)
        
        # Return None for cache to match OLMoBlock interface
        return y, None
    
    def reset_parameters(self):
        """Initialize block parameters using the config's init strategy."""
        from .initialization import init_normal
        
        # Reset norm parameters
        if hasattr(self.norm, 'reset_parameters'):
            self.norm.reset_parameters()
        
        # Initialize STU
        self.stu.stu_core.reset_parameters()
        
        # Initialize MLP projections based on config
        if self.config.init_fn == "normal":
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == "mitchell":
            std = 1.0 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == "full_megatron":
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        
        # Initialize MLP_in (SwiGLU)
        init_normal(self.mlp_in.w1, std, cutoff_factor)
        init_normal(self.mlp_in.w2, std, cutoff_factor)
        
        # Initialize inner-MLP adapter if used
        if self.use_inner_mlp:
            init_normal(self.adapter, std, cutoff_factor)
        
        # Initialize MLP_out with layer-dependent std (for residual path)
        if self.config.init_fn == "mitchell":
            mlp_out_std = 1.0 / math.sqrt(2 * self.mlp_out.in_features * (self.layer_id + 1))
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == "full_megatron":
            mlp_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            mlp_out_std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        
        init_normal(self.mlp_out, mlp_out_std, cutoff_factor)
        
        # Initialize gate (if used) at 0 so model learns to "add in" the sandwich
        if self.use_gate:
            nn.init.zeros_(self.gate)
    
    def set_activation_checkpointing(self, strategy, checkpoint_func=None):
        """Set activation checkpointing for this block."""
        from .config import ActivationCheckpointingStrategy
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            from .model import activation_checkpoint_function
            self._activation_checkpoint_fn = checkpoint_func or activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

