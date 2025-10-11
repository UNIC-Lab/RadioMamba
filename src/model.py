# @Description:
#   Version 13: Simplified version based on v12 Radio-MambaNet architecture.
#   Implements a hybrid U-Net architecture combining a new SS2D Mamba block for global context
#   and a residual convolution block for local features within each stage.
#   Removed multi-scale configuration support for simplified single-model design.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("Warning: `mamba_ssm` package not found. Please install with `pip install mamba-ssm`.")

# --- Helper Modules ---


class ResidualConvBlock(nn.Module):
    """
    A standard residual convolutional block for capturing local features.
    Conv -> Norm -> Act -> Conv -> Norm
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv_branch = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1,
                      groups=dim, bias=False),  # Depthwise
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_branch(x)


class SS2D_Mamba(nn.Module):
    """
    2D Visual Mamba Block with multi-directional scanning.
    This module processes a 2D feature map by scanning it in four directions
    to capture comprehensive spatial dependencies.
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "Mamba-ssm is not installed. Please install it to use SS2D_Mamba.")
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state,
                           d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor):
        """
        x: input tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        # LayerNorm expects (B, ..., C)
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm).permute(0, 3, 1, 2).contiguous()

        # Reshape for sequence processing
        x_seq = x_norm.view(B, C, H * W).transpose(1,
                                                   2).contiguous()  # (B, L, C)

        # Forward scan
        out_fwd = self.mamba(x_seq)

        # Backward scan
        out_bwd = self.mamba(x_seq.flip(dims=[1])).flip(dims=[1])

        # Combine and reshape back to 2D
        x_reconstructed = (out_fwd + out_bwd).transpose(1,
                                                        2).contiguous().view(B, C, H, W)
        return x_reconstructed


class MambaConvBlock(nn.Module):
    """
    The core hybrid block of Radio-MambaNet.
    It contains two parallel branches:
    1. A Mamba branch (SS2D_Mamba) for global, long-range context.
    2. A Convolutional branch (ResidualConvBlock) for local, detailed features.
    """

    def __init__(self, dim: int, d_state: int, d_conv: int, expand: int):
        super().__init__()
        self.mamba_branch = SS2D_Mamba(dim, d_state, d_conv, expand)
        self.conv_branch = ResidualConvBlock(dim)

    def forward(self, x):
        # The input 'x' is fed into both branches simultaneously
        x_mamba = self.mamba_branch(x)
        x_conv = self.conv_branch(x)
        # The outputs are fused by element-wise addition
        return x_mamba + x_conv


# --- Main Model: RadioMambaNet ---

class RadioMambaNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1,
                 dims: List[int] = [32, 64, 128, 256],
                 depths: List[int] = [2, 2, 2, 2],
                 ssm_d_state: int = 16,
                 ssm_d_conv: int = 4,
                 ssm_expand: int = 2):
        super().__init__()
        self.out_channels = out_channels

        # --- Input Projection ---
        self.patch_embed = nn.Conv2d(
            in_channels, dims[0], kernel_size=3, stride=1, padding=1)

        # --- Encoder ---
        self.encoder_stages = nn.ModuleList()
        for i in range(len(dims)):
            stage_blocks = nn.ModuleList([
                MambaConvBlock(dims[i], d_state=ssm_d_state,
                               d_conv=ssm_d_conv, expand=ssm_expand)
                for _ in range(depths[i])
            ])
            self.encoder_stages.append(stage_blocks)

            # Downsampling layer for all but the last stage
            if i < len(dims) - 1:
                downsample = nn.Conv2d(
                    dims[i], dims[i+1], kernel_size=2, stride=2)
                self.encoder_stages.append(downsample)

        # --- Bottleneck ---
        # The last stage of the encoder acts as the bottleneck
        bottleneck_dim = dims[-1]
        self.bottleneck = nn.Sequential(*[
            MambaConvBlock(bottleneck_dim, d_state=ssm_d_state,
                           d_conv=ssm_d_conv, expand=ssm_expand)
            for _ in range(depths[-1])
        ])

        # --- Decoder ---
        self.decoder_stages = nn.ModuleList()
        reversed_dims = dims[::-1]  # [256, 128, 64, 32]

        for i in range(len(reversed_dims) - 1):
            # Upsampling layer
            upsample = nn.ConvTranspose2d(
                reversed_dims[i], reversed_dims[i+1], kernel_size=2, stride=2)
            self.decoder_stages.append(upsample)

            # MambaConv blocks for feature fusion and refinement
            # The input to this block will be concatenated features (skip + upsampled)
            decoder_conv_dim = reversed_dims[i+1] * 2
            fusion_conv = nn.Conv2d(
                decoder_conv_dim, reversed_dims[i+1], kernel_size=1)
            decoder_blocks = nn.ModuleList([
                MambaConvBlock(
                    reversed_dims[i+1], d_state=ssm_d_state, d_conv=ssm_d_conv, expand=ssm_expand)
                # Match encoder depth
                for _ in range(depths[len(dims) - 2 - i])
            ])
            self.decoder_stages.append(
                nn.Sequential(fusion_conv, *decoder_blocks))

        # --- Final Output Layer ---
        self.final_conv = nn.Conv2d(dims[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Input projection
        x = self.patch_embed(x)

        # Encoder path
        skip_connections = []
        for i in range(len(self.encoder_stages)):
            module = self.encoder_stages[i]
            if isinstance(module, nn.ModuleList):  # This is a stage of MambaConvBlocks
                # Save the output of the stage as a skip connection *before* downsampling
                skip_connections.append(x)
                for block in module:
                    x = block(x)
            else:  # This is a downsampling layer
                x = module(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skip_connections.pop()  # The last skip is from the bottleneck level, which we start from
        for i in range(0, len(self.decoder_stages), 2):
            upsample = self.decoder_stages[i]
            decoder_blocks = self.decoder_stages[i+1]

            x = upsample(x)
            skip = skip_connections.pop()

            # Ensure spatial dimensions match before concatenation
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode='bilinear', align_corners=False)

            x = torch.cat([x, skip], dim=1)
            x = decoder_blocks(x)

        # Final output projection
        return self.final_conv(x)


# Alias for compatibility with training script
Model = RadioMambaNet


if __name__ == "__main__":
    import yaml
    import os

    def count_parameters(model):
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def get_model_size(model):
        """计算模型大小（MB）"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return size_all_mb

    def format_params(params):
        """格式化参数数量为友好的显示格式"""
        if params >= 1e9:
            return f"{params/1e9:.2f}B"
        elif params >= 1e6:
            return f"{params/1e6:.2f}M"
        elif params >= 1e3:
            return f"{params/1e3:.2f}K"
        else:
            return str(params)

    # 加载配置文件 (v14)
    config_path = "/mnt/mydisk/hgjia/scr/RadioMambaUnet/config_v14.yaml"

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 获取模型配置 (v14 simplified)
        model_config = config['Model']
        
        print("=" * 80)
        print("RadioMambaNet v14 模型参数统计")
        print("=" * 80)
        
        print(f"基本配置 - 输入通道: {model_config['in_channels']}, 输出通道: {model_config['out_channels']}")
        print(f"描述: {model_config['description']}")
        
        print(f"\n模型配置参数:")
        print(f"  - 维度: {model_config['dims']}")
        print(f"  - 深度: {model_config['depths']}")
        print(f"  - SSM d_state: {model_config['ssm_d_state']}")
        print(f"  - SSM d_conv: {model_config['ssm_d_conv']}")
        print(f"  - SSM expand: {model_config['ssm_expand']}")
        print(f"  - 批次大小: {config['data']['batch_size']}")
        
        try:
            # 创建模型实例
            model = RadioMambaNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                dims=model_config['dims'],
                depths=model_config['depths'],
                ssm_d_state=model_config['ssm_d_state'],
                ssm_d_conv=model_config['ssm_d_conv'],
                ssm_expand=model_config['ssm_expand']
            )

            # 计算参数数量
            total_params = count_parameters(model)
            model_size_mb = get_model_size(model)
            
            print(f"\n实际参数统计:")
            print(f"  - 总参数数量: {total_params:,} ({format_params(total_params)})")
            print(f"  - 模型大小: {model_size_mb:.2f} MB")
            print(f"  - 配置文件预期: {config['model_info']['params']}")
            print(f"  - 描述: {config['model_info']['description']}")
            
            # 验证参数数量是否与配置文件一致
            expected_params = config['model_info']['params']
            if expected_params.endswith('M'):
                expected_num = float(expected_params[:-1]) * 1e6
                if abs(total_params - expected_num) / expected_num < 0.1:  # 10% 误差范围
                    print(f"  ✅ 参数数量与配置文件一致")
                else:
                    print(f"  ⚠️  参数数量与配置文件不一致 (预期: {expected_params})")
            
        except Exception as e:
            if "mamba_ssm" in str(e).lower():
                print(f"  ❌ 需要安装 mamba-ssm 包: pip install mamba-ssm")
            else:
                print(f"  ❌ 模型创建失败: {str(e)}")
        
        # 详细测试模型
        print("\n" + "=" * 80)
        print("详细测试模型")
        print("=" * 80)
        
        try:
            model = RadioMambaNet(
                in_channels=model_config['in_channels'],
                out_channels=model_config['out_channels'],
                dims=model_config['dims'],
                depths=model_config['depths'],
                ssm_d_state=model_config['ssm_d_state'],
                ssm_d_conv=model_config['ssm_d_conv'],
                ssm_expand=model_config['ssm_expand']
            )
            
            # 检查设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用设备: {device}")
            
            # 输入输出尺寸测试
            print("\n输入输出尺寸测试:")
            test_sizes = [(256, 256), (512, 512), (224, 224)]
            
            model.eval()
            
            for h, w in test_sizes:
                try:
                    # 创建测试输入
                    test_input = torch.randn(1, model_config['in_channels'], h, w).to(device)
                    
                    # 将模型移动到设备
                    model = model.to(device)
                    
                    # 前向传播
                    with torch.no_grad():
                        output = model(test_input)
                    
                    print(f"  输入: {test_input.shape} -> 输出: {output.shape}")
                    
                except Exception as e:
                    print(f"  测试尺寸 ({h}, {w}) 失败: {str(e)}")
                    if "mamba_ssm" in str(e).lower() or "cuda" in str(e).lower():
                        print(f"    提示: Mamba模型需要GPU和mamba-ssm包支持")
            
            # 模型架构信息
            print(f"\n模型架构信息:")
            print(f"  编码器阶段数: {len([s for s in model.encoder_stages if isinstance(s, nn.ModuleList)])}")
            print(f"  解码器阶段数: {len(model.decoder_stages)//2}")
            print(f"  瓶颈层块数: {len(model.bottleneck)}")
            
        except Exception as e:
            if "mamba_ssm" in str(e).lower():
                print("错误: 需要安装 mamba-ssm 包")
                print("请运行: pip install mamba-ssm")
            else:
                print(f"模型测试失败: {str(e)}")
        
        # 训练配置信息
        print("\n" + "=" * 80)
        print("训练配置信息")
        print("=" * 80)
        training_config = config['training']
        print(f"学习率: {training_config['learning_rate']}")
        print(f"权重衰减: {training_config['weight_decay']}")
        print(f"损失函数: {training_config['criterion']}")
        print(f"损失权重: {training_config['loss_weights']}")
        print(f"学习率调度耐心值: {training_config['lr_scheduler_patience']}")
        print(f"早停耐心值: {training_config['early_stopping_patience']}")
        print(f"最大训练步数: {config['trainer_config']['max_steps']}")
            
    else:
        print(f"配置文件不存在: {config_path}")
        print("请检查文件路径是否正确")
