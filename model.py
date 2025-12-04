"""
Unified model factory supporting multiple architectures
"""
import torch
import torch.nn as nn
from typing import Optional

from architectures import create_architecture, get_model_info as get_arch_info

def create_model(config) -> nn.Module:
    """
    Create model from configuration.
    
    Supports multiple architectures:
    - PlainUNet: Plain UNet with constant conv blocks per stage
    - ResUNet: ResUNet with progressive residual blocks (nnUNet default)
    - Primus: Vision Transformer-based segmentation
    - MedNeXt: MedNeXt architecture (if available)
    
    Args:
        config: Configuration object with model, data, and training settings
    
    Returns:
        Initialized neural network model
    """
    model = create_architecture(
        architecture=config.model.architecture,
        model_size=config.model.model_size,
        in_channels=config.model.in_channels,
        n_classes=config.model.n_classes,
        patch_size=config.data.patch_size,
        kernel_size=config.model.kernel_size,
        deep_supervision=config.model.deep_supervision,
        primus_patch_embed_size=config.model.primus_patch_embed_size
    )
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get comprehensive model information.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    return get_arch_info(model)


def initialize_weights(model: nn.Module, init_type: str = 'kaiming_normal'):
    """
    Initialize model weights.
    
    Note: This is primarily for CNN architectures (PlainUNet, ResUNet, MedNeXt).
    Transformer architectures (Primus) typically use their own initialization.
    
    Args:
        model: PyTorch model
        init_type: Initialization type ('kaiming_normal', 'kaiming_uniform', 
                   'xavier_normal', 'xavier_uniform')
    """
    def init_func(m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            if init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            if init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(m.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    model.apply(init_func)


if __name__ == "__main__":
    # Test model creation with different architectures
    from config import get_config
    
    config = get_config()
    
    # Test each architecture
    architectures = ['PlainUNet', 'ResUNet', 'Primus']
    
    for arch in architectures:
        print(f"\n{'='*80}")
        print(f"Testing {arch}")
        print('='*80)
        
        config.model.architecture = arch
        config.model.model_size = 'S'  # Use small for quick testing
        
        try:
            model = create_model(config)
            
            # Print model info
            info = get_model_info(model)
            print(f"\n{arch}-{config.model.model_size} Information:")
            for key, value in info.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
            # Test forward pass
            dummy_input = torch.randn(1, 1, *config.data.patch_size)
            print(f"\nInput shape: {dummy_input.shape}")
            
            with torch.no_grad():
                output = model(dummy_input)
                if isinstance(output, list):
                    print("Deep supervision outputs:")
                    for i, out in enumerate(output):
                        print(f"  Output {i}: {out.shape}")
                else:
                    print(f"Output shape: {output.shape}")
            
            print(f"\n✓ {arch} test passed!")
            
        except Exception as e:
            print(f"\n✗ {arch} test failed: {e}")
            import traceback
            traceback.print_exc()