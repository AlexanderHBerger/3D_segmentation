"""
Unified architecture factory supporting multiple network architectures:
- PlainConvUNet: Plain UNet with constant conv blocks per stage
- ResidualEncoderUNet: ResUNet with progressive residual blocks
- Primus: Vision Transformer-based segmentation
- MedNeXt: Existing MedNeXt support

Based on nnUNet's implementation using dynamic_network_architectures.
"""
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Type
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.architectures.primus import (
    PrimusS, PrimusB, PrimusM, PrimusL
)
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm

try:
    import sys
    sys.path.append('/ministorage/ahb/scratch/MedNeXt')
    from nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1
    MEDNEXT_AVAILABLE = True
except ImportError as e:
    MEDNEXT_AVAILABLE = False
    print(f"Warning: MedNeXt not available. Error: {e}")


# Architecture size presets following nnUNet conventions
# These define the complexity of models at different scales

PLAINUNET_CONFIGS = {
    # Model size: (base_features, max_features, n_conv_per_stage)
    'S': {'base_features': 16, 'max_features': 256, 'n_conv_per_stage': 2},
    'B': {'base_features': 32, 'max_features': 320, 'n_conv_per_stage': 2},  # nnUNet default
    'M': {'base_features': 32, 'max_features': 512, 'n_conv_per_stage': 3},
    'L': {'base_features': 48, 'max_features': 512, 'n_conv_per_stage': 3},
}

RESUNET_CONFIGS = {
    # Model size: (base_features, max_features, n_blocks_per_stage_template)
    # n_blocks_per_stage increases in deeper layers (nnUNet pattern)
    'S': {
        'base_features': 16,
        'max_features': 256,
        'n_blocks_per_stage': (1, 2, 3, 4, 4, 4, 4),
    },
    'B': {
        'base_features': 32,
        'max_features': 320,
        'n_blocks_per_stage': (1, 3, 4, 6, 6, 6, 6),  # nnUNet default
    },
    'M': {
        'base_features': 32,
        'max_features': 512,
        'n_blocks_per_stage': (1, 3, 4, 6, 8, 8, 8),
    },
    'L': {
        'base_features': 48,
        'max_features': 512,
        'n_blocks_per_stage': (2, 4, 6, 8, 10, 10, 10),
    },
}

MEDNEXT_CONFIGS = {
    # MedNeXt has built-in size configurations
    # We just map to their naming convention
    'S': 'S',
    'B': 'B',
    'M': 'M',
    'L': 'L',
}


def calculate_n_stages(patch_size: Tuple[int, ...], min_feature_map_size: int = 4) -> int:
    """
    Calculate number of network stages based on patch size.
    
    Following nnUNet pattern: downsample by 2 until feature maps reach minimum size.
    
    Args:
        patch_size: Input patch size (D, H, W) for 3D or (H, W) for 2D
        min_feature_map_size: Minimum feature map size at bottleneck (default: 4)
    
    Returns:
        Number of stages (clamped between 3 and 7)
    """
    # Calculate number of pooling operations per axis
    num_pool_per_axis = []
    for axis_size in patch_size:
        num_pool = 0
        current_size = axis_size
        while current_size >= 2 * min_feature_map_size:
            current_size = current_size / 2
            num_pool += 1
        num_pool_per_axis.append(num_pool)
    
    # Number of stages = 1 (initial) + number of pooling stages
    # Use the minimum across all axes to ensure all feature maps meet the constraint
    n_stages = min(num_pool_per_axis) + 1
    n_stages = max(3, min(7, n_stages))  # Clamp between 3 and 7 (nnUNet limits)
    
    return n_stages


def get_network_parameters(
    architecture: str,
    model_size: str,
    in_channels: int,
    n_classes: int,
    patch_size: Tuple[int, ...],
    primus_patch_embed_size: Tuple[int, int, int],
    kernel_size: Union[int, List[int]] = 3,
    deep_supervision: bool = False,
    conv_op: Optional[Type[nn.Module]] = None,
    norm_op: Optional[Type[nn.Module]] = None,
    dropout_op: Optional[Type[nn.Module]] = None,
    nonlin: Optional[Type[nn.Module]] = None,
) -> dict:
    """
    Generate network parameters for different architectures based on size preset.
    
    Args:
        architecture: One of ['PlainUNet', 'ResUNet', 'Primus', 'MedNeXt']
        model_size: One of ['S', 'B', 'M', 'L']
        in_channels: Number of input channels
        n_classes: Number of output classes
        patch_size: Input patch size (D, H, W) for 3D or (H, W) for 2D
        kernel_size: Convolution kernel size (for CNN architectures)
        deep_supervision: Whether to use deep supervision
        conv_op: Convolution operation (auto-detected if None)
        norm_op: Normalization operation (auto-detected if None)
        dropout_op: Dropout operation (None by default)
        nonlin: Non-linearity (LeakyReLU by default)
        primus_patch_embed_size: Patch embedding size for Primus
    
    Returns:
        Dictionary with architecture-specific parameters
    """
    architecture = architecture.lower()
    model_size = model_size.upper()
    
    # Auto-detect dimension and operations if not provided
    n_dim = len(patch_size)
    if conv_op is None:
        conv_op = convert_dim_to_conv_op(n_dim)
    if norm_op is None:
        norm_op = get_matching_instancenorm(conv_op)
    if nonlin is None:
        nonlin = nn.LeakyReLU
    
    # Calculate number of stages using centralized function
    n_stages = calculate_n_stages(patch_size)
    
    print_feature_map_sizes(architecture, model_size, patch_size, n_stages)

    if architecture in ['plainunet', 'plain_unet', 'plain']:
        return _get_plainunet_params(
            model_size, in_channels, n_classes, n_stages, n_dim,
            kernel_size, deep_supervision, conv_op, norm_op, dropout_op, nonlin
        )
    
    elif architecture in ['resunet', 'residual_unet', 'residual']:
        return _get_resunet_params(
            model_size, in_channels, n_classes, n_stages, n_dim,
            kernel_size, deep_supervision, conv_op, norm_op, dropout_op, nonlin
        )
    
    elif architecture == 'primus':
        return _get_primus_params(
            model_size, in_channels, n_classes, patch_size, deep_supervision,
            primus_patch_embed_size
        )
    
    elif architecture == 'mednext':
        if not MEDNEXT_AVAILABLE:
            raise ImportError("MedNeXt not available. Please install it.")
        return _get_mednext_params(
            model_size, in_channels, n_classes, kernel_size, deep_supervision
        )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Supported: PlainUNet, ResUNet, Primus, MedNeXt")


def _get_plainunet_params(
    model_size: str,
    in_channels: int,
    n_classes: int,
    n_stages: int,
    n_dim: int,
    kernel_size: Union[int, List[int]],
    deep_supervision: bool,
    conv_op: Type[nn.Module],
    norm_op: Type[nn.Module],
    dropout_op: Optional[Type[nn.Module]],
    nonlin: Type[nn.Module],
) -> dict:
    """Generate PlainConvUNet parameters"""
    config = PLAINUNET_CONFIGS[model_size]
    
    # Calculate features per stage (exponentially increasing)
    base_features = config['base_features']
    max_features = config['max_features']
    features_per_stage = [
        min(base_features * (2 ** i), max_features) for i in range(n_stages)
    ]
    
    # Constant conv per stage for PlainUNet
    n_conv_per_stage = [config['n_conv_per_stage']] * n_stages
    n_conv_per_stage_decoder = [config['n_conv_per_stage']] * (n_stages - 1)
    
    # Generate kernel sizes and strides
    if isinstance(kernel_size, int):
        kernel_sizes = [[kernel_size] * n_dim] * n_stages
    else:
        kernel_sizes = [kernel_size] * n_stages
    
    strides = [[1] * n_dim] + [[2] * n_dim] * (n_stages - 1)
    
    return {
        'architecture_class': PlainConvUNet,
        'architecture_kwargs': {
            'input_channels': in_channels,
            'n_stages': n_stages,
            'features_per_stage': features_per_stage,
            'conv_op': conv_op,
            'kernel_sizes': kernel_sizes,
            'strides': strides,
            'n_conv_per_stage': n_conv_per_stage,
            'num_classes': n_classes,
            'n_conv_per_stage_decoder': n_conv_per_stage_decoder,
            'conv_bias': True,
            'norm_op': norm_op,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': dropout_op,
            'dropout_op_kwargs': {'p': 0.0} if dropout_op is not None else None,
            'nonlin': nonlin,
            'nonlin_kwargs': {'inplace': True},
            'deep_supervision': deep_supervision,
        }
    }


def _get_resunet_params(
    model_size: str,
    in_channels: int,
    n_classes: int,
    n_stages: int,
    n_dim: int,
    kernel_size: Union[int, List[int]],
    deep_supervision: bool,
    conv_op: Type[nn.Module],
    norm_op: Type[nn.Module],
    dropout_op: Optional[Type[nn.Module]],
    nonlin: Type[nn.Module],
) -> dict:
    """Generate ResidualEncoderUNet parameters"""
    config = RESUNET_CONFIGS[model_size]
    
    # Calculate features per stage
    base_features = config['base_features']
    max_features = config['max_features']
    features_per_stage = [
        min(base_features * (2 ** i), max_features) for i in range(n_stages)
    ]
    
    # Progressive blocks per stage (nnUNet pattern: more blocks in deeper layers)
    n_blocks_template = config['n_blocks_per_stage']
    n_blocks_per_stage = list(n_blocks_template[:n_stages])
    
    # Decoder uses constant number of blocks
    n_conv_per_stage_decoder = [1] * (n_stages - 1)
    
    # Generate kernel sizes and strides
    if isinstance(kernel_size, int):
        kernel_sizes = [[kernel_size] * n_dim] * n_stages
    else:
        kernel_sizes = [kernel_size] * n_stages
    
    strides = [[1] * n_dim] + [[2] * n_dim] * (n_stages - 1)
    
    return {
        'architecture_class': ResidualEncoderUNet,
        'architecture_kwargs': {
            'input_channels': in_channels,
            'n_stages': n_stages,
            'features_per_stage': features_per_stage,
            'conv_op': conv_op,
            'kernel_sizes': kernel_sizes,
            'strides': strides,
            'n_blocks_per_stage': n_blocks_per_stage,
            'num_classes': n_classes,
            'n_conv_per_stage_decoder': n_conv_per_stage_decoder,
            'conv_bias': True,
            'norm_op': norm_op,
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': dropout_op,
            'dropout_op_kwargs': {'p': 0.0} if dropout_op is not None else None,
            'nonlin': nonlin,
            'nonlin_kwargs': {'inplace': True},
            'deep_supervision': deep_supervision,
        }
    }


def _get_primus_params(
    model_size: str,
    in_channels: int,
    n_classes: int,
    patch_size: Tuple[int, ...],
    deep_supervision: bool,
    primus_patch_embed_size: Tuple[int, int, int],
) -> dict:
    """
    Generate Primus (Vision Transformer) parameters using preset models.
    
    Primus preset models are available as PrimusS, PrimusB, PrimusM, PrimusL.
    These are factory functions that create pre-configured Primus architectures.
    
    Note: deep_supervision is not typically used with transformers,
    but we keep the parameter for API consistency.
    """
    print(f"Creating Primus-{model_size} with patch size {patch_size} and patch embedding size {primus_patch_embed_size}")
    # Map size to preset class
    primus_presets = {
        'S': PrimusS,
        'B': PrimusB,
        'M': PrimusM,
        'L': PrimusL,
    }
    
    if model_size not in primus_presets:
        raise ValueError(f"Primus model size must be one of {list(primus_presets.keys())}")
    
    # Primus requires 3D input
    if len(patch_size) != 3:
        raise ValueError("Primus only supports 3D inputs")
    
    # Patch embedding size - configurable via parameter or adaptive based on input size
    # Smaller patches = more tokens = higher memory but better detail
    # Use the provided patch embedding size
    patch_embed_size = primus_patch_embed_size
    
    return {
        'architecture_class': primus_presets[model_size],
        'architecture_kwargs': {
            'input_channels': in_channels,
            'output_channels': n_classes,  # Primus uses output_channels not num_classes
            'input_shape': patch_size,
            'patch_embed_size': patch_embed_size,
            # Note: deep_supervision not directly supported in Primus
            # but kept for API consistency
        }
    }


def _get_mednext_params(
    model_size: str,
    in_channels: int,
    n_classes: int,
    kernel_size: int,
    deep_supervision: bool,
) -> dict:
    """
    Generate MedNeXt parameters.
    
    Note: MedNeXt is created using create_mednext_v1 factory function,
    not a class constructor. We return a special dict to handle this.
    """
    # MedNeXt handles size internally, we just pass the size string
    return {
        'architecture_class': 'create_mednext_v1',  # Special marker for factory function
        'architecture_kwargs': {
            'num_input_channels': in_channels,
            'num_classes': n_classes,
            'model_id': MEDNEXT_CONFIGS[model_size],
            'kernel_size': kernel_size,
            'deep_supervision': deep_supervision,
        }
    }


def print_feature_map_sizes(
    architecture: str,
    model_size: str,
    patch_size: Tuple[int, ...],
    n_stages: Optional[int] = None
):
    """
    Print feature map sizes for the given architecture configuration.
    
    Args:
        architecture: Architecture name
        model_size: Model size ('S', 'B', 'M', 'L')
        patch_size: Input patch size
        n_stages: Number of stages (if None, will be calculated)
    """
    architecture = architecture.lower()
    model_size = model_size.upper()
    
    # Only applicable for CNN architectures
    if architecture not in ['plainunet', 'plain_unet', 'plain', 'resunet', 'residual_unet', 'residual', 'mednext']:
        return
    
    # Calculate number of stages if not provided
    if n_stages is None:
        n_stages = calculate_n_stages(patch_size)
    
    # Get configuration based on architecture
    if architecture in ['resunet', 'residual_unet', 'residual']:
        config = RESUNET_CONFIGS[model_size]
        arch_name = "ResUNet"
    elif architecture in ['plainunet', 'plain_unet', 'plain']:
        config = PLAINUNET_CONFIGS[model_size]
        arch_name = "PlainUNet"
    else:
        # MedNeXt - skip detailed feature map calculation
        return
    
    base_features = config['base_features']
    max_features = config['max_features']
    
    # Print header
    print(f"\n{'='*80}")
    print(f"Feature Map Sizes - {arch_name}-{model_size}")
    print(f"Input Patch Size: {patch_size}")
    print(f"Number of Stages: {n_stages}")
    print(f"{'='*80}")
    
    # Print encoder stages
    current_spatial_size = list(patch_size)
    for stage in range(n_stages):
        n_features = min(base_features * (2 ** stage), max_features)
        spatial_str = f"({', '.join(str(s) for s in current_spatial_size)})"
        print(f"Stage {stage}: {spatial_str} x {n_features} channels")
        
        # Update spatial size for next stage (downsample by 2)
        if stage < n_stages - 1:
            current_spatial_size = [max(1, s // 2) for s in current_spatial_size]
    
    # Print decoder stages
    print(f"\nDecoder (upsampling path):")
    decoder_stages = n_stages - 1
    
    # Start from the second-to-last encoder stage
    current_spatial_size = [max(1, s // 2) for s in patch_size]
    for i in range(1, n_stages - 1):
        current_spatial_size = [max(1, s // 2) for s in current_spatial_size]
    
    for stage in range(decoder_stages - 1, -1, -1):
        n_features = min(base_features * (2 ** stage), max_features)
        # Upsample
        current_spatial_size = [s * 2 for s in current_spatial_size]
        spatial_str = f"({', '.join(str(s) for s in current_spatial_size)})"
        print(f"Decoder Stage {stage}: {spatial_str} x {n_features} channels")
    
    print(f"{'='*80}\n")


def create_architecture(
    architecture: str,
    model_size: str,
    in_channels: int,
    n_classes: int,
    patch_size: Tuple[int, ...],
    primus_patch_embed_size: Tuple[int, int, int],
    kernel_size: Union[int, List[int]] = 3,
    deep_supervision: bool = False,
    **kwargs
) -> nn.Module:
    """
    Create a network architecture with specified size preset.
    
    Args:
        architecture: Architecture name ('PlainUNet', 'ResUNet', 'Primus', 'MedNeXt')
        model_size: Size preset ('S', 'B', 'M', 'L')
        in_channels: Number of input channels
        n_classes: Number of output classes
        patch_size: Input patch size
        primus_patch_embed_size: Patch embedding size for Primus
        kernel_size: Kernel size for CNN architectures
        deep_supervision: Whether to use deep supervision
        **kwargs: Additional architecture-specific parameters
    Returns:
        Initialized neural network model
    """    
    # Get architecture parameters
    arch_params = get_network_parameters(
        architecture=architecture,
        model_size=model_size,
        in_channels=in_channels,
        n_classes=n_classes,
        patch_size=patch_size,
        kernel_size=kernel_size,
        deep_supervision=deep_supervision,
        primus_patch_embed_size=primus_patch_embed_size,
        **kwargs
    )
    
    # Create model
    architecture_class = arch_params['architecture_class']
    architecture_kwargs = arch_params['architecture_kwargs']
    
    # Special handling for MedNeXt (uses factory function, not class)
    if architecture_class == 'create_mednext_v1':
        if not MEDNEXT_AVAILABLE:
            raise ImportError("MedNeXt not available. Check installation.")
        model = create_mednext_v1(**architecture_kwargs)
    else:
        # Standard class instantiation
        model = architecture_class(**architecture_kwargs)
    
    # Initialize weights (if model has initialize method)
    if hasattr(model, 'initialize'):
        model.apply(model.initialize)
    
    return model


def get_model_info(model: nn.Module) -> dict:
    """
    Get model information including parameter count and size.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'architecture': model.__class__.__name__,
    }


if __name__ == "__main__":
    """Test script to verify all architectures can be created"""
    import torch
    
    # Test configuration
    in_channels = 1
    n_classes = 2
    patch_size_3d = (160, 192, 160)
    patch_size_2d = (256, 256)
    
    print("="*80)
    print("Testing Architecture Creation")
    print("="*80)
    
    # Test PlainUNet
    print("\n" + "="*80)
    print("PlainUNet Architectures")
    print("="*80)
    for size in ['S', 'B', 'M', 'L']:
        try:
            model = create_architecture(
                architecture='PlainUNet',
                model_size=size,
                in_channels=in_channels,
                n_classes=n_classes,
                patch_size=patch_size_3d,
                kernel_size=3,
                deep_supervision=False
            )
            info = get_model_info(model)
            print(f"\nPlainUNet-{size}:")
            print(f"  Parameters: {info['total_params']:,}")
            print(f"  Model Size: {info['model_size_mb']:.2f} MB")
            
            # Test forward pass
            x = torch.randn(1, in_channels, *patch_size_3d)
            with torch.no_grad():
                y = model(x)
            print(f"  Output shape: {y.shape}")
        except Exception as e:
            print(f"PlainUNet-{size} failed: {e}")
    
    # Test ResUNet
    print("\n" + "="*80)
    print("ResidualEncoderUNet Architectures")
    print("="*80)
    for size in ['S', 'B', 'M', 'L']:
        try:
            model = create_architecture(
                architecture='ResUNet',
                model_size=size,
                in_channels=in_channels,
                n_classes=n_classes,
                patch_size=patch_size_3d,
                kernel_size=3,
                deep_supervision=True
            )
            info = get_model_info(model)
            print(f"\nResUNet-{size}:")
            print(f"  Parameters: {info['total_params']:,}")
            print(f"  Model Size: {info['model_size_mb']:.2f} MB")
            
            # Test forward pass
            x = torch.randn(1, in_channels, *patch_size_3d)
            with torch.no_grad():
                y = model(x)
            if isinstance(y, list):
                print(f"  Output shapes (deep supervision): {[yi.shape for yi in y]}")
            else:
                print(f"  Output shape: {y.shape}")
        except Exception as e:
            print(f"ResUNet-{size} failed: {e}")
    
    # Test Primus
    print("\n" + "="*80)
    print("Primus Architectures")
    print("="*80)
    for size in ['S', 'B', 'M', 'L']:
        try:
            model = create_architecture(
                architecture='Primus',
                model_size=size,
                in_channels=in_channels,
                n_classes=n_classes,
                patch_size=patch_size_3d,
                deep_supervision=False
            )
            info = get_model_info(model)
            print(f"\nPrimus-{size}:")
            print(f"  Parameters: {info['total_params']:,}")
            print(f"  Model Size: {info['model_size_mb']:.2f} MB")
            
            # Test forward pass
            x = torch.randn(1, in_channels, *patch_size_3d)
            with torch.no_grad():
                y = model(x)
            print(f"  Output shape: {y.shape}")
        except Exception as e:
            print(f"Primus-{size} failed: {e}")
    
    # Test MedNeXt (if available)
    if MEDNEXT_AVAILABLE:
        print("\n" + "="*80)
        print("MedNeXt Architectures")
        print("="*80)
        for size in ['S', 'B', 'M', 'L']:
            try:
                model = create_architecture(
                    architecture='MedNeXt',
                    model_size=size,
                    in_channels=in_channels,
                    n_classes=n_classes,
                    patch_size=patch_size_3d,
                    kernel_size=3,
                    deep_supervision=False
                )
                info = get_model_info(model)
                print(f"\nMedNeXt-{size}:")
                print(f"  Parameters: {info['total_params']:,}")
                print(f"  Model Size: {info['model_size_mb']:.2f} MB")
                
                # Test forward pass
                x = torch.randn(1, in_channels, *patch_size_3d)
                with torch.no_grad():
                    y = model(x)
                print(f"  Output shape: {y.shape}")
            except Exception as e:
                print(f"MedNeXt-{size} failed: {e}")
    
    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)
