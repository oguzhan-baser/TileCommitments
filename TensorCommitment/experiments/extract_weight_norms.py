#!/usr/bin/env python3
"""
Extract weight norms (L1, L2, L∞) for each layer from model weights.

Saves norms in JSON format matching the structure of score files.
"""

import argparse
import json
import sys
import os
from pathlib import Path
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoConfig

# Add AlphaPruning to path (lib.prune lives in layerSelectionLib/AlphaPruning/lib)
_experiments_dir = Path(__file__).parent
_alphapruning_path = _experiments_dir.parent / "layerSelectionLib" / "AlphaPruning"
if str(_alphapruning_path) not in sys.path:
    sys.path.insert(0, str(_alphapruning_path))

from lib.prune import find_layers


def get_hf_token():
    """Get HuggingFace token from environment variable."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("Error: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Please set it using: export HF_TOKEN=your_token_here", file=sys.stderr)
        sys.exit(1)
    return token


def check_model_cache(model_name, cache_dir):
    """
    Check if model is already cached.
    
    Returns:
        Tuple of (is_cached, cache_path, message)
    """
    cache_path = Path(cache_dir)
    # HuggingFace cache format: models--org--model-name
    cache_name = "models--" + model_name.replace("/", "--")
    model_cache_path = cache_path / cache_name
    
    if model_cache_path.exists():
        # Check if it has snapshots (actual model files)
        snapshots_dir = model_cache_path / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                return True, model_cache_path, f"Model found in cache: {model_cache_path.name}"
    
    return False, None, f"Model not found in cache. Will download to: {cache_path / cache_name}"


def estimate_model_size(model_name):
    """
    Estimate model size in billions of parameters from model name.
    
    Returns:
        Estimated size in billions (float), or None if unknown
    """
    model_lower = model_name.lower()
    
    # Check for explicit size indicators
    if "70b" in model_lower or "70-b" in model_lower:
        return 70.0
    elif "65b" in model_lower or "65-b" in model_lower:
        return 65.0
    elif "30b" in model_lower or "30-b" in model_lower:
        return 30.0
    elif "13b" in model_lower or "13-b" in model_lower:
        return 13.0
    elif "7b" in model_lower or "7-b" in model_lower:
        return 7.0
    elif "3b" in model_lower or "3-b" in model_lower:
        return 3.0
    elif "1.3b" in model_lower or "1.3-b" in model_lower:
        return 1.3
    elif "350m" in model_lower or "350-m" in model_lower:
        return 0.35
    elif "125m" in model_lower or "125-m" in model_lower:
        return 0.125
    
    return None


def get_gpu_memory_info():
    """Get information about available GPUs."""
    if not torch.cuda.is_available():
        return None
    
    info = {
        'count': torch.cuda.device_count(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)  # GB
        info['devices'].append({
            'id': i,
            'name': props.name,
            'total_memory_gb': total_memory
        })
    
    return info


def load_model(model_name, cache_dir=None, device=None, use_multi_gpu=None):
    """
    Load model from HuggingFace with automatic multi-GPU support for large models.
    
    Args:
        model_name: HuggingFace model identifier
        cache_dir: Cache directory for model weights
        device: Device to load model on ('cuda', 'cpu', or None for auto-detection)
        use_multi_gpu: Force multi-GPU (True), single GPU (False), or auto-detect (None)
    """
    # Check for HuggingFace token
    hf_token = get_hf_token()
    
    # Default to layer_selection_integration/llm_weights to avoid duplicate downloads
    if cache_dir is None:
        cache_dir = str(Path(__file__).parent.parent / "layer_selection_integration" / "llm_weights")
    
    # Estimate model size
    model_size = estimate_model_size(model_name)
    is_large_model = model_size is not None and model_size >= 30.0  # 30B+ considered large
    
    # Get GPU information
    gpu_info = get_gpu_memory_info()
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
            if gpu_info:
                print(f"GPU detected: {gpu_info['count']} device(s) available")
                for gpu in gpu_info['devices']:
                    print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory_gb']:.1f} GB)")
        else:
            device = "cpu"
            print("No GPU detected, using CPU")
    
    print(f"Loading model {model_name}...")
    if model_size:
        print(f"Estimated model size: {model_size}B parameters")
    print(f"Using cache directory: {cache_dir}")
    
    # Check if model is already cached
    is_cached, cache_path, cache_msg = check_model_cache(model_name, cache_dir)
    print(cache_msg)
    
    if not is_cached:
        print("Warning: Model not found in cache. This will download the model from HuggingFace.")
        print("This may take a while depending on model size and internet connection.")
    
    # Determine multi-GPU usage
    if device == "cuda" and torch.cuda.is_available():
        if use_multi_gpu is None:
            # Auto-detect: use multi-GPU for large models (30B+) if multiple GPUs available
            use_multi_gpu = is_large_model and gpu_info and gpu_info['count'] > 1
        else:
            use_multi_gpu = use_multi_gpu and gpu_info and gpu_info['count'] > 1
        
        if use_multi_gpu:
            print(f"Using multi-GPU mode: {gpu_info['count']} GPUs")
            # Calculate max memory per GPU (leave some headroom)
            max_memory = {}
            for gpu in gpu_info['devices']:
                # Use 90% of available memory per GPU
                max_memory[gpu['id']] = f"{int(gpu['total_memory_gb'] * 0.9)}GB"
            
            print(f"Max memory per GPU: {max_memory}")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                token=hf_token,
                low_cpu_mem_usage=True,
                device_map="auto",  # Automatically distributes across GPUs
                max_memory=max_memory  # Control memory usage per GPU
            )
        else:
            print("Using single GPU mode")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                token=hf_token,
                low_cpu_mem_usage=True,
                device_map="auto"  # Will use single GPU if only one available
            )
    else:
        # CPU loading
        print("Using CPU mode")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            token=hf_token,
            low_cpu_mem_usage=True
        )
        model = model.to(device)
    
    # Print device placement info if using device_map
    if device == "cuda" and hasattr(model, 'hf_device_map'):
        print("\nModel device placement:")
        device_map = model.hf_device_map
        gpu_usage = {}
        for module_name, device_id in device_map.items():
            if isinstance(device_id, int):
                gpu_usage[device_id] = gpu_usage.get(device_id, 0) + 1
        for gpu_id, module_count in sorted(gpu_usage.items()):
            print(f"  GPU {gpu_id}: {module_count} modules")
    
    model.eval()
    return model


def get_layer_weights(model, model_name):
    """
    Extract weights for each prunable layer.
    Handles both single-GPU and multi-GPU models.
    
    Returns:
        Dictionary mapping layer_name to weight_tensor (all on CPU)
    """
    # Determine architecture
    is_opt = "opt" in model_name.lower()
    
    if is_opt:
        # OPT: layers are in model.model.decoder.layers
        blocks = model.model.decoder.layers
    else:
        # LLaMA/Mistral: layers are in model.model.layers
        blocks = model.model.layers
    
    # Find all prunable layers within the blocks
    layers = find_layers(blocks)
    
    # Create dictionary mapping layer name to weight
    layer_weights = {}
    for name, module in layers.items():
        if hasattr(module, 'weight') and module.weight is not None:
            # Clone and move to CPU (handles both single and multi-GPU)
            # The weight might be on any GPU, so we move it to CPU explicitly
            weight = module.weight.data
            if weight.is_cuda:
                layer_weights[name] = weight.clone().cpu()
            else:
                layer_weights[name] = weight.clone()
    
    return layer_weights


def compute_norms(weight_tensor):
    """
    Compute L1, L2, and L∞ norms of a weight tensor.
    
    Args:
        weight_tensor: torch.Tensor of shape [out_features, in_features]
    
    Returns:
        Tuple of (l1_norm, l2_norm, linf_norm)
    """
    # Flatten to vector for norm computation
    weight_vec = weight_tensor.flatten().float()
    
    # L1 norm: sum of absolute values
    l1_norm = torch.norm(weight_vec, p=1).item()
    
    # L2 norm: Euclidean norm
    l2_norm = torch.norm(weight_vec, p=2).item()
    
    # L∞ norm: maximum absolute value
    linf_norm = torch.norm(weight_vec, p=float('inf')).item()
    
    return l1_norm, l2_norm, linf_norm


def extract_norms_for_model(model_name, scores_file, output_file, cache_dir=None, device=None, use_multi_gpu=None, skip_if_exists=True):
    """
    Extract weight norms for all layers in a model.
    
    Args:
        model_name: HuggingFace model name
        scores_file: Path to scores JSON file (for layer names)
        output_file: Path to save norms JSON
        cache_dir: Cache directory for model weights
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        use_multi_gpu: Force multi-GPU (True), single GPU (False), or auto-detect (None)
        skip_if_exists: Skip extraction if output file already exists (default: True)
    """
    # FIRST: Check if output file already exists BEFORE doing anything else
    output_path = Path(output_file).resolve()  # Resolve to absolute path
    
    if skip_if_exists and output_path.exists():
        print(f"\n{'='*70}")
        print(f"Norm file already exists: {output_path}")
        print(f"{'='*70}")
        
        # Verify the file is valid JSON with expected structure
        try:
            with open(output_path, 'r') as f:
                existing_data = json.load(f)
            if 'layers' in existing_data and 'model_name' in existing_data:
                print(f"✓ Valid norm file found")
                print(f"  Model: {existing_data['model_name']}")
                print(f"  Layers: {len(existing_data['layers'])}")
                print(f"\nSkipping norm extraction. Use --force to overwrite.")
                print(f"{'='*70}\n")
                return existing_data
            else:
                print("Warning: Existing file has invalid structure. Will regenerate.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Error reading existing file ({e}). Will regenerate.")
    
    # Only proceed with extraction if file doesn't exist or is invalid
    print(f"\n{'='*70}")
    print(f"Starting norm extraction for {model_name}")
    print(f"Output file: {output_path}")
    print(f"{'='*70}\n")
    
    # Load scores file to get layer structure
    with open(scores_file, 'r') as f:
        scores_data = json.load(f)
    
    layer_info = scores_data['layers']
    layer_names = [layer['name'] for layer in layer_info]
    
    # Load model (will use multi-GPU for large models automatically)
    model = load_model(model_name, cache_dir=cache_dir, device=device, use_multi_gpu=use_multi_gpu)
    
    # Get layer weights
    print("Extracting layer weights...")
    weight_dict = get_layer_weights(model, model_name)
    
    # Compute norms for each layer
    print("Computing norms...")
    norms_data = {
        'model_name': model_name,
        'num_layers': len(layer_info),
        'layers': []
    }
    
    for layer in layer_info:
        layer_name = layer['name']
        
        if layer_name in weight_dict:
            weight = weight_dict[layer_name]
            l1_norm, l2_norm, linf_norm = compute_norms(weight)
            
            norms_data['layers'].append({
                'index': layer['index'],
                'name': layer_name,
                'norm_1': float(l1_norm),
                'norm_2': float(l2_norm),
                'norm_inf': float(linf_norm),
                'num_parameters': layer.get('num_parameters', 0)
            })
        else:
            print(f"Warning: Layer {layer_name} not found in model weights")
            # Use zeros as fallback
            norms_data['layers'].append({
                'index': layer['index'],
                'name': layer_name,
                'norm_1': 0.0,
                'norm_2': 0.0,
                'norm_inf': 0.0,
                'num_parameters': layer.get('num_parameters', 0)
            })
    
    # Save norms
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(norms_data, f, indent=2)
    
    print(f"Saved norms to {output_path}")
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return norms_data


def main():
    # Check for HuggingFace token early
    if not os.environ.get('HF_TOKEN'):
        print("Error: HF_TOKEN environment variable is not set.", file=sys.stderr)
        print("Please set it using: export HF_TOKEN=your_token_here", file=sys.stderr)
        print("\nTo get your HuggingFace token:", file=sys.stderr)
        print("1. Go to https://huggingface.co/settings/tokens", file=sys.stderr)
        print("2. Create a new token or copy an existing one", file=sys.stderr)
        print("3. Export it: export HF_TOKEN=your_token_here", file=sys.stderr)
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='Extract weight norms from models')
    parser.add_argument('--model', type=str, required=True,
                       help='HuggingFace model name')
    parser.add_argument('--scores_file', type=str, required=True,
                       help='Path to scores JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='Cache directory for model weights (default: ../layer_selection_integration/llm_weights)')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (cuda, cpu, or None for auto-detection)')
    parser.add_argument('--use_multi_gpu', type=lambda x: (str(x).lower() == 'true'),
                       default=None,
                       help='Force multi-GPU (true/false) or auto-detect for large models (None)')
    parser.add_argument('--force', action='store_true',
                       help='Force extraction even if output file already exists')
    
    args = parser.parse_args()
    
    extract_norms_for_model(
        args.model,
        args.scores_file,
        args.output,
        args.cache_dir,
        args.device,
        args.use_multi_gpu,
        skip_if_exists=not args.force
    )


if __name__ == '__main__':
    main()

