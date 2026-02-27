"""
Layer Selection Integration Pipeline

Extracts per-layer significance scores from AlphaPruning pipeline and formats
them as JSON for CVXPY optimization. Does not modify original AlphaPruning scripts.
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add AlphaPruning to path to import its modules
ALPHAPRUNING_PATH = Path(__file__).parent.parent / "AlphaPruning"
sys.path.insert(0, str(ALPHAPRUNING_PATH))

from transformers import AutoModelForCausalLM, AutoConfig
from lib.prune import find_layers
from lib.esd_utils import get_esd_metrics


def find_existing_metrics(model_name, metric_name):
    """
    Check if metrics already exist in AlphaPruning data directory.
    Returns path to existing metric file if found, None otherwise.
    """
    # Get model short name (e.g., "llama-2-7b-hf" from "meta-llama/Llama-2-7b-hf")
    model_short = model_name.split("/")[-1].lower()
    
    # Common naming variations
    potential_names = set([model_short])
    
    # Handle common variations (llama-2 vs llama2)
    if "llama-2" in model_short:
        potential_names.add(model_short.replace("llama-2", "llama2"))
    elif "llama2" in model_short:
        potential_names.add(model_short.replace("llama2", "llama-2"))
    
    # Check AlphaPruning data directory
    data_dir = ALPHAPRUNING_PATH / "data"
    
    for name in potential_names:
        metric_path = data_dir / name / f"{metric_name}.npy"
        if metric_path.exists():
            return str(metric_path)
    
    return None


def load_or_compute_metrics(model_name, metric_name, cache_dir, metric_cache_dir):
    """
    Load pre-computed raw ESD metrics (alpha values) or compute them on-the-fly.
    
    Note: The .npy files in AlphaPruning/data/ contain raw ESD metrics (alpha values),
    NOT significance scores. Significance scores are computed from these raw metrics
    later in the pipeline.
    """
    # First, check if metrics exist in AlphaPruning data directory
    existing_path = find_existing_metrics(model_name, metric_name)
    
    if existing_path:
        print(f"Using existing raw ESD metrics from AlphaPruning data directory")
        print(f"Loading raw metrics (alpha values) from {existing_path}")
        print(f"  Note: These are raw ESD metrics, not significance scores.")
        print(f"  Significance scores will be computed from these metrics.")
        metrics = np.load(existing_path)
        
        # Also save/copy to requested cache directory if different
        os.makedirs(metric_cache_dir, exist_ok=True)
        requested_path = os.path.join(metric_cache_dir, f"{metric_name}.npy")
        if requested_path != existing_path and not os.path.exists(requested_path):
            # Copy to requested location for future use
            import shutil
            shutil.copy2(existing_path, requested_path)
            print(f"Copied metrics to {requested_path} for future use")
        
        return metrics
    
    # Metrics not found, compute them
    os.makedirs(metric_cache_dir, exist_ok=True)
    metric_path = os.path.join(metric_cache_dir, f"{metric_name}.npy")
    
    if os.path.exists(metric_path):
        print(f"Loading metrics from {metric_path}")
        metrics = np.load(metric_path)
    else:
        print(f"Computing metrics for {model_name}...")
        print("This may take 5-60 minutes depending on model size...")
        metrics = get_esd_metrics(model_name, metric_name, cache_dir)
        np.save(metric_path, metrics)
        print(f"Saved metrics to {metric_path}")
    
    return metrics


def infer_layer_structure_from_metrics(model_name, num_metrics):
    """
    Infer layer structure directly from number of metrics without loading config.
    This works for common architectures when we know the pattern.
    """
    model_lower = model_name.lower()
    
    # Common patterns: 224 metrics = 32 blocks * 7 layers (LLaMA-7B)
    #                  72 metrics = 12 blocks * 6 layers (OPT-125M)
    #                  280 metrics = 40 blocks * 7 layers (LLaMA-13B)
    
    is_opt = "opt" in model_lower
    
    if is_opt:
        # OPT: 6 layers per block
        if num_metrics % 6 != 0:
            return None, None, None
        blocks_count = num_metrics // 6
        
        # Common OPT sizes
        if blocks_count == 12:  # OPT-125M
            hidden_size = 768
            intermediate_size = 3072
        elif blocks_count == 24:  # OPT-350M
            hidden_size = 1024
            intermediate_size = 4096
        elif blocks_count == 32:  # OPT-1.3B
            hidden_size = 2048
            intermediate_size = 8192
        else:
            # Default: infer from common ratios
            hidden_size = 768
            intermediate_size = 3072
        
        layer_info = []
        for block_idx in range(blocks_count):
            for attn_name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']:
                layer_info.append({
                    'name': f"{block_idx}.{attn_name}",
                    'num_parameters': hidden_size * hidden_size,
                    'weight_shape': [hidden_size, hidden_size]
                })
            layer_info.append({
                'name': f"{block_idx}.fc1",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [intermediate_size, hidden_size]
            })
            layer_info.append({
                'name': f"{block_idx}.fc2",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [hidden_size, intermediate_size]
            })
        
        return layer_info, True, blocks_count
    
    else:
        # LLaMA/Mistral: 7 layers per block
        if num_metrics % 7 != 0:
            return None, None, None
        blocks_count = num_metrics // 7
        
        # Common LLaMA sizes
        if blocks_count == 32:  # LLaMA-7B
            hidden_size = 4096
            intermediate_size = 11008
        elif blocks_count == 40:  # LLaMA-13B
            hidden_size = 5120
            intermediate_size = 13824
        elif blocks_count == 80:  # LLaMA-65B
            hidden_size = 8192
            intermediate_size = 22016
        else:
            # Default: infer from common ratios
            hidden_size = 4096
            intermediate_size = 11008
        
        layer_info = []
        for block_idx in range(blocks_count):
            for attn_name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']:
                layer_info.append({
                    'name': f"{block_idx}.{attn_name}",
                    'num_parameters': hidden_size * hidden_size,
                    'weight_shape': [hidden_size, hidden_size]
                })
            layer_info.append({
                'name': f"{block_idx}.mlp.gate_proj",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [intermediate_size, hidden_size]
            })
            layer_info.append({
                'name': f"{block_idx}.mlp.up_proj",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [intermediate_size, hidden_size]
            })
            layer_info.append({
                'name': f"{block_idx}.mlp.down_proj",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [hidden_size, intermediate_size]
            })
        
        return layer_info, False, blocks_count


def get_layer_structure_from_config(model_name, config, num_metrics):
    """
    Infer layer structure and parameter counts from config without loading full model.
    
    Args:
        model_name: Model identifier
        config: Model configuration object
        num_metrics: Number of metric values (to verify consistency)
    
    Returns:
        layer_info, is_opt, blocks_count
    """
    # Determine architecture type
    is_opt = "opt" in model_name.lower() or hasattr(config, 'model_type') and config.model_type == 'opt'
    
    # Get number of blocks
    if hasattr(config, 'num_hidden_layers'):
        blocks_count = config.num_hidden_layers
    elif hasattr(config, 'num_layers'):
        blocks_count = config.num_layers
    else:
        raise ValueError(f"Cannot determine number of layers from config")
    
    # Get hidden size
    hidden_size = config.hidden_size
    
    layer_info = []
    
    if is_opt:
        # OPT architecture: 6 prunable layers per block
        # self_attn.q_proj, k_proj, v_proj, out_proj, fc1, fc2
        num_heads = config.num_attention_heads
        intermediate_size = config.ffn_dim if hasattr(config, 'ffn_dim') else config.intermediate_size
        
        # Attention projections: hidden_size x hidden_size each
        attn_proj_size = hidden_size  # q, k, v, out_proj are all hidden_size x hidden_size
        
        for block_idx in range(blocks_count):
            # Attention layers
            for attn_name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.out_proj']:
                layer_info.append({
                    'name': f"{block_idx}.{attn_name}",
                    'num_parameters': hidden_size * hidden_size,
                    'weight_shape': [hidden_size, hidden_size]
                })
            
            # MLP layers
            layer_info.append({
                'name': f"{block_idx}.fc1",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [intermediate_size, hidden_size]
            })
            layer_info.append({
                'name': f"{block_idx}.fc2",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [hidden_size, intermediate_size]
            })
    else:
        # LLaMA/Mistral architecture: 7 prunable layers per block
        # self_attn.q_proj, k_proj, v_proj, o_proj, mlp.gate_proj, up_proj, down_proj
        num_heads = config.num_attention_heads
        num_key_value_heads = getattr(config, 'num_key_value_heads', num_heads)
        intermediate_size = config.intermediate_size
        
        # Calculate head dimensions
        head_dim = hidden_size // num_heads
        
        for block_idx in range(blocks_count):
            # Attention layers
            # q_proj, k_proj, v_proj, o_proj are all hidden_size x hidden_size
            for attn_name in ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj']:
                layer_info.append({
                    'name': f"{block_idx}.{attn_name}",
                    'num_parameters': hidden_size * hidden_size,
                    'weight_shape': [hidden_size, hidden_size]
                })
            
            # MLP layers
            layer_info.append({
                'name': f"{block_idx}.mlp.gate_proj",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [intermediate_size, hidden_size]
            })
            layer_info.append({
                'name': f"{block_idx}.mlp.up_proj",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [intermediate_size, hidden_size]
            })
            layer_info.append({
                'name': f"{block_idx}.mlp.down_proj",
                'num_parameters': hidden_size * intermediate_size,
                'weight_shape': [hidden_size, intermediate_size]
            })
    
    # Verify consistency with number of metrics
    if len(layer_info) != num_metrics:
        print(f"Warning: Number of layers inferred from config ({len(layer_info)}) "
              f"does not match number of metrics ({num_metrics}). "
              f"Falling back to loading full model.")
        return None, None, None
    
    return layer_info, is_opt, blocks_count


def get_layer_structure(model):
    """Extract layer structure and parameter counts from loaded model."""
    if "opt" in str(type(model)).lower() or "OPT" in model.__class__.__name__:
        blocks = model.model.decoder.layers
        is_opt = True
    else:
        blocks = model.model.layers
        is_opt = False
    
    # Find all prunable layers
    layers = find_layers(blocks)
    
    layer_info = []
    for name, layer_module in layers.items():
        num_params = layer_module.weight.numel()
        layer_info.append({
            'name': name,
            'num_parameters': int(num_params),
            'weight_shape': list(layer_module.weight.shape)
        })
    
    return layer_info, is_opt, len(blocks)


def compute_pruning_ratios(metrics, layer_info, blocks_count, sparsity_ratio, epsilon, mapping_type='block_wise'):
    """Compute pruning ratios using same logic as AlphaPruning's ww_sparsity()."""
    # Calculate layers per block
    layer_num_in_block = int(len(layer_info) / blocks_count) if blocks_count > 0 else len(layer_info)
    
    # Handle block-wise mapping if requested
    if mapping_type == 'block_wise' and layer_num_in_block > 0:
        block_metrics = []
        for i in range(0, len(metrics), layer_num_in_block):
            block_metrics.append(np.mean(metrics[i:i+layer_num_in_block]))
        # Repeat block metric for each layer in the block
        metrics = [m for m in block_metrics for _ in range(layer_num_in_block)]
    
    scores = torch.tensor(metrics, dtype=torch.float32)
    prunables = torch.tensor([info['num_parameters'] for info in layer_info], dtype=torch.float32)
    
    # Linear mapping
    s1 = 1.0 - epsilon
    s2 = 1.0 + epsilon
    
    min_score = torch.min(scores)
    max_score = torch.max(scores)
    
    if max_score > min_score:
        layerwise_pruning_ratios = (((scores - min_score) / (max_score - min_score)) * (s2 - s1) + s1)
    else:
        layerwise_pruning_ratios = torch.full_like(scores, (s1 + s2) / 2)
    
    # Scale to match target sparsity
    scaler = torch.sum(prunables) * sparsity_ratio / torch.sum(prunables * layerwise_pruning_ratios)
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    layerwise_pruning_ratios = torch.clamp(layerwise_pruning_ratios, 0.0, 1.0)
    
    return layerwise_pruning_ratios.cpu().numpy().tolist()


def compute_significance_scores(pruning_ratios, method='inverse_ratio'):
    """
    Compute significance scores from pruning ratios.
    
    Args:
        pruning_ratios: List of pruning ratios (0-1)
        method: 'inverse_ratio' (1 - ratio) or 'metric_based'
    
    Returns:
        List of raw significance scores (higher = more significant)
    """
    if method == 'inverse_ratio':
        # Inverse pruning ratio: less pruning = more significant
        scores = [1.0 - ratio for ratio in pruning_ratios]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return scores


def extract_layer_scores(model_name, metric_name='alpha_peak', cache_dir='llm_weights',
                         metric_cache_dir=None, sparsity_ratio=0.7, epsilon=0.3,
                         mapping_type='block_wise', significance_method='inverse_ratio'):
    """
    Main function to extract layer significance scores.
    
    Returns:
        Dictionary with layer information and scores
    """
    print(f"Extracting layer scores for model: {model_name}")
    print(f"Metric: {metric_name}, Sparsity: {sparsity_ratio}, Epsilon: {epsilon}")
    
    # Setup metric cache directory
    if metric_cache_dir is None:
        model_short_name = model_name.split("/")[-1].lower()
        metric_cache_dir = os.path.join(str(ALPHAPRUNING_PATH), "data", model_short_name)
    
    # Load or compute raw ESD metrics (alpha values) from .npy files
    # Note: These are NOT significance scores - they're raw ESD metrics that need
    # to be converted to pruning ratios and then to significance scores
    metrics = load_or_compute_metrics(model_name, metric_name, cache_dir, metric_cache_dir)
    print(f"Loaded {len(metrics)} raw ESD metric values (alpha values)")
    
    # Try to infer layer structure from metrics first (no model/config needed)
    print("Inferring layer structure from metrics...")
    layer_info, is_opt, blocks_count = infer_layer_structure_from_metrics(model_name, len(metrics))
    
    # If inference failed, try loading config (lightweight)
    if layer_info is None:
        print("Trying to load model config (lightweight)...")
        try:
            config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
            layer_info, is_opt, blocks_count = get_layer_structure_from_config(
                model_name, config, len(metrics)
            )
            if layer_info is None:
                raise ValueError("Config-based inference failed")
            print(f"Inferred {len(layer_info)} prunable layers in {blocks_count} transformer blocks from config")
        except Exception as e:
            print(f"Could not load config: {e}")
            print("Falling back to loading full model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                device_map="cpu"
            )
            layer_info, is_opt, blocks_count = get_layer_structure(model)
            print(f"Found {len(layer_info)} prunable layers in {blocks_count} transformer blocks")
    else:
        print(f"Inferred {len(layer_info)} prunable layers in {blocks_count} transformer blocks from metrics pattern")
    
    # Convert raw ESD metrics to pruning ratios
    print("Converting raw ESD metrics to pruning ratios...")
    pruning_ratios = compute_pruning_ratios(
        metrics, layer_info, blocks_count, sparsity_ratio, epsilon, mapping_type
    )
    
    # Compute significance scores from pruning ratios (raw values)
    print(f"Computing significance scores from pruning ratios (method: {significance_method})...")
    significance_scores = compute_significance_scores(pruning_ratios, significance_method)
    
    # Prepare output
    layers_data = []
    total_params = 0
    for idx, (info, ratio, score) in enumerate(zip(layer_info, pruning_ratios, significance_scores)):
        layers_data.append({
            'index': idx,
            'name': info['name'],
            'num_parameters': info['num_parameters'],
            'significance_score': float(score),  # Raw value
            'pruning_ratio': float(ratio),
            'weight_shape': info['weight_shape']
        })
        total_params += info['num_parameters']
    
    # Calculate block-level info
    layer_num_in_block = len(layer_info) // blocks_count if blocks_count > 0 else 1
    block_info = []
    for block_idx in range(blocks_count):
        start_idx = block_idx * layer_num_in_block
        end_idx = start_idx + layer_num_in_block
        block_layers = layers_data[start_idx:end_idx]
        block_info.append({
            'block_index': block_idx,
            'start_layer_index': start_idx,
            'end_layer_index': end_idx - 1,
            'num_layers': len(block_layers),
            'total_parameters': sum(l['num_parameters'] for l in block_layers),
            'avg_significance': np.mean([l['significance_score'] for l in block_layers]),
            'avg_pruning_ratio': np.mean([l['pruning_ratio'] for l in block_layers])
        })
    
    output = {
        'model_name': model_name,
        'metric_name': metric_name,
        'sparsity_ratio': sparsity_ratio,
        'epsilon': epsilon,
        'mapping_type': mapping_type,
        'significance_method': significance_method,
        'total_parameters': total_params,
        'num_blocks': blocks_count,
        'num_prunable_layers': len(layer_info),
        'layers_per_block': layer_num_in_block,
        'is_opt_model': is_opt,
        'layers': layers_data,
        'blocks': block_info,
        'metadata': {
            'raw_metrics_min': float(np.min(metrics)),
            'raw_metrics_max': float(np.max(metrics)),
            'raw_metrics_mean': float(np.mean(metrics)),
            'pruning_ratios_min': float(np.min(pruning_ratios)),
            'pruning_ratios_max': float(np.max(pruning_ratios)),
            'significance_scores_min': float(np.min(significance_scores)),
            'significance_scores_max': float(np.max(significance_scores)),
            'significance_scores_mean': float(np.mean(significance_scores))
        }
    }
    
    return output


def main():
    parser = argparse.ArgumentParser(
        description='Extract layer significance scores from AlphaPruning pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Model identifier (HuggingFace Hub name or path)')
    parser.add_argument('--ww_metric', type=str, default='alpha_peak',
                       help='ESD metric name (alpha_peak, alpha_mid, etc.)')
    parser.add_argument('--cache_dir', type=str, default='llm_weights',
                       help='Directory to cache model weights')
    parser.add_argument('--ww_metric_cache', type=str, default=None,
                       help='Directory for metric cache (default: AlphaPruning/data/<model_name>/)')
    parser.add_argument('--sparsity_ratio', type=float, default=0.85,
                       help='Target sparsity ratio')
    parser.add_argument('--epsilon', type=float, default=0.9,
                       help='Epsilon for pruning ratio allocation')
    parser.add_argument('--mapping_type', type=str, default='block_wise',
                       choices=['block_wise', 'layer_wise'],
                       help='Mapping type for pruning ratios')
    parser.add_argument('--significance_method', type=str, default='inverse_ratio',
                       choices=['inverse_ratio'],
                       help='Method to compute significance scores')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file path')
    parser.add_argument('--pretty', action='store_true',
                       help='Pretty-print JSON output')
    
    args = parser.parse_args()
    
    try:
        # Extract scores
        output_data = extract_layer_scores(
            model_name=args.model,
            metric_name=args.ww_metric,
            cache_dir=args.cache_dir,
            metric_cache_dir=args.ww_metric_cache,
            sparsity_ratio=args.sparsity_ratio,
            epsilon=args.epsilon,
            mapping_type=args.mapping_type,
            significance_method=args.significance_method
        )
        
        # Save to JSON
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        with open(args.output, 'w') as f:
            if args.pretty:
                json.dump(output_data, f, indent=2)
            else:
                json.dump(output_data, f)
        
        print(f"\n{'='*70}")
        print("EXTRACTION COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Output saved to: {args.output}")
        print(f"✓ Total layers: {output_data['num_prunable_layers']}")
        print(f"✓ Total parameters: {output_data['total_parameters']:,}")
        print(f"✓ Significance range: [{output_data['metadata']['significance_scores_min']:.6f}, "
              f"{output_data['metadata']['significance_scores_max']:.6f}]")
        print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

