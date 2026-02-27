"""
Utility functions to load layer scores from JSON files.
"""

import json
from typing import Dict, List, Tuple


def load_scores_from_json(json_path: str) -> Tuple[List[float], List[float], Dict]:
    """
    Load layer values (benefits) and costs from a scores JSON file.
    
    Args:
        json_path: Path to JSON file containing layer scores
        
    Returns:
        Tuple of (values, costs, metadata) where:
        - values: List of L benefit values (significance_score)
        - costs: List of L cost values (num_parameters)
        - metadata: Dictionary with model information
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'layers' not in data:
        raise ValueError(f"JSON file {json_path} does not contain 'layers' key")
    
    layers = data['layers']
    if not layers:
        raise ValueError(f"JSON file {json_path} has no layers")
    
    values = []
    costs = []
    
    for layer in layers:
        if 'significance_score' not in layer:
            raise ValueError(f"Layer {layer.get('index', 'unknown')} missing 'significance_score'")
        if 'num_parameters' not in layer:
            raise ValueError(f"Layer {layer.get('index', 'unknown')} missing 'num_parameters'")
        
        values.append(float(layer['significance_score']))
        costs.append(float(layer['num_parameters']))
    
    metadata = {
        'model_name': data.get('model_name', 'unknown'),
        'num_layers': len(layers),
        'total_parameters': data.get('total_parameters', sum(costs)),
        'num_blocks': data.get('num_blocks', None),
        'layers_per_block': data.get('layers_per_block', None),
    }
    
    return values, costs, metadata

