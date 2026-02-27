"""
Utility functions for model name handling.
"""


def get_model_short_name(model_name):
    """
    Generate a consistent short name for a model that matches existing naming convention.
    
    Examples:
        meta-llama/Llama-2-70b-hf -> llama2_70b
        meta-llama/Llama-2-13b-hf -> llama2_13b
        meta-llama/Llama-2-7b-hf -> llama2_7b
        facebook/opt-125m -> opt_125m
    """
    # Extract just the model part (after the slash)
    model_part = model_name.split("/")[-1].lower()
    
    # Remove common suffixes
    if model_part.endswith("-hf"):
        model_part = model_part[:-3]
    
    # Convert patterns like "llama-2-70b" to "llama2_70b"
    if "llama-2" in model_part:
        model_part = model_part.replace("llama-2-", "llama2_").replace("llama-2", "llama2")
    
    # Replace remaining hyphens with underscores
    model_part = model_part.replace("-", "_")
    
    return model_part

