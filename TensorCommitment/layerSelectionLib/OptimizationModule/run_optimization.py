#!/usr/bin/env python3
"""
Convenience script to run layer selection optimization.
This is an alias for optimize_layers.py for compatibility.
"""

import sys
from pathlib import Path

# Add OptimizationModule to path
module_path = Path(__file__).parent
if str(module_path.parent) not in sys.path:
    sys.path.insert(0, str(module_path.parent))

# Import and run the main function from optimize_layers
from OptimizationModule.optimize_layers import main

if __name__ == '__main__':
    main()


