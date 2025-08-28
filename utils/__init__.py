"""
Utils module for drone simulator
"""

__version__ = "5.2.0"

try:
    from .gpu_utils import *
    from .logging_config import setup_logging
    
except ImportError as e:
    # 基本回退
    def setup_logging():
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger()
    
    def get_array_module():
        import numpy as np
        return np
    
    def is_gpu_enabled():
        return False

__all__ = [
    'setup_logging', 
    'get_array_module', 
    'is_gpu_enabled',
    'compute_manager',
    'asarray',
    'to_cpu',
    'to_gpu'
]