#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulator Package for Advanced Drone Swarm Simulator v5.1
Contains main simulator logic and file parsers

This package provides:
- AdvancedDroneSimulator: Main simulator class
- File parsers for QGC and CSV formats
- Factory pattern for parser selection
- Comprehensive error handling and logging
"""

import logging
from typing import Dict, List, Optional
from wsgiref.validate import validator

# Package metadata
__version__ = "5.1.0"
__author__ = "Drone Path Planning Laboratory"
__email__ = "contact@dronelab.example.com"
__description__ = "Advanced drone swarm simulation with collision avoidance"

# Import file parsers (safe imports that don't cause circular dependencies)
from simulator.file_parser import (
    QGCFileParser,
    CSVFileParser, 
    FileParserFactory
)

# Package-level logger
logger = logging.getLogger(__name__)

# Public API - what gets imported when someone does "from simulator import *"
__all__ = [
    # Main simulator class (imported separately to avoid circular imports)
    'AdvancedDroneSimulator',
    
    # File parsers
    'QGCFileParser',
    'CSVFileParser',
    'FileParserFactory',
    
    # Utility functions
    'get_supported_file_types',
    'validate_mission_file',
    'get_parser_for_file',
    
    # Package info
    '__version__',
    '__author__',
    '__description__'
]

# Supported file types for mission import
SUPPORTED_FILE_TYPES = {
    '.waypoints': 'QGroundControl Waypoint File',
    '.csv': 'Comma Separated Values File',
    '.txt': 'Text File (CSV format)'
}

# File format specifications
FILE_FORMAT_SPECS = {
    'qgc': {
        'description': 'QGroundControl Waypoint Format',
        'extensions': ['.waypoints'],
        'example': 'mission.waypoints',
        'required_header': 'QGC WPL 110',
        'column_count': 12,
        'documentation': 'https://dev.qgroundcontrol.com/master/en/file_formats/plan.html'
    },
    'csv': {
        'description': 'CSV Waypoint Format',
        'extensions': ['.csv', '.txt'],
        'example': 'waypoints.csv',
        'required_columns': ['lat', 'lon', 'alt'],
        'optional_columns': ['speed', 'heading', 'radius', 'loiter_time'],
        'supported_names': [
            'latitude, longitude, altitude',
            'lat, lon, alt', 
            'x, y, z',
            'Latitude, Longitude, Altitude'
        ]
    }
}

def get_supported_file_types() -> Dict[str, str]:
    """
    Get dictionary of supported file extensions and descriptions
    
    Returns:
        Dictionary mapping file extensions to descriptions
    """
    return SUPPORTED_FILE_TYPES.copy()

def validate_mission_file(file_path: str) -> Dict[str, any]:
    """
    Validate mission file and return format information
    
    Args:
        file_path: Path to mission file
        
    Returns:
        Dictionary with validation results and file info
    """
    import os
    from pathlib import Path
    
    file_path = Path(file_path)
    
    result = {
        'valid': False,
        'file_type': None,
        'parser_class': None,
        'error': None,
        'info': {},
        'warnings': []
    }
    
    try:
        # Check if file exists
        if not file_path.exists():
            result['error'] = f"File not found: {file_path}"
            return result
        
        # Check file extension
        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_FILE_TYPES:
            result['error'] = f"Unsupported file type: {extension}"
            result['warnings'].append(f"Supported types: {list(SUPPORTED_FILE_TYPES.keys())}")
            return result
        
        # Get parser
        try:
            parser = FileParserFactory.create_parser(str(file_path))
            result['parser_class'] = type(parser).__name__
            
            if extension == '.waypoints':
                result['file_type'] = 'qgc'
                # Basic QGC validation
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if not first_line.startswith('QGC WPL'):
                        result['warnings'].append("File may not be a valid QGC waypoint file")
                    else:
                        result['info']['qgc_version'] = first_line
                        
            elif extension in ['.csv', '.txt']:
                result['file_type'] = 'csv'
                # Basic CSV validation using CSV parser
                if hasattr(parser, 'detect_file_format'):
                    format_info = parser.detect_file_format(str(file_path))
                    result['info'].update(format_info)
            
            result['valid'] = True
            result['info']['file_size'] = file_path.stat().st_size
            result['info']['file_extension'] = extension
            
        except Exception as e:
            result['error'] = f"Parser validation failed: {e}"
            return result
            
    except Exception as e:
        result['error'] = f"Validation error: {e}"
    
    return result

def get_parser_for_file(file_path: str):
    """
    Get appropriate parser instance for file
    
    Args:
        file_path: Path to file
        
    Returns:
        Parser instance
        
    Raises:
        ValueError: If file type not supported
    """
    try:
        return FileParserFactory.create_parser(file_path)
    except Exception as e:
        logger.error(f"Failed to create parser for {file_path}: {e}")
        raise

def parse_mission_file(file_path: str) -> List[Dict]:
    """
    Parse mission file using appropriate parser
    
    Args:
        file_path: Path to mission file
        
    Returns:
        List of waypoint dictionaries
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    return FileParserFactory.parse_mission_file(file_path)

def get_file_format_info(format_type: str) -> Optional[Dict]:
    """
    Get detailed information about a file format
    
    Args:
        format_type: 'qgc' or 'csv'
        
    Returns:
        Format specification dictionary or None
    """
    return FILE_FORMAT_SPECS.get(format_type.lower())

def list_supported_formats() -> str:
    """
    Get formatted string listing all supported formats
    
    Returns:
        Formatted string with format information
    """
    lines = ["Supported Mission File Formats:\n"]
    
    for format_type, spec in FILE_FORMAT_SPECS.items():
        lines.append(f"📁 {spec['description']}")
        lines.append(f"   Extensions: {', '.join(spec['extensions'])}")
        lines.append(f"   Example: {spec['example']}")
        
        if format_type == 'qgc':
            lines.append(f"   Required Header: {spec['required_header']}")
            lines.append(f"   Columns: {spec['column_count']}")
        elif format_type == 'csv':
            lines.append(f"   Required Columns: {', '.join(spec['required_columns'])}")
            lines.append(f"   Supported Names: {spec['supported_names'][0]}")
        
        lines.append("")
    
    return "\n".join(lines)

# Safe import of main simulator class to avoid circular imports
def get_simulator_class():
    """
    Safely import and return the AdvancedDroneSimulator class
    
    This function allows lazy loading of the main simulator class
    to avoid circular import issues.
    
    Returns:
        AdvancedDroneSimulator class
    """
    try:
        from .drone_simulator import AdvancedDroneSimulator
        return AdvancedDroneSimulator
    except ImportError as e:
        logger.error(f"Failed to import AdvancedDroneSimulator: {e}")
        raise ImportError(f"Could not import main simulator class: {e}")

# Create a lazy-loaded property for the simulator class
class _SimulatorLoader:
    """Lazy loader for AdvancedDroneSimulator to avoid circular imports"""
    
    _simulator_class = None
    
    @property
    def AdvancedDroneSimulator(self):
        if self._simulator_class is None:
            self._simulator_class = get_simulator_class()
        return self._simulator_class

# Create module-level instance
_loader = _SimulatorLoader()

# Make AdvancedDroneSimulator available at module level
def __getattr__(name):
    """Dynamic attribute access for lazy loading"""
    if name == 'AdvancedDroneSimulator':
        return _loader.AdvancedDroneSimulator
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Package initialization
def _initialize_package():
    """Initialize simulator package"""
    logger.info(f"📦 Simulator package v{__version__} initialized")
    logger.debug(f"Supported file types: {list(SUPPORTED_FILE_TYPES.keys())}")

# Configuration validation
def validate_package_dependencies():
    """
    Validate that all required dependencies are available
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'missing_packages': [],
        'version_issues': [],
        'warnings': []
    }
    
    # Check required packages
    required_packages = [
        ('numpy', '1.21.0'),
        ('pandas', '1.3.0'),
        ('matplotlib', '3.5.0')
    ]
    
    for package_name, min_version in required_packages:
        try:
            import importlib
            package = importlib.import_module(package_name)
            
            # Check version if available
            if hasattr(package, '__version__'):
                from packaging import version
                if version.parse(package.__version__) < version.parse(min_version):
                    results['version_issues'].append(
                        f"{package_name} {package.__version__} < {min_version} (required)"
                    )
                    results['valid'] = False
                    
        except ImportError:
            results['missing_packages'].append(package_name)
            results['valid'] = False
        except Exception as e:
            results['warnings'].append(f"Could not check {package_name}: {e}")
    
    # Check optional packages
    optional_packages = ['psutil']
    for package_name in optional_packages:
        try:
            import importlib
            importlib.import_module(package_name)
        except ImportError:
            results['warnings'].append(f"Optional package {package_name} not available")
    
    return results

# Error handling for package imports
class SimulatorImportError(ImportError):
    """Custom exception for simulator import errors"""
    pass

class FileFormatError(ValueError):
    """Custom exception for file format errors"""
    pass

# Package-level constants
PACKAGE_NAME = "simulator"
PACKAGE_DESCRIPTION = __description__
PACKAGE_VERSION = __version__

# Feature flags
FEATURES = {
    'qgc_import': True,
    'csv_import': True,
    'collision_detection': True,
    'trajectory_analysis': True,
    'mission_modification': True,
    'collision_logging': True
}

# Performance settings
PERFORMANCE_SETTINGS = {
    'max_drones': 4,
    'trajectory_resolution': 0.5,  # seconds
    'collision_check_interval': 0.1,  # seconds
    'animation_fps': 30
}

# Initialize package
_initialize_package()

# Example usage documentation
if __name__ == "__main__":
    print(f"""
Advanced Drone Swarm Simulator v{__version__}
Simulator Package

Example usage:

    # Import simulator
    from simulator import AdvancedDroneSimulator
    
    # Create and run simulator
    simulator = AdvancedDroneSimulator()
    simulator.run()
    
    # Import file parsers
    from simulator import QGCFileParser, CSVFileParser, FileParserFactory
    
    # Parse mission file
    waypoints = FileParserFactory.parse_mission_file('mission.waypoints')
    
    # Validate file
    validation = validate_mission_file('mission.csv')
    print(f"Valid: {validator['valid']}")
    
    # Get supported formats
    formats = get_supported_file_types()
    print(f"Supported: {list(format.keys())}")

Supported File Formats:
{list_supported_formats()}
""")