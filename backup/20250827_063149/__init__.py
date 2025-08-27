# ==================================================
# config/__init__.py
# ==================================================
"""
Configuration package for Advanced Drone Swarm Simulator v5.1
Contains settings, constants, and configuration classes
"""

from .settings import (
    SafetyConfig,
    TakeoffConfig, 
    FlightPhase,
    SimulatorConfig,
    CollisionLogConfig,
    UILabels,
    AxisLabels,
    EARTH_RADIUS_KM,
    METERS_PER_DEGREE_LAT
)

__version__ = "5.1.0"
__author__ = "Drone Path Planning Laboratory"

__all__ = [
    'SafetyConfig',
    'TakeoffConfig',
    'FlightPhase', 
    'SimulatorConfig',
    'CollisionLogConfig',
    'UILabels',
    'AxisLabels',
    'EARTH_RADIUS_KM',
    'METERS_PER_DEGREE_LAT'
]

# ==================================================
# core/__init__.py  
# ==================================================
"""
Core modules for Advanced Drone Swarm Simulator v5.1
Contains coordinate system, collision avoidance, and flight management
"""

from core.coordinate_system import EarthCoordinateSystem
from core.collision_logger import CollisionLogger
from core.collision_avoidance import CollisionAvoidanceSystem
from core.flight_manager import TakeoffManager, QGCWaypointGenerator

__version__ = "5.1.0"

__all__ = [
    'EarthCoordinateSystem',
    'CollisionLogger',
    'CollisionAvoidanceSystem', 
    'TakeoffManager',
    'QGCWaypointGenerator'
]

# ==================================================
# gui/__init__.py
# ==================================================
"""
GUI modules for Advanced Drone Swarm Simulator v5.1
Contains main window, control panel, and 3D plot management
"""

from gui.main_window import MainWindow
from gui.control_panel import ControlPanel
from gui.plot_manager import Plot3DManager

__version__ = "5.1.0"

__all__ = [
    'MainWindow',
    'ControlPanel', 
    'Plot3DManager'
]

# ==================================================
# simulator/__init__.py
# ==================================================
"""
Simulator modules for Advanced Drone Swarm Simulator v5.1
Contains main simulator logic and file parsers
"""

from simulator.file_parser import (
    QGCFileParser,
    CSVFileParser, 
    FileParserFactory
)

__version__ = "5.1.0"

__all__ = [
    'QGCFileParser',
    'CSVFileParser',
    'FileParserFactory'
]

# Note: AdvancedDroneSimulator will be imported separately in main.py
# to avoid circular imports

# ==================================================
# utils/__init__.py
# ==================================================
"""
Utility modules for Advanced Drone Swarm Simulator v5.1
Contains logging configuration and helper functions
"""

from utils.logging_config import (
    setup_logging,
    get_module_logger,
    log_performance,
    log_method_calls,
    LogContext,
    CollisionEventFilter,
    setup_collision_logging,
    log_system_info,
    cleanup_old_logs
)

__version__ = "5.1.0"

__all__ = [
    'setup_logging',
    'get_module_logger',
    'log_performance',
    'log_method_calls', 
    'LogContext',
    'CollisionEventFilter',
    'setup_collision_logging',
    'log_system_info',
    'cleanup_old_logs'
]

# ==================================================
# tests/__init__.py
# ==================================================
"""
Test modules for Advanced Drone Swarm Simulator v5.1
Contains unit tests and integration tests
"""

__version__ = "5.1.0"
__test_framework__ = "pytest"

# Test configuration
TEST_DATA_DIR = "test_data"
TEST_OUTPUT_DIR = "test_output"

# Test categories
UNIT_TESTS = [
    'test_coordinate',
    'test_collision', 
    'test_collision_logger',
    'test_file_parser',
    'test_flight_manager'
]

INTEGRATION_TESTS = [
    'test_simulator_integration',
    'test_gui_integration',
    'test_file_io_integration'
]

__all__ = [
    'TEST_DATA_DIR',
    'TEST_OUTPUT_DIR',
    'UNIT_TESTS', 
    'INTEGRATION_TESTS'
]

# ==================================================
# Root package __init__.py (drone_simulator_v5.1/__init__.py)
# ==================================================
"""
Advanced Drone Swarm Simulator v5.1 - Professional Edition

A professional-grade drone swarm simulation system with:
- Real-time collision detection and avoidance
- 2x2 formation takeoff with configurable spacing (default 6m)
- Professional 3D visualization with Blender-style collision markers
- QGC waypoint file generation and modification
- Comprehensive collision logging and analysis
- English professional interface
- Mouse wheel zoom and interactive controls

Modules:
- config: Configuration settings and constants
- core: Core simulation logic (coordinate system, collision avoidance, flight management)
- gui: Graphical user interface components  
- simulator: Main simulator and file parsers
- utils: Logging and utility functions
- tests: Unit and integration tests

Example usage:
    from simulator.drone_simulator import AdvancedDroneSimulator
    
    simulator = AdvancedDroneSimulator()
    simulator.run()
"""

# Version information
__version__ = "5.1.0"
__edition__ = "Professional Edition"
__author__ = "Drone Path Planning Laboratory"
__email__ = "contact@dronelab.example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024 Advanced Drone Swarm Simulator Project"

# Package metadata
__title__ = "Advanced Drone Swarm Simulator"
__description__ = "Professional drone swarm simulation with collision avoidance"
__url__ = "https://github.com/drone-lab/advanced-drone-simulator"

# Feature flags
FEATURES = {
    'collision_logging': True,
    'formation_takeoff': True, 
    'mouse_wheel_zoom': True,
    'blender_style_markers': True,
    'english_interface': True,
    'qgc_export': True,
    'csv_import': True,
    'professional_3d': True
}

# System requirements
REQUIREMENTS = {
    'python_version': '>=3.8',
    'numpy': '>=1.21.0',
    'pandas': '>=1.3.0', 
    'matplotlib': '>=3.5.0',
    'psutil': '>=5.8.0',
    'tkinter': 'included'
}

# Import main classes for convenience
try:
    from config import SimulatorConfig
    from utils import setup_logging
    
    # Setup package-level logger
    _logger = setup_logging("INFO", log_to_file=False)
    _logger.info(f"Advanced Drone Swarm Simulator {__version__} - {__edition__}")
    
except ImportError as e:
    # Fallback if imports fail during installation
    import logging
    logging.warning(f"Package import warning: {e}")

__all__ = [
    '__version__',
    '__edition__', 
    '__author__',
    '__title__',
    '__description__',
    'FEATURES',
    'REQUIREMENTS'
]