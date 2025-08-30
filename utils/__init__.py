# __init__.py - 根目錄
"""
無人機群飛模擬器 GPU 加速版本
專業級無人機群飛模擬系統，支援GPU/CPU計算後端靈活切換
"""

__version__ = "2.0.0"
__author__ = "無人機路徑規劃實驗室"
__description__ = "專業級無人機群飛模擬系統"

# 主要模組導入
try:
    from simulator.advanced_simulator_main import AdvancedDroneSimulator
    from config.settings import get_simulation_config
    from utils.gpu_utils import auto_detect_backend
    
    __all__ = [
        'AdvancedDroneSimulator',
        'get_simulation_config', 
        'auto_detect_backend'
    ]
    
except ImportError as e:
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []

# =============================================
# config/__init__.py
"""
配置管理模組
"""

from config.settings import (
    SimulationConfig,
    SafetyConfig, 
    FlightConfig,
    VisualizationConfig,
    get_simulation_config,
    get_config_manager
)

__all__ = [
    'SimulationConfig',
    'SafetyConfig',
    'FlightConfig', 
    'VisualizationConfig',
    'get_simulation_config',
    'get_config_manager'
]

# =============================================
# core/__init__.py
"""
核心演算法模組
"""

from core.trajectory_calculator import GPUTrajectoryCalculator, create_trajectory_calculator, FlightPhase

__all__ = [
    'EarthCoordinateSystem',
    'create_coordinate_system',
    'CollisionAvoidanceSystem', 
    'create_collision_system',
    'GPUTrajectoryCalculator',
    'create_trajectory_calculator',
    'FlightPhase'
]

# =============================================
# simulator/__init__.py
"""
模擬器主體模組
"""

__all__ = ['AdvancedDroneSimulator']

# =============================================
# gui/__init__.py
"""
圖形用戶界面模組
"""

from gui.gui_advanced_plotter import Advanced3DPlotter
from gui.control_panel import CompactControlPanel

__all__ = [
    'Advanced3DPlotter',
    'CompactControlPanel'
]

# =============================================
# utils/__init__.py
"""
工具模組
"""

from utils.gpu_utils import (
    get_array_module,
    ensure_gpu_compatibility,
    setup_gpu_environment,
    auto_detect_backend,
    GPUSystemChecker,
    run_performance_benchmark
)

from utils.qgc_handlers import (
    QGCWaypointParser,
    QGCWaypointGenerator, 
    CSVWaypointHandler,
    MissionFileExporter,
    parse_mission_file
)

from utils.logging_config import (
    setup_logging,
    get_logger,
    get_performance_logger,
    log_gpu_operation,
    log_collision_event
)

__all__ = [
    # GPU utilities
    'get_array_module',
    'ensure_gpu_compatibility', 
    'setup_gpu_environment',
    'auto_detect_backend',
    'GPUSystemChecker',
    'run_performance_benchmark',
    
    # QGC handlers
    'QGCWaypointParser',
    'QGCWaypointGenerator',
    'CSVWaypointHandler', 
    'MissionFileExporter',
    'parse_mission_file',
    
    # Logging
    'setup_logging',
    'get_logger',
    'get_performance_logger',
    'log_gpu_operation',
    'log_collision_event'
]