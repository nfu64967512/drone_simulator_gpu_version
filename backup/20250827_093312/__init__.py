"""
無人機模擬器配置模組
提供GPU/CPU後端選擇和所有相關設定
"""

# 導入所有配置類別和枚舉
from .settings import (
    # 計算後端
    ComputeBackend,
    
    # 飛行相關枚舉
    FlightPhase,
    DroneStatus, 
    MissionType,
    
    # 配置類別
    GPUConfig,
    TakeoffConfig,
    SafetyConfig,
    PerformanceConfig,
    VisualizationConfig,
    SimulationSettings,
    
    # 向後相容的配置
    DroneConfig,
    WeatherConfig,
    NetworkConfig,
    
    # 設定實例
    settings,
    drone_config,
    weather_config,
    network_config,
    
    # 工具函數
    get_compute_backend_info,
    set_compute_backend
)

# 版本資訊
__version__ = "5.2.0"
__author__ = "Drone Simulator Team"

# 向後相容的別名
DEFAULT_SETTINGS = settings
GPU_CONFIG = settings.gpu
SAFETY_CONFIG = settings.safety
TAKEOFF_CONFIG = settings.takeoff

# 快速訪問常用設定
def get_gpu_enabled():
    """檢查是否啟用GPU"""
    return settings.gpu.backend == ComputeBackend.GPU

def get_safety_distance():
    """獲取安全距離"""
    return settings.safety.safety_distance

def get_collision_check_interval():
    """獲取碰撞檢查間隔"""
    return settings.safety.collision_check_interval

def get_takeoff_formation_spacing():
    """獲取起飛編隊間距"""
    return settings.takeoff.formation_spacing

# 配置驗證
def validate_config():
    """驗證配置設定的合理性"""
    issues = []
    
    # 檢查安全距離設定
    if settings.safety.safety_distance <= 0:
        issues.append("安全距離必須大於0")
    
    if settings.safety.warning_distance <= settings.safety.critical_distance:
        issues.append("警告距離應該大於緊急距離")
    
    # 檢查起飛設定
    if settings.takeoff.formation_spacing <= 0:
        issues.append("編隊間距必須大於0")
    
    if settings.takeoff.takeoff_altitude <= 0:
        issues.append("起飛高度必須大於0")
    
    # 檢查性能設定
    if settings.performance.update_interval <= 0:
        issues.append("更新間隔必須大於0")
    
    return issues

# 導出所有重要項目
__all__ = [
    # 枚舉
    'ComputeBackend',
    'FlightPhase', 
    'DroneStatus',
    'MissionType',
    
    # 配置類別
    'GPUConfig',
    'TakeoffConfig', 
    'SafetyConfig',
    'PerformanceConfig',
    'VisualizationConfig',
    'SimulationSettings',
    'DroneConfig',
    'WeatherConfig', 
    'NetworkConfig',
    
    # 設定實例
    'settings',
    'drone_config',
    'weather_config',
    'network_config',
    
    # 向後相容別名
    'DEFAULT_SETTINGS',
    'GPU_CONFIG', 
    'SAFETY_CONFIG',
    'TAKEOFF_CONFIG',
    
    # 工具函數
    'get_compute_backend_info',
    'set_compute_backend',
    'get_gpu_enabled',
    'get_safety_distance', 
    'get_collision_check_interval',
    'get_takeoff_formation_spacing',
    'validate_config',
    
    # 版本資訊
    '__version__',
    '__author__'
]