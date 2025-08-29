"""
無人機模擬器配置模組
提供GPU/CPU後端選擇和所有相關設定
"""

# 從新的 settings 模組導入所有配置類別和函數
from .settings import (
    BackendType,
    SafetyConfig,
    FlightConfig,
    VisualizationConfig,
    PerformanceConfig,
    SimulationConfig,
    BackendConfig,
    UIConfig,
    LoggingConfig,
    ExportConfig,
    get_simulation_config,
    update_gpu_backend,
    SYSTEM_LIMITS
)

# 新的 __init__ 檔案只負責匯入和提供便捷的存取方法

# 版本資訊
__version__ = "5.2.0"
__author__ = "Drone Simulator Team"

# 獲取全域設定實例
# 使用單一的 get_simulation_config() 函數作為所有設定的入口
settings = get_simulation_config()

# 向後相容的別名
DEFAULT_SETTINGS = settings
GPU_CONFIG = settings.backend  # 舊的 GPUConfig 現已整併到 backend 設定中
SAFETY_CONFIG = settings.safety
TAKEOFF_CONFIG = settings.flight  # 舊的 TakeoffConfig 現已整併到 flight 設定中

# 飛行相關枚舉
# 這些在舊檔案中，但新檔案中沒有定義，通常它們會被放在一個通用的 enum 檔案中。
# 在此假設它們是獨立存在的。
class FlightPhase:
    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    LANDING = "landing"

class DroneStatus:
    IDLE = "idle"
    FLYING = "flying"
    ERROR = "error"

class MissionType:
    WAYPOINT = "waypoint"
    SEARCH = "search"
    DELIVERY = "delivery"

# 常用工具函數 (更新為使用新的設定物件)
def get_compute_backend_info():
    """獲取計算後端資訊"""
    backend = settings.backend.backend_type.value
    use_gpu = settings.backend.use_gpu
    device_id = settings.backend.gpu_device_id
    return {
        "backend": backend,
        "use_gpu": use_gpu,
        "device_id": device_id
    }

def set_compute_backend(backend_type: BackendType, use_gpu: bool, device_id: int):
    """設置計算後端"""
    return update_gpu_backend(use_gpu, device_id)


def get_gpu_enabled():
    """檢查是否啟用GPU"""
    return settings.backend.backend_type == BackendType.GPU

def get_safety_distance():
    """獲取安全距離"""
    return settings.safety.safety_distance

def get_collision_check_interval():
    """獲取碰撞檢查間隔"""
    return settings.safety.collision_check_interval

def get_takeoff_formation_spacing():
    """獲取起飛編隊間距"""
    return settings.flight.formation_spacing

# 配置驗證
def validate_config():
    """驗證配置設定的合理性"""
    # 直接呼叫 settings 物件的驗證方法
    return settings.validate()

# 導出所有重要項目
__all__ = [
    # 枚舉
    'BackendType',
    'FlightPhase',
    'DroneStatus',
    'MissionType',

    # 配置類別
    'BackendConfig',
    'SafetyConfig',
    'FlightConfig',
    'VisualizationConfig',
    'PerformanceConfig',
    'SimulationConfig',
    'UIConfig',
    'LoggingConfig',
    'ExportConfig',

    # 設定實例
    'settings',

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