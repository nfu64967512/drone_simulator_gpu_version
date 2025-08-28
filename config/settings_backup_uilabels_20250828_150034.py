"""
Enhanced configuration settings with GPU acceleration support
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import os

class ComputeBackend(Enum):
    """計算後端選項"""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"  # 自動檢測並選擇最佳後端

class FlightPhase(Enum):
    """飛行階段枚舉（向後相容）"""
    GROUND = "ground"
    TAXI = "taxi"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    DESCENT = "descent"
    APPROACH = "approach"
    LANDING = "landing"
    LANDED = "landed"

class DroneStatus(Enum):
    """無人機狀態枚舉（向後相容）"""
    IDLE = "idle"
    READY = "ready"
    FLYING = "flying"
    HOVERING = "hovering"
    RETURNING = "returning"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

class MissionType(Enum):
    """任務類型枚舉（向後相容）"""
    SURVEY = "survey"
    DELIVERY = "delivery"
    PATROL = "patrol"
    SEARCH_RESCUE = "search_rescue"
    FORMATION_FLIGHT = "formation_flight"
    CUSTOM = "custom"

@dataclass
class GPUConfig:
    """GPU加速配置"""
    backend: ComputeBackend = ComputeBackend.AUTO
    device_id: int = 0  # GPU設備ID (如果有多個GPU)
    memory_pool: bool = True  # 是否使用記憶體池優化
    enable_fallback: bool = True  # GPU不可用時自動回退到CPU
    batch_size: int = 1000  # GPU批次處理大小
    force_sync: bool = False  # 強制同步GPU操作（除錯用）
    
    # GPU加速模組開關
    accelerate_collision_detection: bool = True
    accelerate_coordinate_conversion: bool = True  
    accelerate_trajectory_calculation: bool = True
    accelerate_visualization: bool = True

@dataclass
class TakeoffConfig:
    """起飛配置"""
    formation_spacing: float = 6.0  # 無人機間距（公尺）
    takeoff_altitude: float = 10.0  # 初始爬升高度
    hover_time: float = 2.0  # 懸停時間
    east_offset: float = 50.0  # 起飛區域偏移

@dataclass
class SafetyConfig:
    """安全配置"""
    safety_distance: float = 5.0  # 最小安全距離
    warning_distance: float = 8.0  # 警告閾值
    critical_distance: float = 3.0  # 緊急碰撞閾值
    collision_check_interval: float = 0.1  # 檢查頻率（秒）

@dataclass
class PerformanceConfig:
    """效能配置"""
    max_trajectory_points: int = 10000  # 最大軌跡點數
    update_interval: float = 0.02  # 畫面更新間隔（秒）
    parallel_workers: int = 4  # CPU並行工作數量
    
@dataclass
class VisualizationConfig:
    """視覺化配置"""
    resolution: tuple = (1920, 1080)
    dpi: int = 100
    trail_length: int = 100  # 軌跡長度
    collision_marker_size: float = 0.5
    enable_3d_acceleration: bool = True
    
@dataclass
class SimulationSettings:
    """主要模擬設定"""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    takeoff: TakeoffConfig = field(default_factory=TakeoffConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    def __post_init__(self):
        """初始化後的配置檢查"""
        # 從環境變數讀取GPU設定
        if 'DRONE_SIM_GPU' in os.environ:
            gpu_setting = os.environ['DRONE_SIM_GPU'].lower()
            if gpu_setting == 'true' or gpu_setting == '1':
                self.gpu.backend = ComputeBackend.GPU
            elif gpu_setting == 'false' or gpu_setting == '0':
                self.gpu.backend = ComputeBackend.CPU

# 向後相容的設定別名
@dataclass 
class DroneConfig:
    """單個無人機配置（向後相容）"""
    max_speed: float = 15.0  # m/s
    max_altitude: float = 120.0  # meters
    battery_capacity: float = 5000.0  # mAh
    flight_time: float = 25.0  # minutes
    payload_capacity: float = 2.0  # kg

@dataclass
class WeatherConfig:
    """天氣配置（向後相容）"""
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # degrees
    temperature: float = 20.0  # celsius
    humidity: float = 50.0  # percentage
    visibility: float = 10000.0  # meters

@dataclass
class NetworkConfig:
    """網路配置（向後相容）"""
    enable_logging: bool = True
    log_level: str = "INFO"
    auto_save: bool = True
    save_interval: int = 60  # seconds

@dataclass
class SimulatorConfig:
    """模擬器配置（向後相容）"""
    max_drones: int = 50
    simulation_speed: float = 1.0
    auto_start: bool = False
    enable_physics: bool = True
    collision_detection: bool = True
    real_time_mode: bool = True

# 全域設定實例
settings = SimulationSettings()

# 向後相容的額外設定實例
drone_config = DroneConfig()
weather_config = WeatherConfig()  
network_config = NetworkConfig()
simulator_config = SimulatorConfig()

def get_compute_backend_info():
    """獲取計算後端資訊"""
    return {
        'backend': settings.gpu.backend,
        'device_id': settings.gpu.device_id,
        'fallback_enabled': settings.gpu.enable_fallback,
        'acceleration_modules': {
            'collision_detection': settings.gpu.accelerate_collision_detection,
            'coordinate_conversion': settings.gpu.accelerate_coordinate_conversion,
            'trajectory_calculation': settings.gpu.accelerate_trajectory_calculation,
            'visualization': settings.gpu.accelerate_visualization,
        }
    }

def set_compute_backend(backend: ComputeBackend, device_id: int = 0):
    """設置計算後端"""
    settings.gpu.backend = backend
    settings.gpu.device_id = device_id
    print(f"✅ 計算後端設置為: {backend.value.upper()}")
    if backend == ComputeBackend.GPU:
        print(f"📱 GPU設備ID: {device_id}")