"""
無人機模擬器設定檔案
包含所有系統配置參數，支援GPU/CPU後端切換
"""

import os
#import yaml
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class BackendType(Enum):
    """後端類型"""
    AUTO = "auto"
    GPU = "gpu" 
    CPU = "cpu"
    HYBRID = "hybrid"

class LogLevel(Enum):
    """日誌級別"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

@dataclass
class BackendConfig:
    """後端配置"""
    backend_type: BackendType = BackendType.AUTO
    use_gpu: bool = True
    gpu_device_id: int = 0
    force_cpu_modules: List[str] = None
    batch_size: int = 1000
    memory_pool: bool = True
    
    def __post_init__(self):
        if self.force_cpu_modules is None:
            self.force_cpu_modules = []

@dataclass
class SafetyConfig:
    """安全配置"""
    safety_distance: float = 5.0          # 安全距離 (m)
    warning_distance: float = 8.0         # 警告距離 (m) 
    critical_distance: float = 3.0        # 危險距離 (m)
    emergency_distance: float = 1.5       # 緊急距離 (m)
    collision_check_interval: float = 0.1 # 碰撞檢查間隔 (s)
    max_loiter_time: float = 30.0         # 最大等待時間 (s)
    
    def validate(self) -> bool:
        """驗證安全配置的合理性"""
        if not (self.emergency_distance < self.critical_distance < 
                self.safety_distance < self.warning_distance):
            logger.error("安全距離配置不合理")
            return False
        return True

@dataclass
class FlightConfig:
    """飛行配置"""
    cruise_speed: float = 8.0             # 巡航速度 (m/s)
    climb_rate: float = 2.0               # 爬升率 (m/s)
    descent_rate: float = 1.5             # 下降率 (m/s)
    takeoff_altitude: float = 10.0        # 起飛高度 (m)
    landing_speed: float = 1.0            # 降落速度 (m/s)
    hover_time: float = 2.0               # 懸停時間 (s)
    acceleration: float = 2.0             # 最大加速度 (m/s²)
    turn_radius: float = 5.0              # 轉彎半徑 (m)
    max_altitude: float = 120.0           # 最大飛行高度 (m)
    formation_spacing: float = 3.0        # 編隊間距 (m)

@dataclass
class VisualizationConfig:
    """視覺化配置"""
    # 3D繪圖設定
    figure_size: tuple = (18, 12)
    dpi: int = 100
    update_interval: int = 33             # ~30fps
    
    # 色彩配置
    background_color: str = "#1e1e1e"
    grid_color: str = "#404040"
    text_color: str = "#00d4aa"
    warning_color: str = "#ff5722"
    
    # 軌跡顯示
    trajectory_alpha: float = 0.4
    flown_path_alpha: float = 0.9
    waypoint_size: int = 25
    drone_model_size: int = 200
    
    # 碰撞警告
    collision_line_width: float = 4.0
    warning_marker_size: int = 300
    critical_marker_size: int = 500
    
    # 性能設定
    max_trajectory_points: int = 10000
    render_quality: str = "high"          # "high", "medium", "low"
    enable_shadows: bool = True
    enable_smooth_lines: bool = True

@dataclass
class PerformanceConfig:
    """性能配置"""
    # GPU設定
    gpu_memory_limit: float = 0.8         # GPU記憶體使用限制 (0-1)
    enable_memory_pool: bool = True
    auto_gc_threshold: float = 0.9        # 自動垃圾回收閾值
    
    # 並行處理
    max_threads: int = 4
    enable_multithreading: bool = True
    thread_pool_size: int = 8
    
    # 快取設定
    enable_trajectory_cache: bool = True
    cache_size_limit: int = 1000          # MB
    enable_position_cache: bool = True
    
    # 優化設定
    enable_batch_processing: bool = True
    batch_size: int = 1000
    enable_vectorization: bool = True
    optimize_memory_usage: bool = True

@dataclass
class UIConfig:
    """用戶界面配置"""
    # 主視窗
    window_title: str = "進階無人機群飛模擬器 - GPU版本"
    window_geometry: str = "1920x1080"
    maximize_on_start: bool = True
    
    # 主題設定
    theme: str = "dark"                   # "dark", "light"
    font_family: str = "Arial"
    font_size: int = 10
    
    # 控制面板
    control_panel_width: int = 280
    status_text_height: int = 12
    warning_text_height: int = 6
    
    # 快捷鍵
    shortcuts: Dict[str, str] = None
    
    def __post_init__(self):
        if self.shortcuts is None:
            self.shortcuts = {
                "play_pause": "space",
                "reset": "r",
                "stop": "s", 
                "top_view": "1",
                "side_view": "2",
                "3d_view": "3",
                "export": "ctrl+s",
                "load": "ctrl+o",
                "quit": "escape"
            }

@dataclass
class LoggingConfig:
    """日誌配置"""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # 檔案輸出
    log_to_file: bool = True
    log_file_path: str = "logs/simulator.log"
    max_file_size: int = 10               # MB
    backup_count: int = 5
    
    # 控制台輸出  
    log_to_console: bool = True
    console_level: LogLevel = LogLevel.INFO
    
    # GPU相關日誌
    log_gpu_operations: bool = False
    log_memory_usage: bool = True
    log_performance_metrics: bool = True

@dataclass 
class ExportConfig:
    """導出配置"""
    # 預設路徑
    default_export_dir: str = "exports"
    
    # 檔案格式
    mission_file_format: str = "waypoints"  # "waypoints", "json"
    add_timestamp: bool = True
    create_summary: bool = True
    
    # QGC設定
    qgc_version: str = "110"
    default_speed: float = 8.0
    default_altitude: float = 15.0
    
    # 壓縮設定
    compress_exports: bool = False
    compression_format: str = "zip"       # "zip", "tar"

@dataclass
class SimulationConfig:
    """完整的模擬配置"""
    # 子配置
    backend: BackendConfig = None
    safety: SafetyConfig = None
    flight: FlightConfig = None
    visualization: VisualizationConfig = None
    performance: PerformanceConfig = None  
    ui: UIConfig = None
    logging: LoggingConfig = None
    export: ExportConfig = None
    
    # 基本設定
    version: str = "2.0.0"
    debug_mode: bool = False
    
    def __post_init__(self):
        """初始化預設值"""
        if self.backend is None:
            self.backend = BackendConfig()
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.flight is None:
            self.flight = FlightConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.performance is None:
            self.performance = PerformanceConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.export is None:
            self.export = ExportConfig()
    
    def validate(self) -> bool:
        """驗證所有配置"""
        if not self.safety.validate():
            return False
        
        # 檢查飛行參數
        if self.flight.cruise_speed <= 0:
            logger.error("巡航速度必須大於0")
            return False
            
        if self.flight.takeoff_altitude <= 0:
            logger.error("起飛高度必須大於0") 
            return False
        
        # 檢查性能配置
        if not (0 < self.performance.gpu_memory_limit <= 1):
            logger.error("GPU記憶體限制必須在(0,1]範圍內")
            return False
        
        return True


class ConfigManager:
    """
    配置管理器
    負責載入、保存和管理所有配置
    """
    
    def __init__(self, config_dir: str = "config"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置檔案目錄
        """
        self.config_dir = config_dir
        self.config_file = os.path.join(config_dir, "settings.yaml")
        self.user_config_file = os.path.join(config_dir, "user_settings.yaml")
        
        # 確保配置目錄存在
        os.makedirs(config_dir, exist_ok=True)
        
        self._config: Optional[SimulationConfig] = None
    
    def load_config(self, config_file: Optional[str] = None) -> SimulationConfig:
        """
        載入配置檔案
        
        Args:
            config_file: 配置檔案路徑（可選）
            
        Returns:
            模擬配置對象
        """
        if config_file is None:
            config_file = self.config_file
        
        # 載入預設配置
        config = SimulationConfig()
        
        # 如果配置檔案存在，載入並合併
        # if os.path.exists(config_file):
        #     try:
        #         with open(config_file, 'r', encoding='utf-8') as f:
        #             config_data = yaml.safe_load(f)
                
        #         if config_data:
        #             config = self._merge_config(config, config_data)
                    
        #         logger.info(f"成功載入配置檔案: {config_file}")
                
        #     except Exception as e:
        #         logger.error(f"載入配置檔案失敗: {e}")
        #         logger.info("使用預設配置")
        
        # 載入用戶自定義配置
        # if os.path.exists(self.user_config_file):
        #     try:
        #         with open(self.user_config_file, 'r', encoding='utf-8') as f:
        #             user_config_data = yaml.safe_load(f)
                
        #         if user_config_data:
        #             config = self._merge_config(config, user_config_data)
                    
        #         logger.info(f"成功載入用戶配置: {self.user_config_file}")
                
        #     except Exception as e:
        #         logger.warning(f"載入用戶配置失敗: {e}")
        
        # 驗證配置
        if not config.validate():
            logger.error("配置驗證失敗，使用預設配置")
            config = SimulationConfig()
        
        self._config = config
        return config
    
    def save_config(self, config: SimulationConfig, 
                   config_file: Optional[str] = None) -> bool:
        """
        保存配置檔案
        
        Args:
            config: 要保存的配置
            config_file: 配置檔案路徑（可選）
            
        Returns:
            是否保存成功
        """
        # if config_file is None:
        #     config_file = self.user_config_file
        
        # try:
        #     config_dict = asdict(config)
            
        #     with open(config_file, 'w', encoding='utf-8') as f:
        #         yaml.dump(config_dict, f, default_flow_style=False, 
        #                  allow_unicode=True, indent=2)
            
        #     logger.info(f"配置已保存至: {config_file}")
        #     return True
            
        # except Exception as e:
        #     logger.error(f"保存配置失敗: {e}")
        #     return False
    
    def _merge_config(self, base_config: SimulationConfig, 
                     config_data: Dict[str, Any]) -> SimulationConfig:
        """
        合併配置數據
        
        Args:
            base_config: 基礎配置
            config_data: 要合併的配置數據
            
        Returns:
            合併後的配置
        """
        try:
            # 遞歸更新配置
            for key, value in config_data.items():
                if hasattr(base_config, key):
                    attr = getattr(base_config, key)
                    
                    if isinstance(value, dict) and hasattr(attr, '__dict__'):
                        # 遞歸更新嵌套配置
                        for sub_key, sub_value in value.items():
                            if hasattr(attr, sub_key):
                                setattr(attr, sub_key, sub_value)
                    else:
                        # 直接設定屬性
                        setattr(base_config, key, value)
                        
        except Exception as e:
            logger.warning(f"合併配置時出錯: {e}")
        
        return base_config
    
    def get_config(self) -> SimulationConfig:
        """獲取當前配置"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_backend_config(self, backend_type: BackendType, 
                            use_gpu: bool = None, 
                            gpu_device: int = None) -> bool:
        """
        更新後端配置
        
        Args:
            backend_type: 後端類型
            use_gpu: 是否使用GPU
            gpu_device: GPU設備ID
            
        Returns:
            是否更新成功
        """
        try:
            config = self.get_config()
            config.backend.backend_type = backend_type
            
            if use_gpu is not None:
                config.backend.use_gpu = use_gpu
                
            if gpu_device is not None:
                config.backend.gpu_device_id = gpu_device
            
            self.save_config(config)
            logger.info(f"後端配置已更新: {backend_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"更新後端配置失敗: {e}")
            return False
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """獲取GPU相關配置"""
        config = self.get_config()
        
        return {
            'use_gpu': config.backend.use_gpu,
            'device_id': config.backend.gpu_device_id,
            'batch_size': config.backend.batch_size,
            'memory_pool': config.backend.memory_pool,
            'memory_limit': config.performance.gpu_memory_limit,
            'enable_cache': config.performance.enable_trajectory_cache
        }
    
    def create_default_config_file(self) -> bool:
        """創建預設配置檔案"""
        try:
            default_config = SimulationConfig()
            return self.save_config(default_config, self.config_file)
        except Exception as e:
            logger.error(f"創建預設配置檔案失敗: {e}")
            return False


# 全域配置管理器實例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """獲取全域配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_simulation_config() -> SimulationConfig:
    """獲取模擬配置"""
    return get_config_manager().get_config()

def update_gpu_backend(use_gpu: bool, device_id: int = 0) -> bool:
    """更新GPU後端設定"""
    manager = get_config_manager()
    backend_type = BackendType.GPU if use_gpu else BackendType.CPU
    return manager.update_backend_config(backend_type, use_gpu, device_id)


# 預設配置常量
DEFAULT_COLORS = [
    '#FF4444', '#44FF44', '#4444FF', '#FFFF44',
    '#FF44FF', '#44FFFF', '#FFAA44', '#AA44FF'
]

# 系統限制
SYSTEM_LIMITS = {
    'max_drones': 16,
    'max_waypoints_per_drone': 1000,
    'max_simulation_time': 3600,  # 1小時
    'min_safety_distance': 1.0,
    'max_safety_distance': 50.0,
    'min_speed': 1.0,
    'max_speed': 30.0
}