#!/usr/bin/env python3
"""
分析项目中所有缺失的导入，并生成完整的config/settings.py
"""
import os
import re
import ast
from pathlib import Path
from collections import defaultdict

def find_all_python_files():
    """查找项目中所有Python文件"""
    python_files = []
    
    # 扫描主要目录
    directories = ['.', 'gui', 'core', 'simulator', 'utils']
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            for file_path in dir_path.rglob('*.py'):
                if file_path.name != 'settings.py':  # 排除我们正在修改的文件
                    python_files.append(file_path)
    
    return python_files

def extract_config_imports(file_path):
    """提取文件中从config.settings的所有导入"""
    imports = set()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找各种import模式
        patterns = [
            r'from\s+config\.settings\s+import\s+([^#\n]+)',
            r'from\s+config\s+import\s+([^#\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # 分割多个导入项
                items = [item.strip() for item in match.split(',')]
                imports.update(items)
        
        # 查找直接使用的属性
        # 如 config.settings.SomeClass 或 settings.SomeClass
        attr_patterns = [
            r'config\.settings\.(\w+)',
            r'settings\.(\w+)',
        ]
        
        for pattern in attr_patterns:
            matches = re.findall(pattern, content)
            imports.update(matches)
            
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
    
    return imports

def analyze_all_imports():
    """分析所有文件中的config导入"""
    print("分析项目中的所有config导入...")
    
    all_imports = defaultdict(list)
    python_files = find_all_python_files()
    
    print(f"找到 {len(python_files)} 个Python文件")
    
    for file_path in python_files:
        imports = extract_config_imports(file_path)
        if imports:
            all_imports[str(file_path)] = list(imports)
            print(f"  {file_path}: {', '.join(sorted(imports))}")
    
    # 汇总所有唯一的导入
    unique_imports = set()
    for imports in all_imports.values():
        unique_imports.update(imports)
    
    print(f"\n找到的所有唯一导入项:")
    for item in sorted(unique_imports):
        print(f"  - {item}")
    
    return unique_imports, all_imports

def generate_complete_settings(imports):
    """根据发现的导入生成完整的settings.py"""
    
    # 现有已知类
    known_classes = {
        'ComputeBackend', 'FlightPhase', 'DroneStatus', 'MissionType',
        'GPUConfig', 'TakeoffConfig', 'SafetyConfig', 'PerformanceConfig', 
        'VisualizationConfig', 'SimulationSettings', 'DroneConfig',
        'WeatherConfig', 'NetworkConfig', 'SimulatorConfig', 'UILabels'
    }
    
    # 可能需要的额外类
    possible_classes = {
        'ConnectionConfig': '@dataclass\nclass ConnectionConfig:\n    host: str = "localhost"\n    port: int = 14550\n    timeout: float = 5.0',
        'LoggingConfig': '@dataclass\nclass LoggingConfig:\n    level: str = "INFO"\n    file_output: bool = True\n    console_output: bool = True',
        'DisplayConfig': '@dataclass\nclass DisplayConfig:\n    theme: str = "dark"\n    font_size: int = 12\n    language: str = "zh_TW"',
        'AnimationConfig': '@dataclass\nclass AnimationConfig:\n    enabled: bool = True\n    speed: float = 1.0\n    smooth: bool = True',
        'PathConfig': '@dataclass\nclass PathConfig:\n    data_dir: str = "data"\n    log_dir: str = "logs"\n    config_dir: str = "config"',
        'ValidationConfig': '@dataclass\nclass ValidationConfig:\n    strict_mode: bool = False\n    auto_fix: bool = True',
        'ExportConfig': '@dataclass\nclass ExportConfig:\n    format: str = "json"\n    include_metadata: bool = True',
        'ImportConfig': '@dataclass\nclass ImportConfig:\n    auto_detect_format: bool = True\n    validate_data: bool = True',
        'SecurityConfig': '@dataclass\nclass SecurityConfig:\n    encrypt_logs: bool = False\n    require_auth: bool = False',
        'DebugConfig': '@dataclass\nclass DebugConfig:\n    enabled: bool = False\n    verbose: bool = False\n    profile: bool = False'
    }
    
    # 生成设置文件内容
    settings_content = '''"""
Enhanced configuration settings with GPU acceleration support
完整配置系统，包含所有必要的设置类
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import os

class ComputeBackend(Enum):
    """計算後端選項"""
    CPU = "cpu"
    GPU = "gpu"
    AUTO = "auto"

class FlightPhase(Enum):
    """飛行階段枚舉"""
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
    """無人機狀態枚舉"""
    IDLE = "idle"
    READY = "ready"
    FLYING = "flying"
    HOVERING = "hovering"
    RETURNING = "returning"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"

class MissionType(Enum):
    """任務類型枚舉"""
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
    device_id: int = 0
    memory_pool: bool = True
    enable_fallback: bool = True
    batch_size: int = 1000
    force_sync: bool = False
    accelerate_collision_detection: bool = True
    accelerate_coordinate_conversion: bool = True  
    accelerate_trajectory_calculation: bool = True
    accelerate_visualization: bool = True

@dataclass
class TakeoffConfig:
    """起飛配置"""
    formation_spacing: float = 6.0
    takeoff_altitude: float = 10.0
    hover_time: float = 2.0
    east_offset: float = 50.0

@dataclass
class SafetyConfig:
    """安全配置"""
    safety_distance: float = 5.0
    warning_distance: float = 8.0
    critical_distance: float = 3.0
    collision_check_interval: float = 0.1

@dataclass
class PerformanceConfig:
    """效能配置"""
    max_trajectory_points: int = 10000
    update_interval: float = 0.02
    parallel_workers: int = 4
    
@dataclass
class VisualizationConfig:
    """視覺化配置"""
    resolution: tuple = (1920, 1080)
    dpi: int = 100
    trail_length: int = 100
    collision_marker_size: float = 0.5
    enable_3d_acceleration: bool = True

@dataclass
class DroneConfig:
    """單個無人機配置"""
    max_speed: float = 15.0
    max_altitude: float = 120.0
    battery_capacity: float = 5000.0
    flight_time: float = 25.0
    payload_capacity: float = 2.0

@dataclass
class WeatherConfig:
    """天氣配置"""
    wind_speed: float = 0.0
    wind_direction: float = 0.0
    temperature: float = 20.0
    humidity: float = 50.0
    visibility: float = 10000.0

@dataclass
class NetworkConfig:
    """網路配置"""
    enable_logging: bool = True
    log_level: str = "INFO"
    auto_save: bool = True
    save_interval: int = 60

@dataclass
class SimulatorConfig:
    """模擬器配置"""
    max_drones: int = 50
    simulation_speed: float = 1.0
    auto_start: bool = False
    enable_physics: bool = True
    collision_detection: bool = True
    real_time_mode: bool = True

@dataclass
class UILabels:
    """用戶界面標籤配置"""
    window_title: str = "無人機群模擬器"
    menu_file: str = "檔案"
    menu_edit: str = "編輯"
    menu_view: str = "檢視"
    menu_help: str = "說明"
    btn_start: str = "開始"
    btn_stop: str = "停止"
    btn_pause: str = "暫停"
    btn_reset: str = "重置"
    btn_load: str = "載入"
    btn_save: str = "儲存"
    status_ready: str = "就緒"
    status_running: str = "運行中"
    status_paused: str = "已暫停"
    status_stopped: str = "已停止"
    tooltip_start: str = "開始模擬"
    tooltip_stop: str = "停止模擬"
    tooltip_pause: str = "暫停模擬"
    tooltip_reset: str = "重置模擬"

'''
    
    # 添加可能需要的额外类
    missing_classes = imports - known_classes
    for class_name in sorted(missing_classes):
        if class_name in possible_classes:
            settings_content += f"\n{possible_classes[class_name]}\n"
        else:
            # 生成通用类
            settings_content += f"""
@dataclass
class {class_name}:
    \"\"\"自动生成的{class_name}配置类\"\"\"
    enabled: bool = True
    value: Any = None
    options: Dict[str, Any] = field(default_factory=dict)
"""
    
    # 添加主要设置类和实例
    settings_content += '''
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
        if 'DRONE_SIM_GPU' in os.environ:
            gpu_setting = os.environ['DRONE_SIM_GPU'].lower()
            if gpu_setting in ['true', '1']:
                self.gpu.backend = ComputeBackend.GPU
            elif gpu_setting in ['false', '0']:
                self.gpu.backend = ComputeBackend.CPU

# 全域設定實例
settings = SimulationSettings()

# 向後相容的額外設定實例
drone_config = DroneConfig()
weather_config = WeatherConfig()  
network_config = NetworkConfig()
simulator_config = SimulatorConfig()
ui_labels = UILabels()

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
    print(f"計算後端設置為: {backend.value.upper()}")
    if backend == ComputeBackend.GPU:
        print(f"GPU設備ID: {device_id}")
'''
    
    return settings_content

def main():
    """主函数"""
    print("分析缺失导入工具")
    print("=" * 40)
    
    # 分析所有导入
    imports, file_imports = analyze_all_imports()
    
    if not imports:
        print("未发现任何config导入")
        return
    
    # 生成完整设置文件
    print(f"\n生成包含 {len(imports)} 个导入项的完整settings.py...")
    
    settings_content = generate_complete_settings(imports)
    
    # 备份现有文件
    settings_file = Path("config/settings.py")
    if settings_file.exists():
        backup_name = f"settings_backup_analysis.py"
        backup_path = settings_file.parent / backup_name
        import shutil
        shutil.copy2(settings_file, backup_path)
        print(f"已备份现有文件到: {backup_path}")
    
    # 写入新文件
    settings_file.write_text(settings_content, encoding='utf-8')
    print(f"已生成新的config/settings.py")
    
    # 测试导入
    print("\n测试新的settings.py...")
    try:
        # 清理模块缓存
        import sys
        modules_to_clear = [m for m in sys.modules.keys() if m.startswith('config')]
        for module in modules_to_clear:
            del sys.modules[module]
        
        # 测试导入
        from config.settings import settings
        print("基本导入测试通过")
        
        # 测试发现的导入项
        success_count = 0
        for import_item in sorted(imports):
            try:
                exec(f"from config.settings import {import_item}")
                success_count += 1
            except ImportError as e:
                print(f"  导入失败: {import_item} - {e}")
        
        print(f"成功导入 {success_count}/{len(imports)} 个项目")
        
        if success_count == len(imports):
            print("所有导入项测试通过！")
        else:
            print("部分导入项需要手动调整")
    
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    main()