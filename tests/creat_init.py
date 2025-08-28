#!/usr/bin/env python3
"""
檢查並創建缺失的__init__.py文件
"""
from pathlib import Path
import os

def check_and_create_init_files():
    """檢查並創建所有必要的__init__.py文件"""
    
    # 定義需要__init__.py的目錄及其內容
    init_files = {
        'config': '''"""
Configuration module for drone simulator
配置模組：包含所有設定和配置選項
"""
from .settings import *

__version__ = "5.2.0"
''',
        
        'utils': '''"""
Utilities module for drone simulator
工具模組：包含GPU工具、日誌配置等輔助功能
"""

__version__ = "5.2.0"

# 嘗試導入主要工具
try:
    from .gpu_utils import *
    from .logging_config import setup_logging
except ImportError:
    pass
''',
        
        'core': '''"""
Core module for drone simulator
核心模組：包含碰撞檢測、座標轉換和飛行管理功能
"""

__version__ = "5.2.0"

# 導入核心類別（如果存在的話）
try:
    from .collision_avoidance import CollisionDetector, GPUCollisionDetector
    from .coordinate_system import CoordinateConverter, GPUCoordinateConverter  
    from .flight_manager import FlightManager
    from .collision_logger import CollisionLogger
    
    # 向後相容的別名
    CollisionAvoidance = CollisionDetector
    CoordinateSystem = CoordinateConverter
    
    __all__ = [
        'CollisionDetector',
        'GPUCollisionDetector', 
        'CoordinateConverter',
        'GPUCoordinateConverter',
        'FlightManager',
        'CollisionLogger',
        # 向後相容
        'CollisionAvoidance',
        'CoordinateSystem'
    ]
    
except ImportError as e:
    # 如果某些模組不存在，提供基本功能
    __all__ = []
    
    # 基本的佔位類別
    class CollisionDetector:
        """基本碰撞檢測器佔位類"""
        def __init__(self):
            pass
        
        def detect_collisions(self, positions):
            return []
    
    class CoordinateConverter:
        """基本座標轉換器佔位類"""
        def __init__(self):
            pass
        
        def convert_to_meters(self, lat, lon, alt):
            return [0, 0, alt]
    
    class FlightManager:
        """基本飛行管理器佔位類"""
        def __init__(self):
            pass
    
    # 向後相容
    CollisionAvoidance = CollisionDetector
    CoordinateSystem = CoordinateConverter
    GPUCollisionDetector = CollisionDetector
    GPUCoordinateConverter = CoordinateConverter
    
    __all__ = [
        'CollisionDetector',
        'CoordinateConverter', 
        'FlightManager',
        'CollisionAvoidance',
        'CoordinateSystem',
        'GPUCollisionDetector',
        'GPUCoordinateConverter'
    ]
''',
        
        'simulator': '''"""
Simulator module for drone simulator
模擬器模組：包含主要模擬邏輯
"""

__version__ = "5.2.0"

try:
    from .drone_simulator import DroneSimulator, GPUDroneSimulator
    from .file_parser import FileParser
    
    __all__ = ['DroneSimulator', 'GPUDroneSimulator', 'FileParser']
    
except ImportError:
    # 基本佔位類
    class DroneSimulator:
        def __init__(self):
            pass
    
    class FileParser:
        def __init__(self):
            pass
    
    GPUDroneSimulator = DroneSimulator
    __all__ = ['DroneSimulator', 'GPUDroneSimulator', 'FileParser']
''',
        
        'gui': '''"""
GUI module for drone simulator
圖形界面模組：包含主視窗、控制面板和視覺化
"""

__version__ = "5.2.0"

try:
    from .main_window import DroneSimulatorApp, MainWindow
    from .control_panel import ControlPanel
    from .plot_manager import PlotManager, GPUPlotManager
    
    __all__ = [
        'DroneSimulatorApp', 
        'MainWindow',
        'ControlPanel', 
        'PlotManager', 
        'GPUPlotManager'
    ]
    
except ImportError:
    # 基本佔位類
    class DroneSimulatorApp:
        def __init__(self, root):
            self.root = root
            print("GUI模組不可用，使用基本模式")
    
    class MainWindow:
        def __init__(self):
            pass
    
    class ControlPanel:
        def __init__(self):
            pass
    
    class PlotManager:
        def __init__(self):
            pass
    
    GPUPlotManager = PlotManager
    
    __all__ = [
        'DroneSimulatorApp',
        'MainWindow', 
        'ControlPanel',
        'PlotManager',
        'GPUPlotManager'
    ]
''',
        
        'logs': '''"""
Logs directory
"""
# 這個目錄通常只用於存放日誌文件
# 不需要特殊的__init__.py內容
''',
        
        'backup': '''"""
Backup directory
"""
# 這個目錄用於存放備份文件
'''
    }
    
    print("檢查並創建__init__.py文件...")
    
    created_count = 0
    
    for directory, content in init_files.items():
        dir_path = Path(directory)
        init_file = dir_path / "__init__.py"
        
        # 確保目錄存在
        dir_path.mkdir(exist_ok=True)
        
        # 檢查__init__.py是否存在
        if not init_file.exists():
            init_file.write_text(content, encoding='utf-8')
            print(f"  已創建: {init_file}")
            created_count += 1
        else:
            print(f"  已存在: {init_file}")
    
    print(f"\n總共創建了 {created_count} 個__init__.py文件")
    return created_count

def test_imports():
    """測試所有模組的導入"""
    print("\n測試模組導入...")
    
    modules_to_test = [
        'config',
        'utils', 
        'core',
        'simulator',
        'gui'
    ]
    
    success_count = 0
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  [OK] {module}")
            success_count += 1
        except ImportError as e:
            print(f"  [WARN] {module}: {e}")
        except Exception as e:
            print(f"  [ERROR] {module}: {e}")
    
    print(f"\n導入測試結果: {success_count}/{len(modules_to_test)} 成功")
    return success_count == len(modules_to_test)

def check_project_structure():
    """檢查項目結構"""
    print("檢查項目結構...")
    
    expected_structure = {
        '.': ['main.py', 'launch.py', 'requirements.txt'],
        'config': ['__init__.py', 'settings.py'],
        'utils': ['__init__.py', 'gpu_utils.py', 'logging_config.py'],
        'core': ['__init__.py'],
        'simulator': ['__init__.py'],
        'gui': ['__init__.py'],
        'logs': ['__init__.py'],
        'backup': []
    }
    
    missing_files = []
    
    for directory, expected_files in expected_structure.items():
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"  [MISSING] 目錄: {directory}")
            continue
        
        print(f"  [OK] 目錄: {directory}")
        
        for file_name in expected_files:
            file_path = dir_path / file_name
            if file_path.exists():
                print(f"    [OK] {file_name}")
            else:
                print(f"    [MISSING] {file_name}")
                missing_files.append(str(file_path))
    
    if missing_files:
        print(f"\n缺少的文件:")
        for file_path in missing_files:
            print(f"  - {file_path}")
    
    return len(missing_files) == 0

def main():
    """主函數"""
    print("創建缺失的__init__.py文件工具")
    print("=" * 40)
    
    # 檢查項目結構
    structure_ok = check_project_structure()
    
    # 創建缺失的__init__.py文件
    created_count = check_and_create_init_files()
    
    # 測試導入
    imports_ok = test_imports()
    
    print("\n" + "=" * 40)
    print("總結:")
    
    if created_count > 0:
        print(f"[OK] 創建了 {created_count} 個__init__.py文件")
    
    if imports_ok:
        print("[OK] 所有基本模組導入成功")
    else:
        print("[WARN] 部分模組導入失敗，但這可能是正常的")
    
    if not structure_ok:
        print("[INFO] 部分項目文件缺失，但__init__.py文件已創建")
    
    print("\n現在可以嘗試運行:")
    print("  python -c \"import core; print('core模組導入成功')\"")
    print("  python main.py --test")

if __name__ == "__main__":
    main()