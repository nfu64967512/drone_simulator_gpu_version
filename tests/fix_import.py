#!/usr/bin/env python3
"""
修正版導入問題修復腳本
解決新GPU版本與現有代碼的相容性問題，並自動創建必要文件
"""
import os
import sys
import subprocess
from pathlib import Path
import shutil
from datetime import datetime

def print_header():
    """顯示標題"""
    print("🔧 無人機模擬器 - 導入問題修復工具 (修正版)")
    print("=" * 60)
    print(f"📍 當前目錄: {os.getcwd()}")
    print(f"🐍 Python版本: {sys.version.split()[0]}")
    print("=" * 60)

def backup_existing_files():
    """備份現有配置文件"""
    print("📦 備份現有配置文件...")
    
    backup_dir = Path("backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_backup = [
        "config/__init__.py",
        "config/settings.py", 
        "utils/gpu_utils.py",
        "utils/logging_config.py",
        "main.py"
    ]
    
    backed_up = []
    for file_path in files_to_backup:
        file_path = Path(file_path)
        if file_path.exists():
            backup_path = backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            backed_up.append(str(file_path))
            print(f"  ✅ {file_path} -> {backup_path}")
    
    if backed_up:
        print(f"📁 備份目錄: {backup_dir}")
        return backup_dir
    else:
        print("  ℹ️ 沒有找到需要備份的文件")
        return None

def fix_cupy_conflict():
    """修復CuPy版本衝突"""
    print("\n🧹 修復CuPy版本衝突...")
    
    try:
        # 檢查已安裝的cupy套件
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'list'
        ], capture_output=True, text=True, check=True)
        
        cupy_packages = []
        for line in result.stdout.split('\n'):
            if line.strip().lower().startswith('cupy'):
                cupy_packages.append(line.strip().split()[0])
        
        if len(cupy_packages) > 1:
            print(f"  ⚠️ 檢測到多個CuPy版本: {', '.join(cupy_packages)}")
            
            # 移除所有CuPy版本
            for pkg in cupy_packages:
                print(f"  🗑️ 移除 {pkg}...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'uninstall', pkg, '-y'
                ], capture_output=True)
            
            # 清理pip快取
            subprocess.run([
                sys.executable, '-m', 'pip', 'cache', 'purge'
            ], capture_output=True)
            
            # 安裝正確版本
            print("  📦 安裝cupy-cuda12x...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ CuPy安裝成功")
            else:
                print(f"  ❌ CuPy安裝失敗: {result.stderr}")
        
        elif len(cupy_packages) == 1:
            print(f"  ✅ 已安裝單一CuPy版本: {cupy_packages[0]}")
        
        else:
            print("  ❌ 未安裝CuPy")
            # 安裝CuPy
            print("  📦 安裝cupy-cuda12x...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("  ✅ CuPy安裝成功")
            else:
                print(f"  ❌ CuPy安裝失敗，將使用CPU模式")
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ 無法檢查pip套件: {e}")

def create_directory_structure():
    """創建完整的目錄結構"""
    print("\n📁 創建目錄結構...")
    
    required_dirs = ["config", "utils", "core", "simulator", "gui", "logs", "backup"]
    created_dirs = []
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            created_dirs.append(dir_name)
            print(f"  ✅ 已創建 {dir_name}/")
        else:
            print(f"  📁 {dir_name}/ (已存在)")
        
        # 創建 __init__.py 文件
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_content = f'"""{dir_name.title()} module for drone simulator"""\n'
            init_file.write_text(init_content, encoding='utf-8')
    
    return created_dirs

def create_essential_config_files():
    """創建必要的配置文件"""
    print("\n📝 創建必要的配置文件...")
    
    # config/settings.py 的基本內容
    config_settings_content = '''"""
GPU加速無人機模擬器配置系統
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
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

@dataclass
class GPUConfig:
    """GPU配置"""
    backend: ComputeBackend = ComputeBackend.AUTO
    device_id: int = 0
    enable_fallback: bool = True
    batch_size: int = 1000
    accelerate_collision_detection: bool = True
    accelerate_coordinate_conversion: bool = True
    accelerate_trajectory_calculation: bool = True
    accelerate_visualization: bool = True

@dataclass
class SafetyConfig:
    """安全配置"""
    safety_distance: float = 5.0
    warning_distance: float = 8.0
    critical_distance: float = 3.0
    collision_check_interval: float = 0.1

@dataclass
class TakeoffConfig:
    """起飛配置"""
    formation_spacing: float = 6.0
    takeoff_altitude: float = 10.0
    hover_time: float = 2.0
    east_offset: float = 50.0

@dataclass
class SimulationSettings:
    """主要模擬設定"""
    gpu: GPUConfig = field(default_factory=GPUConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    takeoff: TakeoffConfig = field(default_factory=TakeoffConfig)

# 全域設定實例
settings = SimulationSettings()

def get_compute_backend_info():
    """獲取計算後端資訊"""
    return {
        'backend': settings.gpu.backend,
        'device_id': settings.gpu.device_id,
        'fallback_enabled': settings.gpu.enable_fallback,
    }

def set_compute_backend(backend: ComputeBackend, device_id: int = 0):
    """設置計算後端"""
    settings.gpu.backend = backend
    settings.gpu.device_id = device_id
    print(f"✅ 計算後端設置為: {backend.value.upper()}")
'''

    # config/__init__.py 的內容
    config_init_content = '''"""
無人機模擬器配置模組
"""
from .settings import (
    ComputeBackend,
    FlightPhase,
    DroneStatus,
    GPUConfig,
    SafetyConfig,
    TakeoffConfig,
    SimulationSettings,
    settings,
    get_compute_backend_info,
    set_compute_backend
)

__all__ = [
    'ComputeBackend',
    'FlightPhase', 
    'DroneStatus',
    'GPUConfig',
    'SafetyConfig',
    'TakeoffConfig',
    'SimulationSettings',
    'settings',
    'get_compute_backend_info',
    'set_compute_backend'
]
'''

    # utils/gpu_utils.py 的基本內容
    utils_gpu_content = '''"""
GPU/CPU 統一計算工具模組
"""
import numpy as np
from typing import Any

class ComputeManager:
    """計算後端管理器"""
    
    def __init__(self):
        self._backend = "CPU"
        self._xp = np
        self._gpu_available = self._detect_gpu()
    
    def _detect_gpu(self):
        """檢測GPU可用性"""
        try:
            import cupy as cp
            test_array = cp.array([1, 2, 3])
            _ = cp.sum(test_array)
            cp.cuda.Device().synchronize()
            self._xp = cp
            self._backend = "GPU"
            return True
        except:
            self._xp = np
            self._backend = "CPU"
            return False
    
    @property
    def xp(self):
        return self._xp
    
    @property
    def backend(self):
        return self._backend
    
    def asarray(self, array):
        return self._xp.asarray(array)
    
    def to_cpu(self, array):
        if hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)
    
    def is_gpu_enabled(self):
        return self._backend == "GPU"

# 全域計算管理器
compute_manager = ComputeManager()

def get_array_module():
    return compute_manager.xp

def asarray(array):
    return compute_manager.asarray(array)

def to_cpu(array):
    return compute_manager.to_cpu(array)

def is_gpu_enabled():
    return compute_manager.is_gpu_enabled()

print(f"🚀 GPU工具初始化完成 (後端: {compute_manager.backend})")
'''

    # 寫入文件
    files_to_create = [
        ("config/settings.py", config_settings_content),
        ("config/__init__.py", config_init_content),
        ("utils/gpu_utils.py", utils_gpu_content)
    ]
    
    for file_path, content in files_to_create:
        file_path = Path(file_path)
        if not file_path.exists():
            file_path.write_text(content, encoding='utf-8')
            print(f"  ✅ 已創建 {file_path}")
        else:
            print(f"  📄 {file_path} (已存在，跳過)")

def test_imports_safe():
    """安全的導入測試，避免模組路徑問題"""
    print("\n🧪 測試關鍵導入...")
    
    # 添加當前目錄到Python路徑
    current_dir = str(Path.cwd())
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    test_cases = [
        ("基本依賴", [
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("matplotlib", "matplotlib")
        ]),
        ("項目模組", [
            ("config.settings", "settings"),
            ("config.settings", "ComputeBackend"),
            ("config.settings", "FlightPhase"),
            ("utils.gpu_utils", "get_array_module")
        ])
    ]
    
    total_success = 0
    total_tests = 0
    
    for category, imports in test_cases:
        print(f"\n  📦 {category}:")
        for module_name, attr_name in imports:
            total_tests += 1
            try:
                if attr_name == module_name:
                    # 簡單模組導入
                    __import__(module_name)
                    print(f"    ✅ import {module_name}")
                else:
                    # 屬性導入
                    module = __import__(module_name, fromlist=[attr_name])
                    getattr(module, attr_name)
                    print(f"    ✅ from {module_name} import {attr_name}")
                total_success += 1
                
            except ImportError as e:
                print(f"    ❌ {module_name}: 模組不存在 - {e}")
            except AttributeError as e:
                print(f"    ⚠️ {module_name}.{attr_name}: 屬性不存在 - {e}")
            except Exception as e:
                print(f"    ❌ {module_name}: 未知錯誤 - {e}")
    
    print(f"\n📊 導入測試結果: {total_success}/{total_tests} 成功")
    return total_success == total_tests

def test_cupy_functionality():
    """測試CuPy功能"""
    print("\n🚀 測試GPU功能...")
    
    try:
        import cupy as cp
        print("  ✅ CuPy導入成功")
        
        # 測試基本操作
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        cp.cuda.Device().synchronize()
        
        print(f"  ✅ GPU基本運算成功: sum([1,2,3,4,5]) = {result}")
        
        # 獲取設備資訊
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"  🖥️ GPU設備數量: {device_count}")
        
        return True
        
    except ImportError:
        print("  ❌ CuPy未安裝，將使用CPU模式")
        return False
    except Exception as e:
        print(f"  ⚠️ GPU測試失敗，將使用CPU模式: {e}")
        return False

def create_basic_main_py():
    """創建基本的main.py文件（如果不存在）"""
    print("\n📝 檢查main.py...")
    
    main_py = Path("main.py")
    if not main_py.exists():
        main_content = '''#!/usr/bin/env python3
"""
無人機群模擬器主程式
"""
import sys
import argparse
from pathlib import Path

# 確保項目路徑在Python路徑中
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config.settings import settings, ComputeBackend
    from utils.gpu_utils import is_gpu_enabled
    print(f"🚁 無人機模擬器啟動")
    print(f"計算後端: {'GPU' if is_gpu_enabled() else 'CPU'}")
    
except ImportError as e:
    print(f"❌ 導入錯誤: {e}")
    sys.exit(1)

def run_performance_test():
    """性能測試"""
    print("🧪 性能測試...")
    
    from utils.gpu_utils import get_array_module, asarray
    import time
    
    xp = get_array_module()
    
    # 測試陣列運算
    test_data = [1000, 5000, 10000]
    for size in test_data:
        data = xp.random.random((size, 3)).astype(xp.float32)
        
        start_time = time.perf_counter()
        result = xp.sum(data * 2.0)
        if hasattr(xp, 'cuda'):
            xp.cuda.Device().synchronize()
        elapsed = time.perf_counter() - start_time
        
        print(f"  陣列大小 {size}: {elapsed*1000:.2f} ms")
    
    print("✅ 性能測試完成")

def main():
    parser = argparse.ArgumentParser(description='無人機群模擬器')
    parser.add_argument('--test', action='store_true', help='運行性能測試')
    args = parser.parse_args()
    
    if args.test:
        run_performance_test()
    else:
        print("🚁 無人機模擬器運行中...")
        print("使用 --test 參數運行性能測試")

if __name__ == "__main__":
    main()
'''
        main_py.write_text(main_content, encoding='utf-8')
        print("  ✅ 已創建基本的main.py")
    else:
        print("  📄 main.py已存在")

def provide_solutions():
    """提供解決方案和下一步建議"""
    print("\n💡 解決方案和建議:")
    print("1. 📦 安裝完整依賴: pip install -r requirements.txt")
    print("2. 🚀 測試GPU功能: python main.py --test")
    print("3. 📝 查看日誌: logs/ 目錄")
    
    print("\n🔧 如果仍有問題:")
    print("• 重新安裝Python套件: pip install --upgrade --force-reinstall cupy-cuda12x")
    print("• 檢查CUDA環境: nvcc --version && nvidia-smi")
    print("• 使用虛擬環境: python -m venv venv && venv\\Scripts\\activate")

def main():
    """主函數"""
    print_header()
    
    try:
        # 1. 備份現有文件
        backup_dir = backup_existing_files()
        
        # 2. 修復CuPy衝突
        fix_cupy_conflict()
        
        # 3. 創建目錄結構
        created_dirs = create_directory_structure()
        
        # 4. 創建必要的配置文件
        create_essential_config_files()
        
        # 5. 創建基本主程式（如果需要）
        create_basic_main_py()
        
        # 6. 測試導入
        imports_working = test_imports_safe()
        
        # 7. 測試GPU功能
        gpu_working = test_cupy_functionality()
        
        # 總結
        print("\n" + "=" * 60)
        print("🎯 修復總結:")
        
        if imports_working:
            print("✅ 所有導入測試通過!")
            if gpu_working:
                print("🚀 GPU功能正常，可以使用GPU加速")
            else:
                print("🖥️ GPU功能不可用，將使用CPU模式")
            
            print("\n🚀 現在可以運行:")
            print("   python main.py --test  # 性能測試")
            print("   python main.py         # 啟動模擬器")
            
        else:
            print("❌ 仍有導入問題需要解決")
            provide_solutions()
        
        if backup_dir:
            print(f"\n📁 備份位置: {backup_dir}")
        
    except Exception as e:
        print(f"\n❌ 修復過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        provide_solutions()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 使用者中斷修復過程")
    except Exception as e:
        print(f"\n❌ 修復腳本執行錯誤: {e}")
        import traceback
        traceback.print_exc()