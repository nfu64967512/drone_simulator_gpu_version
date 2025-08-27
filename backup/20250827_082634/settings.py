#!/usr/bin/env python3
"""
環境檢查腳本
診斷無人機模擬器的安裝和配置狀態
"""
import os
import sys
import subprocess
from pathlib import Path

def print_section(title):
    """印出段落標題"""
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def check_directory_structure():
    """檢查目錄結構"""
    print_section("目錄結構檢查")
    
    current_dir = Path.cwd()
    print(f"📍 當前目錄: {current_dir}")
    
    required_dirs = ['config', 'utils', 'core', 'simulator', 'gui']
    required_files = [
        'config/__init__.py',
        'config/settings.py', 
        'utils/__init__.py',
        'utils/gpu_utils.py',
        'main.py'
    ]
    
    # 檢查目錄
    print("\n📁 目錄檢查:")
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (不存在)")
    
    # 檢查文件
    print("\n📄 關鍵文件檢查:")
    for file_path in required_files:
        file_path = Path(file_path)
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"  ✅ {file_path} ({size} bytes)")
        else:
            print(f"  ❌ {file_path} (不存在)")

def check_python_environment():
    """檢查Python環境"""
    print_section("Python環境檢查")
    
    print(f"🐍 Python版本: {sys.version}")
    print(f"📦 Python執行檔: {sys.executable}")
    
    print(f"\n📂 Python路徑:")
    for i, path in enumerate(sys.path[:8]):  # 只顯示前8個路徑
        print(f"  {i+1}. {path}")

def check_cupy_installation():
    """檢查CuPy安裝狀況"""
    print_section("CuPy安裝檢查")
    
    # 檢查已安裝的cupy相關套件
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'list'
        ], capture_output=True, text=True, check=True)
        
        cupy_packages = []
        for line in result.stdout.split('\n'):
            if 'cupy' in line.lower():
                cupy_packages.append(line.strip())
        
        if cupy_packages:
            print("📦 已安裝的CuPy相關套件:")
            for pkg in cupy_packages:
                print(f"  • {pkg}")
            
            # 檢查是否有衝突
            cupy_variants = [pkg for pkg in cupy_packages if pkg.startswith('cupy-cuda')]
            if len(cupy_variants) > 1:
                print("⚠️ 警告: 檢測到多個CuPy版本，可能造成衝突!")
                print("建議執行: pip uninstall cupy-cuda11x cupy-cuda12x cupy -y")
        else:
            print("❌ 未找到CuPy安裝")
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 無法檢查套件列表: {e}")

def test_cupy_functionality():
    """測試CuPy功能"""
    print_section("CuPy功能測試")
    
    try:
        import cupy as cp
        print("✅ CuPy導入成功")
        
        # 測試基本GPU操作
        test_array = cp.array([1, 2, 3, 4, 5])
        result = cp.sum(test_array)
        cp.cuda.Device().synchronize()
        
        print(f"✅ GPU基本運算測試成功: sum([1,2,3,4,5]) = {result}")
        
        # 獲取GPU資訊
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"🖥️ 檢測到 {device_count} 個GPU設備")
        
        for i in range(device_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode()
            memory = props['totalGlobalMem'] / (1024**3)
            print(f"  GPU {i}: {name} ({memory:.1f} GB)")
            
    except ImportError as e:
        print(f"❌ CuPy導入失敗: {e}")
        print("💡 如需GPU功能，請安裝: pip install cupy-cuda12x")
    except Exception as e:
        print(f"❌ CuPy功能測試失敗: {e}")

def test_module_imports():
    """測試關鍵模組導入"""
    print_section("模組導入測試")
    
    # 基本導入測試
    basic_modules = [
        'numpy',
        'pandas', 
        'matplotlib',
        'tkinter'
    ]
    
    print("📦 基本模組:")
    for module_name in basic_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {module_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
    
    # 專案特定導入測試
    project_imports = [
        ('config', 'settings'),
        ('config.settings', 'ComputeBackend'),
        ('config.settings', 'FlightPhase'),
    ]
    
    print("\n🚁 專案模組:")
    for module_name, attr_name in project_imports:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            getattr(module, attr_name)
            print(f"  ✅ from {module_name} import {attr_name}")
        except ImportError as e:
            print(f"  ❌ {module_name}: {e}")
        except AttributeError as e:
            print(f"  ⚠️ {module_name}.{attr_name}: {e}")

def check_cuda_environment():
    """檢查CUDA環境"""
    print_section("CUDA環境檢查")
    
    # 檢查nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if 'release' in line.lower():
                    print(f"✅ NVCC: {line.strip()}")
                    break
        else:
            print("❌ NVCC不可用")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ NVCC未安裝或不在PATH中")
    
    # 檢查nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA-SMI可用")
            gpu_info = result.stdout.strip().split('\n')
            for i, info in enumerate(gpu_info):
                if info.strip():
                    print(f"  GPU {i}: {info.strip()}")
        else:
            print("❌ nvidia-smi執行失敗")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi不可用")

def provide_recommendations():
    """提供建議"""
    print_section("建議和解決方案")
    
    print("🚀 建議的下一步:")
    print("1. 確保所有必要目錄和檔案都已創建")
    print("2. 解決CuPy版本衝突（如有）")
    print("3. 將新的GPU加速檔案放入對應目錄")
    print("4. 測試運行: python main.py --test")
    
    print("\n🛠️ 常見問題解決:")
    print("• 模組導入失敗 → 檢查文件路徑和__init__.py")
    print("• CuPy衝突 → pip uninstall cupy-* && pip install cupy-cuda12x")
    print("• GPU不可用 → 檢查CUDA安裝和驅動程式")
    
    print("\n📞 獲得幫助:")
    print("• 查看詳細日誌: logs/ 目錄")
    print("• 提交問題: GitHub Issues")
    print("• 系統診斷: python check_setup.py")

def main():
    """主函數"""
    print("🔧 無人機模擬器環境檢查工具")
    print("="*60)
    
    # 依序執行所有檢查
    check_directory_structure()
    check_python_environment() 
    check_cupy_installation()
    test_cupy_functionality()
    test_module_imports()
    check_cuda_environment()
    provide_recommendations()
    
    print(f"\n🎯 檢查完成!")
    print("如需進一步協助，請將此輸出結果提供給技術支援。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 檢查被使用者中斷")
    except Exception as e:
        print(f"\n❌ 檢查過程發生錯誤: {e}")
        import traceback
        traceback.print_exc()