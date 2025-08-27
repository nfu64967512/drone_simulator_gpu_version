#!/usr/bin/env python3
"""
無人機模擬器快速啟動腳本
提供簡單的命令列介面來選擇GPU/CPU模式
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """顯示啟動橫幅"""
    banner = """
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║            🚁 無人機群模擬器 GPU加速版                           ║
    ║                                                                ║
    ║                Professional Drone Swarm Simulator             ║
    ║                     with GPU Acceleration                     ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """檢查Python版本"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ 錯誤: 需要Python 3.8或更高版本")
        print(f"   當前版本: Python {version.major}.{version.minor}.{version.micro}")
        print("   請升級Python後重試")
        return False
    
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """檢查基本依賴"""
    required_packages = [
        'numpy',
        'pandas', 
        'matplotlib',
        'tkinter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 未安裝")
            missing_packages.append(package)
    
    return missing_packages

def check_gpu_support():
    """檢查GPU支援"""
    print("\n🔍 檢測GPU支援:")
    
    # 檢查CUDA
    cuda_available = False
    cuda_version = None
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 解析CUDA版本
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line.lower():
                    import re
                    version_match = re.search(r'release\s+(\d+\.\d+)', line)
                    if version_match:
                        cuda_version = version_match.group(1)
                        break
            
            print(f"✅ NVCC (CUDA編譯器) 可用 - 版本: {cuda_version}")
            cuda_available = True
        else:
            print("❌ NVCC 不可用")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ NVCC 不可用或未找到")
    
    # 檢查nvidia-smi
    nvidia_smi_available = False
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ nvidia-smi 可用")
            nvidia_smi_available = True
            
            # 顯示GPU資訊
            lines = result.stdout.split('\n')
            for line in lines:
                if 'GeForce' in line or 'Quadro' in line or 'Tesla' in line or 'RTX' in line:
                    gpu_info = line.strip()
                    print(f"🔧 檢測到GPU: {gpu_info}")
                    break
        else:
            print("❌ nvidia-smi 執行失敗")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi 不可用")
    
    # 檢查CuPy
    cupy_available = False
    try:
        import cupy
        print("✅ CuPy 已安裝")
        
        # 測試GPU功能
        test_array = cupy.array([1, 2, 3])
        _ = cupy.sum(test_array)
        cupy.cuda.Device().synchronize()
        print("✅ CuPy GPU功能正常")
        cupy_available = True
        
    except ImportError:
        print("❌ CuPy 未安裝 - GPU加速不可用")
        
        # 根據CUDA版本給出安裝建議
        if cuda_version:
            major_version = int(float(cuda_version))
            if major_version >= 12:
                print("💡 建議安裝: pip install cupy-cuda12x")
            elif major_version == 11:
                print("💡 建議安裝: pip install cupy-cuda11x") 
            else:
                print(f"💡 CUDA版本 {cuda_version} 可能需要特定的CuPy版本")
        
    except Exception as e:
        print(f"❌ CuPy GPU測試失敗: {e}")
    
    return cuda_available and nvidia_smi_available and cupy_available

def install_dependencies():
    """安裝依賴項目"""
    print("\n📦 安裝依賴項目...")
    
    # 檢查pip
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pip 不可用，請先安裝pip")
        return False
    
    # 安裝requirements.txt
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt 文件不存在")
        return False
    
    try:
        print("正在安裝依賴項目，這可能需要幾分鐘...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)
        ], check=True, capture_output=True, text=True)
        
        print("✅ 依賴項目安裝完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 依賴項目安裝失敗:")
        print(e.stderr)
        return False

def show_launch_menu():
    """顯示啟動選單"""
    print("\n🚀 啟動選項:")
    print("1. 🔄 自動選擇最佳後端 (推薦)")
    print("2. 🚀 強制使用GPU加速")
    print("3. 🖥️  強制使用CPU運算")
    print("4. 🧪 性能測試")
    print("5. 📦 安裝/更新依賴項目")
    print("6. 🔧 系統診斷")
    print("0. ❌ 退出")
    
    while True:
        try:
            choice = input("\n請選擇 (0-6): ").strip()
            if choice in ['0', '1', '2', '3', '4', '5', '6']:
                return choice
            else:
                print("❌ 無效選擇，請輸入 0-6")
        except KeyboardInterrupt:
            print("\n👋 使用者取消")
            return '0'

def launch_simulator(backend='auto'):
    """啟動模擬器"""
    main_script = Path(__file__).parent / "main.py"
    
    if not main_script.exists():
        print("❌ main.py 文件不存在")
        return False
    
    cmd = [sys.executable, str(main_script)]
    
    if backend != 'auto':
        cmd.extend(['--backend', backend])
        cmd.append('--no-gui-select')  # 跳過GUI選擇
    
    try:
        print(f"\n🚀 啟動模擬器 (後端: {backend.upper()})...")
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 模擬器啟動失敗: {e}")
        return False
    except KeyboardInterrupt:
        print("\n👋 使用者中斷")
        return True

def run_performance_test():
    """運行性能測試"""
    main_script = Path(__file__).parent / "main.py"
    
    if not main_script.exists():
        print("❌ main.py 文件不存在")
        return False
    
    try:
        print("\n🧪 開始性能測試...")
        result = subprocess.run([
            sys.executable, str(main_script), '--test'
        ], check=True, capture_output=True, text=True)
        
        # 顯示輸出
        if result.stdout:
            print(result.stdout)
        
        print("✅ 性能測試完成")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 性能測試失敗:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        
        # 提供故障排除建議
        if "ImportError" in e.stderr:
            print("\n💡 導入錯誤故障排除:")
            print("1. 確保所有新文件都已正確放置")
            print("2. 檢查 config/__init__.py 是否正確")
            print("3. 嘗試重新安裝依賴: pip install -r requirements.txt")
            
        return False

def system_diagnosis():
    """系統診斷"""
    print("\n🔧 系統診斷報告:")
    print("=" * 60)
    
    # 系統資訊
    print(f"操作系統: {platform.system()} {platform.release()}")
    print(f"處理器: {platform.processor()}")
    print(f"Python版本: {sys.version}")
    print(f"Python路徑: {sys.executable}")
    
    # 檢查依賴
    print("\n📦 依賴檢查:")
    missing = check_dependencies()
    
    # GPU檢查
    gpu_supported = check_gpu_support()
    
    # 記憶體資訊
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n💾 記憶體資訊:")
        print(f"總記憶體: {memory.total / (1024**3):.1f} GB")
        print(f"可用記憶體: {memory.available / (1024**3):.1f} GB")
        print(f"使用率: {memory.percent:.1f}%")
    except ImportError:
        print("\n💾 記憶體資訊: psutil未安裝，無法獲取")
    
    # 建議
    print(f"\n💡 建議:")
    if missing:
        print("- 請安裝缺少的依賴項目")
    if not gpu_supported:
        print("- 如需GPU加速，請安裝CUDA和CuPy")
    if gpu_supported:
        print("- ✅ 系統支援GPU加速，建議使用GPU模式")
    else:
        print("- 建議使用CPU模式以確保相容性")
    
    print("=" * 60)

def main():
    """主函數"""
    print_banner()
    
    # 檢查Python版本
    if not check_python_version():
        sys.exit(1)
    
    # 主循環
    while True:
        choice = show_launch_menu()
        
        if choice == '0':
            print("\n👋 再見！")
            break
        
        elif choice == '1':
            launch_simulator('auto')
        
        elif choice == '2':
            # 檢查GPU支援
            if not check_gpu_support():
                answer = input("\n⚠️ GPU支援檢測失敗，仍要繼續嗎？ (y/N): ")
                if answer.lower() != 'y':
                    continue
            launch_simulator('gpu')
        
        elif choice == '3':
            launch_simulator('cpu')
        
        elif choice == '4':
            run_performance_test()
        
        elif choice == '5':
            install_dependencies()
        
        elif choice == '6':
            system_diagnosis()
        
        # 詢問是否繼續
        if choice in ['1', '2', '3', '4']:
            answer = input("\n是否返回主選單？ (Y/n): ")
            if answer.lower() == 'n':
                break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 程式被使用者中斷")
    except Exception as e:
        print(f"\n❌ 程式執行錯誤: {e}")
        import traceback
        traceback.print_exc()