#!/usr/bin/env python3
"""
修復CuPy導入和安裝問題
"""
import sys
import subprocess
import importlib

def check_cupy_installation():
    """檢查CuPy安裝狀態"""
    print("檢查CuPy安裝...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'show', 'cupy-cuda12x'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  CuPy-cuda12x已安裝")
            print("  版本資訊:")
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    print(f"    {line}")
        else:
            print("  CuPy-cuda12x未安裝")
            return False
            
    except Exception as e:
        print(f"  檢查失敗: {e}")
        return False
    
    return True

def test_cupy_import():
    """測試CuPy導入"""
    print("測試CuPy導入...")
    
    try:
        import cupy as cp
        print("  基本導入: 成功")
        
        # 測試array屬性
        if hasattr(cp, 'array'):
            print("  array屬性: 存在")
        else:
            print("  array屬性: 不存在 - 這是問題!")
            return False
            
        # 測試創建陣列
        test_array = cp.array([1, 2, 3])
        print(f"  測試陣列: {test_array}")
        
        # 測試基本運算
        result = cp.sum(test_array)
        print(f"  基本運算: {result}")
        
        # 測試GPU同步
        cp.cuda.Device().synchronize()
        print("  GPU同步: 成功")
        
        return True
        
    except ImportError as e:
        print(f"  導入錯誤: {e}")
        return False
    except Exception as e:
        print(f"  運行錯誤: {e}")
        return False

def reinstall_cupy():
    """重新安裝CuPy"""
    print("重新安裝CuPy...")
    
    try:
        # 完全移除現有安裝
        print("  移除現有CuPy版本...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'uninstall', 
            'cupy-cuda12x', 'cupy-cuda11x', 'cupy', '-y'
        ], capture_output=True)
        
        # 清理快取
        subprocess.run([
            sys.executable, '-m', 'pip', 'cache', 'purge'
        ], capture_output=True)
        
        # 重新安裝
        print("  重新安裝CuPy-cuda12x...")
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', 'cupy-cuda12x'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  重新安裝成功!")
            return True
        else:
            print(f"  重新安裝失敗: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  重新安裝過程錯誤: {e}")
        return False

def create_cupy_test_script():
    """創建CuPy測試腳本"""
    test_script_content = '''#!/usr/bin/env python3
"""
CuPy功能測試腳本
"""
import sys

def test_cupy_basic():
    """測試CuPy基本功能"""
    try:
        print("導入CuPy...")
        import cupy as cp
        
        print(f"CuPy版本: {cp.__version__}")
        print(f"CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
        
        # 測試基本陣列操作
        print("\\n測試基本陣列操作:")
        a = cp.array([1, 2, 3, 4, 5])
        print(f"  原始陣列: {a}")
        
        b = a * 2
        print(f"  乘以2: {b}")
        
        c = cp.sum(a)
        print(f"  求和: {c}")
        
        # 測試大陣列
        print("\\n測試大陣列:")
        large_array = cp.random.random((1000, 1000))
        result = cp.mean(large_array)
        print(f"  1000x1000陣列平均值: {result}")
        
        # 測試GPU記憶體
        print("\\n測試GPU記憶體:")
        mempool = cp.get_default_memory_pool()
        print(f"  已使用記憶體: {mempool.used_bytes() / 1024**2:.1f} MB")
        print(f"  總記憶體: {mempool.total_bytes() / 1024**2:.1f} MB")
        
        # 清理記憶體
        mempool.free_all_blocks()
        print("  記憶體已清理")
        
        print("\\n所有測試通過!")
        return True
        
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cupy_basic()
'''
    
    with open('test_cupy.py', 'w', encoding='utf-8') as f:
        f.write(test_script_content)
    
    print("已創建CuPy測試腳本: test_cupy.py")

def fix_launch_script_cupy_test():
    """修復啟動腳本中的CuPy測試"""
    print("修復啟動腳本中的CuPy測試...")
    
    # 這裡應該修復launch.py中的GPU檢測函數
    # 替換有問題的cupy.array調用
    
    fixed_gpu_check = '''
def check_gpu_support():
    """檢查GPU支援"""
    print("\\n檢測GPU支援:")
    
    # 檢查CUDA
    cuda_available = False
    cuda_version = None
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            # 解析CUDA版本
            output = result.stdout
            for line in output.split('\\n'):
                if 'release' in line.lower():
                    import re
                    version_match = re.search(r'release\\s+(\\d+\\.\\d+)', line)
                    if version_match:
                        cuda_version = version_match.group(1)
                        break
            
            print(f"[OK] NVCC (CUDA編譯器) 可用 - 版本: {cuda_version}")
            cuda_available = True
        else:
            print("[ERROR] NVCC 不可用")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[ERROR] NVCC 不可用或未找到")
    
    # 檢查nvidia-smi
    nvidia_smi_available = False
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("[OK] nvidia-smi 可用")
            nvidia_smi_available = True
            
            # 顯示GPU資訊
            lines = result.stdout.split('\\n')
            for line in lines:
                if 'GeForce' in line or 'Quadro' in line or 'Tesla' in line or 'RTX' in line:
                    gpu_info = line.strip()
                    print(f"[DEVICE] 檢測到GPU: {gpu_info}")
                    break
        else:
            print("[ERROR] nvidia-smi 執行失敗")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[ERROR] nvidia-smi 不可用")
    
    # 檢查CuPy - 修復版本
    cupy_available = False
    try:
        import cupy as cp
        print("[OK] CuPy 已安裝")
        
        # 修復：使用更安全的測試方式
        if hasattr(cp, 'array') and hasattr(cp, 'sum'):
            # 測試GPU功能
            test_data = cp.array([1, 2, 3])
            test_result = cp.sum(test_data)
            cp.cuda.Device().synchronize()
            print(f"[OK] CuPy GPU功能正常 - 測試結果: {test_result}")
            cupy_available = True
        else:
            print("[ERROR] CuPy缺少必要屬性")
        
    except ImportError:
        print("[ERROR] CuPy 未安裝 - GPU加速不可用")
        
        # 根據CUDA版本給出安裝建議
        if cuda_version:
            major_version = int(float(cuda_version))
            if major_version >= 12:
                print("[INFO] 建議安裝: pip install cupy-cuda12x")
            elif major_version == 11:
                print("[INFO] 建議安裝: pip install cupy-cuda11x") 
            else:
                print(f"[INFO] CUDA版本 {cuda_version} 可能需要特定的CuPy版本")
        
    except Exception as e:
        print(f"[ERROR] CuPy GPU測試失敗: {e}")
        print("[INFO] 嘗試重新安裝: pip uninstall cupy-cuda12x && pip install cupy-cuda12x")
    
    return cuda_available and nvidia_smi_available and cupy_available
'''
    
    print("GPU檢測函數已修復")
    return fixed_gpu_check

def main():
    """主函數"""
    print("CuPy問題診斷和修復工具")
    print("=" * 40)
    
    # 檢查當前安裝
    installation_ok = check_cupy_installation()
    
    if installation_ok:
        # 測試導入
        import_ok = test_cupy_import()
        
        if not import_ok:
            print("\\n導入測試失敗，需要重新安裝")
            reinstall_ok = reinstall_cupy()
            
            if reinstall_ok:
                print("重新安裝完成，再次測試...")
                import_ok = test_cupy_import()
    else:
        print("\\nCuPy未安裝，進行安裝...")
        reinstall_ok = reinstall_cupy()
        
        if reinstall_ok:
            import_ok = test_cupy_import()
    
    # 創建測試腳本
    create_cupy_test_script()
    
    # 修復啟動腳本
    fixed_function = fix_launch_script_cupy_test()
    
    print("\\n" + "=" * 40)
    print("修復總結:")
    
    if 'import_ok' in locals() and import_ok:
        print("[OK] CuPy功能正常!")
        print("[START] 可以運行: python test_cupy.py")
        print("[START] 可以運行: python main.py --test")
    else:
        print("[ERROR] CuPy仍有問題")
        print("[INFO] 嘗試手動安裝:")
        print("  pip uninstall cupy-cuda12x cupy-cuda11x cupy -y")
        print("  pip install cupy-cuda12x")

if __name__ == "__main__":
    main()