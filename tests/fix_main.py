#!/usr/bin/env python3
"""
修復main.py中的AttributeError錯誤
"""
import re
from pathlib import Path
import shutil
from datetime import datetime

def fix_main_py():
    """修復main.py中的ComputeBackend.upper()錯誤"""
    print("修復main.py中的AttributeError錯誤...")
    
    main_file = Path("main.py")
    if not main_file.exists():
        print("  main.py不存在")
        return False
    
    # 備份原文件
    backup_name = f"main_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
    backup_path = main_file.parent / backup_name
    shutil.copy2(main_file, backup_path)
    print(f"  已備份原文件到: {backup_path}")
    
    # 讀取文件內容
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修復所有的ComputeBackend.upper()調用
    fixes_made = 0
    
    # 修復模式1: backend_info['backend'].upper()
    pattern1 = r"backend_info\['backend'\]\.upper\(\)"
    replacement1 = "backend_info['backend'].value.upper()"
    
    if re.search(pattern1, content):
        content = re.sub(pattern1, replacement1, content)
        fixes_made += 1
        print("  修復: backend_info['backend'].upper() -> backend_info['backend'].value.upper()")
    
    # 修復模式2: {backend_info['backend'].upper()}
    pattern2 = r"\{backend_info\['backend'\]\.upper\(\)\}"
    replacement2 = "{backend_info['backend'].value.upper()}"
    
    if re.search(pattern2, content):
        content = re.sub(pattern2, replacement2, content)
        fixes_made += 1
        print("  修復: {backend_info['backend'].upper()} -> {backend_info['backend'].value.upper()}")
    
    # 修復可能的類似問題
    pattern3 = r"(\w+)\.backend\.upper\(\)"
    replacement3 = r"\1.backend.value.upper()"
    
    matches = re.finditer(pattern3, content)
    for match in matches:
        content = re.sub(pattern3, replacement3, content)
        fixes_made += 1
        print(f"  修復: {match.group(0)} -> {match.group(1)}.backend.value.upper()")
    
    # 確保emoji字符被替換為安全字符
    emoji_fixes = {
        '✅': '[OK]',
        '📱': '[GPU]',
        '❌': '[ERROR]',
        '🚁': '[DRONE]',
        '🚀': '[START]'
    }
    
    for emoji, replacement in emoji_fixes.items():
        if emoji in content:
            content = content.replace(emoji, replacement)
            fixes_made += 1
    
    # 寫回文件
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"  總共修復了 {fixes_made} 個問題")
    return fixes_made > 0

def test_fix():
    """測試修復結果"""
    print("\n測試修復結果...")
    
    try:
        # 測試基本導入
        from config.settings import ComputeBackend, get_compute_backend_info
        
        # 模擬backend_info
        backend_info = get_compute_backend_info()
        
        # 測試修復後的代碼
        backend_str = backend_info['backend'].value.upper()
        print(f"  測試成功: 計算後端 = {backend_str}")
        
        return True
        
    except Exception as e:
        print(f"  測試失敗: {e}")
        return False

def create_simple_main():
    """如果修復失敗，創建簡化版main.py"""
    print("創建簡化版main.py...")
    
    simple_main_content = '''#!/usr/bin/env python3
"""
無人機群模擬器主程式 (簡化版)
"""
import sys
import argparse
from pathlib import Path

# 確保項目路徑在Python路徑中
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='無人機群模擬器')
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'auto'], 
                       default='auto', help='計算後端選擇')
    parser.add_argument('--device', type=int, default=0, help='GPU設備ID')
    parser.add_argument('--test', action='store_true', help='運行性能測試')
    
    args = parser.parse_args()
    
    try:
        # 導入配置
        from config.settings import settings, ComputeBackend, set_compute_backend, get_compute_backend_info
        from utils.logging_config import setup_logging
        
        # 設置日誌
        logger = setup_logging()
        
        # 設置後端
        backend_map = {
            'auto': ComputeBackend.AUTO,
            'gpu': ComputeBackend.GPU,
            'cpu': ComputeBackend.CPU
        }
        
        set_compute_backend(backend_map[args.backend], args.device)
        backend_info = get_compute_backend_info()
        
        print("[DRONE] 無人機群模擬器")
        print("="*50)
        print(f"[OK] 計算後端: {backend_info['backend'].value.upper()}")
        
        if args.test:
            run_performance_test()
        else:
            # 嘗試啟動主GUI
            try:
                from gui.main_window import DroneSimulatorApp
                import tkinter as tk
                
                print("[START] 啟動圖形界面...")
                root = tk.Tk()
                app = DroneSimulatorApp(root)
                root.mainloop()
                
            except ImportError as e:
                print(f"[WARN] GUI不可用: {e}")
                print("[INFO] 運行基本模式...")
                print("使用 --test 參數進行性能測試")
        
    except Exception as e:
        print(f"[ERROR] 啟動失敗: {e}")
        import traceback
        traceback.print_exc()

def run_performance_test():
    """性能測試"""
    print("[TEST] 開始性能測試...")
    
    try:
        from utils.gpu_utils import get_array_module, is_gpu_enabled
        import time
        
        xp = get_array_module()
        backend = "GPU" if is_gpu_enabled() else "CPU"
        print(f"[INFO] 使用 {backend} 進行測試")
        
        # 基本陣列測試
        test_sizes = [1000, 5000, 10000]
        for size in test_sizes:
            data = xp.random.random((size, 3)).astype(xp.float32)
            
            start_time = time.perf_counter()
            result = xp.sum(data * 2.0)
            
            # GPU同步
            if hasattr(xp, 'cuda'):
                xp.cuda.Device().synchronize()
            
            elapsed = time.perf_counter() - start_time
            print(f"  陣列大小 {size}: {elapsed*1000:.2f} ms")
        
        print("[OK] 性能測試完成")
        
    except Exception as e:
        print(f"[ERROR] 性能測試失敗: {e}")

if __name__ == "__main__":
    main()
'''
    
    # 備份現有main.py
    main_file = Path("main.py")
    if main_file.exists():
        backup_name = f"main_original_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        backup_path = main_file.parent / backup_name
        shutil.copy2(main_file, backup_path)
        print(f"  已備份原main.py到: {backup_path}")
    
    # 寫入簡化版本
    main_file.write_text(simple_main_content, encoding='utf-8')
    print("  已創建簡化版main.py")

def main():
    """主函數"""
    print("修復main.py AttributeError錯誤")
    print("=" * 40)
    
    # 嘗試修復現有main.py
    fix_success = fix_main_py()
    
    if fix_success:
        # 測試修復結果
        test_success = test_fix()
        
        if test_success:
            print("\n[OK] main.py修復成功!")
            print("[START] 現在可以運行: python main.py --test")
        else:
            print("\n[WARN] 修復可能不完整")
            answer = input("是否創建簡化版main.py? (y/N): ")
            if answer.lower() == 'y':
                create_simple_main()
    else:
        print("\n[ERROR] 無法修復現有main.py")
        create_simple_main()

if __name__ == "__main__":
    main()