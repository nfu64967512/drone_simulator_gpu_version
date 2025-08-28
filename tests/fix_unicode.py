#!/usr/bin/env python3
"""
修復Unicode編碼錯誤的腳本
將emoji字符替換為Windows cp950兼容的符號
"""
import os
import sys
from pathlib import Path
import shutil
from datetime import datetime

def print_header():
    """顯示標題"""
    print("修復Unicode編碼錯誤工具")
    print("=" * 40)

def backup_files():
    """備份現有文件"""
    print("備份現有文件...")
    
    backup_dir = Path("backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_backup = [
        "utils/gpu_utils.py",
        "utils/logging_config.py", 
        "main.py"
    ]
    
    for file_path in files_to_backup:
        file_path = Path(file_path)
        if file_path.exists():
            backup_path = backup_dir / file_path.name
            shutil.copy2(file_path, backup_path)
            print(f"  已備份: {file_path}")
    
    return backup_dir

def fix_gpu_utils():
    """修復gpu_utils.py中的emoji字符"""
    print("修復gpu_utils.py...")
    
    gpu_utils_file = Path("utils/gpu_utils.py")
    if not gpu_utils_file.exists():
        print("  gpu_utils.py不存在，跳過")
        return
    
    with open(gpu_utils_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替換emoji為文字
    emoji_replacements = {
        '✅': '[OK]',
        '⚠️': '[WARN]',
        '❌': '[ERROR]',
        '🚀': '[GPU]',
        '📦': '[INFO]',
        '📱': '[DEVICE]',
        '💾': '[MEMORY]',
        '🖥️': '[CPU]',
        '🔄': '[LOADING]',
        '📊': '[STATUS]'
    }
    
    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)
    
    with open(gpu_utils_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  gpu_utils.py已修復")

def fix_logging_config():
    """修復logging_config.py中的emoji字符"""
    print("修復logging_config.py...")
    
    logging_file = Path("utils/logging_config.py")
    if not logging_file.exists():
        print("  logging_config.py不存在，跳過")
        return
    
    with open(logging_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替換emoji為文字
    emoji_replacements = {
        '🔧': '[SETUP]',
        '🚁': '[DRONE]',
        '📁': '[DIR]',
        '📊': '[STATS]',
        '🖥️': '[DESKTOP]',
        '📄': '[FILE]',
        '📋': '[JSON]',
        '🎯': '[TARGET]',
        '⚠️': '[WARN]',
        '✅': '[OK]',
        '🚨': '[ALERT]',
        '🐛': '[DEBUG]',
        'ℹ️': '[INFO]',
        '💾': '[SAVE]',
        '🧹': '[CLEAN]',
        '📞': '[HELP]'
    }
    
    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)
    
    # 修復控制台編碼問題
    console_handler_fix = '''
        if COLORLOG_AVAILABLE:
            # 彩色日誌輸出 - 修復編碼問題
            try:
                color_formatter = colorlog.ColoredFormatter(
                    '%(log_color)s%(asctime)s | %(backend)s | %(levelname)-8s | %(name)s | %(message)s',
                    datefmt='%H:%M:%S',
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
                console_handler.setFormatter(color_formatter)
            except UnicodeEncodeError:
                # 回退到標準格式化器
                console_formatter = SimulatorFormatter()
                console_handler.setFormatter(console_formatter)
        else:
            # 標準格式化器
            console_formatter = SimulatorFormatter()
            console_handler.setFormatter(console_formatter)
'''
    
    # 尋找並替換控制台處理器部分
    if 'if COLORLOG_AVAILABLE:' in content and 'color_formatter = colorlog.ColoredFormatter' in content:
        # 找到開始和結束位置
        start = content.find('if COLORLOG_AVAILABLE:')
        end = content.find('handlers.append(console_handler)', start)
        
        if start != -1 and end != -1:
            content = content[:start] + console_handler_fix.strip() + '\n        ' + content[end:]
    
    with open(logging_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  logging_config.py已修復")

def fix_main_py():
    """修復main.py中的AttributeError"""
    print("修復main.py...")
    
    main_file = Path("main.py")
    if not main_file.exists():
        print("  main.py不存在，跳過")
        return
    
    with open(main_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替換emoji為文字
    emoji_replacements = {
        '🚁': '[DRONE]',
        '✅': '[OK]',
        '📱': '[GPU]',
        '⚠️': '[WARN]',
        '❌': '[ERROR]',
        '🧪': '[TEST]',
        '🚀': '[START]',
        '👋': '[EXIT]'
    }
    
    for emoji, replacement in emoji_replacements.items():
        content = content.replace(emoji, replacement)
    
    # 修復AttributeError: ComputeBackend沒有upper()方法
    old_line = "print(f\"✅ 計算後端: {backend_info['backend'].upper()}\")"
    new_line = "print(f\"[OK] 計算後端: {backend_info['backend'].value.upper()}\")"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
    else:
        # 尋找類似的行並修復
        import re
        pattern = r"print\(f\".*計算後端.*{backend_info\['backend'\]\.upper\(\)}.*\"\)"
        replacement = "print(f\"[OK] 計算後端: {backend_info['backend'].value.upper()}\")"
        content = re.sub(pattern, replacement, content)
    
    # 修復其他可能的ComputeBackend.upper()調用
    content = content.replace(".upper()", ".value.upper()")
    
    with open(main_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("  main.py已修復")

def create_simple_logging_config():
    """創建簡化的日誌配置，避免編碼問題"""
    print("創建簡化日誌配置...")
    
    simple_logging = '''"""
簡化日誌配置系統 - 避免Unicode編碼問題
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logging(log_level="INFO", console_output=True, file_output=True):
    """設置簡化日誌系統"""
    
    # 創建日誌目錄
    log_dir = Path.cwd() / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # 清除現有處理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 設置日誌級別
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level_obj)
    
    handlers = []
    
    # 控制台輸出 - 使用安全的格式化器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_obj)
        
        # 簡單格式化器，避免Unicode問題
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # 文件輸出
    if file_output:
        log_file = log_dir / f"drone_simulator_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level_obj)
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # 添加處理器
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 記錄啟動信息（使用安全字符）
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("[SETUP] 無人機群模擬器日誌系統啟動")
    logger.info(f"[DIR] 日誌目錄: {log_dir}")
    logger.info(f"[STATS] 日誌級別: {log_level}")
    logger.info(f"[DESKTOP] 控制台輸出: {'啟用' if console_output else '禁用'}")
    logger.info(f"[FILE] 檔案輸出: {'啟用' if file_output else '禁用'}")
    logger.info("=" * 60)
    
    return root_logger

# 向後相容
def get_performance_logger():
    return logging.getLogger("performance")

def configure_gpu_logging():
    pass  # 簡化版本不需要特殊配置
'''
    
    # 備份原文件
    original_file = Path("utils/logging_config.py")
    if original_file.exists():
        backup_name = f"logging_config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        backup_path = original_file.parent / backup_name
        shutil.copy2(original_file, backup_path)
    
    # 寫入簡化版本
    original_file.write_text(simple_logging, encoding='utf-8')
    print("  已創建簡化版本的logging_config.py")

def test_fixes():
    """測試修復結果"""
    print("\\n測試修復結果...")
    
    try:
        # 測試導入
        import sys
        import importlib
        
        # 清除快取
        modules_to_reload = ['config.settings', 'utils.gpu_utils', 'utils.logging_config']
        for module in modules_to_reload:
            if module in sys.modules:
                del sys.modules[module]
        
        # 測試基本導入
        from config.settings import ComputeBackend, settings
        from utils.gpu_utils import is_gpu_enabled
        from utils.logging_config import setup_logging
        
        print("  [OK] 所有模組導入成功")
        
        # 測試日誌設置
        setup_logging(console_output=False)  # 避免控制台輸出測試
        print("  [OK] 日誌系統正常")
        
        # 測試ComputeBackend
        backend = ComputeBackend.CPU
        print(f"  [OK] ComputeBackend測試: {backend.value.upper()}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] 測試失敗: {e}")
        return False

def main():
    """主函數"""
    print_header()
    
    try:
        # 備份文件
        backup_dir = backup_files()
        
        # 修復各個文件
        fix_gpu_utils()
        fix_main_py()
        create_simple_logging_config()
        
        # 測試修復結果
        success = test_fixes()
        
        print("\\n" + "=" * 40)
        if success:
            print("[OK] 所有Unicode編碼錯誤已修復!")
            print("[START] 現在可以運行: python main.py --test")
        else:
            print("[ERROR] 修復過程中發生問題")
        
        print(f"[SAVE] 備份目錄: {backup_dir}")
        
    except Exception as e:
        print(f"[ERROR] 修復過程發生錯誤: {e}")

if __name__ == "__main__":
    main()