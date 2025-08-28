"""
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
