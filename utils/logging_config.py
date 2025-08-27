"""
專業日誌配置系統
支持多種輸出格式和GPU/CPU性能監控
"""
import logging
import logging.handlers
import sys
import os
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, Any

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False

class GPUPerformanceFilter(logging.Filter):
    """GPU性能日誌過濾器"""
    
    def filter(self, record):
        # 添加GPU/CPU後端資訊到日誌記錄
        try:
            from utils.gpu_utils import is_gpu_enabled
            record.backend = "GPU" if is_gpu_enabled() else "CPU"
        except:
            record.backend = "CPU"
        
        return True

class JSONFormatter(logging.Formatter):
    """JSON格式日誌格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'backend': getattr(record, 'backend', 'CPU')
        }
        
        # 添加異常資訊
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加額外字段
        if hasattr(record, 'drone_count'):
            log_entry['drone_count'] = record.drone_count
        
        if hasattr(record, 'fps'):
            log_entry['fps'] = record.fps
        
        if hasattr(record, 'gpu_memory'):
            log_entry['gpu_memory_mb'] = record.gpu_memory
        
        return json.dumps(log_entry, ensure_ascii=False)

class SimulatorFormatter(logging.Formatter):
    """無人機模擬器專用格式化器"""
    
    def format(self, record):
        # 添加後端資訊
        backend = getattr(record, 'backend', 'CPU')
        
        # 創建基本格式
        base_format = f"%(asctime)s | {backend} | %(levelname)-8s | %(name)s | %(message)s"
        
        # 根據日誌級別調整格式
        if record.levelno >= logging.ERROR:
            format_str = f"🚨 {base_format}"
        elif record.levelno >= logging.WARNING:
            format_str = f"⚠️  {base_format}"
        elif record.levelno >= logging.INFO:
            format_str = f"ℹ️  {base_format}"
        else:
            format_str = f"🐛 {base_format}"
        
        formatter = logging.Formatter(format_str, datefmt='%H:%M:%S')
        return formatter.format(record)

def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    json_output: bool = False,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    設置專業日誌系統
    
    Args:
        log_level: 日誌級別
        log_dir: 日誌目錄
        console_output: 是否輸出到控制台
        file_output: 是否輸出到檔案
        json_output: 是否輸出JSON格式日誌
        max_file_size: 日誌檔案最大大小
        backup_count: 保留的日誌檔案數量
    
    Returns:
        配置好的根日誌器
    """
    
    # 創建日誌目錄
    if log_dir is None:
        log_dir = Path.cwd() / "logs"
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # 清除現有處理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 設置日誌級別
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level_obj)
    
    # 添加性能過濾器
    performance_filter = GPUPerformanceFilter()
    
    handlers = []
    
    # 控制台輸出
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level_obj)
        console_handler.addFilter(performance_filter)
        
        if COLORLOG_AVAILABLE:
            # 彩色日誌輸出
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
        else:
            # 標準格式化器
            console_formatter = SimulatorFormatter()
            console_handler.setFormatter(console_formatter)
        
        handlers.append(console_handler)
    
    # 檔案輸出
    if file_output:
        # 主日誌檔案
        main_log_file = log_dir / f"drone_simulator_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level_obj)
        file_handler.addFilter(performance_filter)
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(backend)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
        
        # 錯誤日誌檔案
        error_log_file = log_dir / f"drone_simulator_error_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.addFilter(performance_filter)
        error_handler.setFormatter(file_formatter)
        handlers.append(error_handler)
    
    # JSON格式輸出
    if json_output:
        json_log_file = log_dir / f"drone_simulator_json_{datetime.now().strftime('%Y%m%d')}.jsonl"
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        json_handler.setLevel(log_level_obj)
        json_handler.addFilter(performance_filter)
        json_handler.setFormatter(JSONFormatter())
        handlers.append(json_handler)
    
    # 添加處理器到根日誌器
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # 設置第三方庫的日誌級別
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # 記錄啟動資訊
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("🚁 無人機群模擬器日誌系統啟動")
    logger.info(f"📁 日誌目錄: {log_dir}")
    logger.info(f"📊 日誌級別: {log_level}")
    logger.info(f"🖥️  控制台輸出: {'啟用' if console_output else '禁用'}")
    logger.info(f"📄 檔案輸出: {'啟用' if file_output else '禁用'}")
    logger.info(f"📋 JSON輸出: {'啟用' if json_output else '禁用'}")
    logger.info("="*60)
    
    return root_logger

class PerformanceLogger:
    """效能監控日誌器"""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_fps(self, fps: float, backend: str = "Unknown"):
        """記錄FPS資訊"""
        # 創建特殊日誌記錄
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"FPS: {fps:.1f}",
            args=(),
            exc_info=None
        )
        record.fps = fps
        record.backend = backend
        self.logger.handle(record)
    
    def log_gpu_memory(self, used_mb: float, total_mb: float):
        """記錄GPU記憶體使用"""
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO, 
            pathname="",
            lineno=0,
            msg=f"GPU記憶體: {used_mb:.1f}MB / {total_mb:.1f}MB ({used_mb/total_mb*100:.1f}%)",
            args=(),
            exc_info=None
        )
        record.gpu_memory = used_mb
        record.gpu_memory_total = total_mb
        self.logger.handle(record)
    
    def log_drone_count(self, count: int):
        """記錄無人機數量"""
        record = logging.LogRecord(
            name=self.logger.name,
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"無人機數量: {count}",
            args=(),
            exc_info=None
        )
        record.drone_count = count
        self.logger.handle(record)

def get_performance_logger() -> PerformanceLogger:
    """獲取效能日誌器實例"""
    return PerformanceLogger()

def configure_gpu_logging():
    """配置GPU相關日誌"""
    try:
        from utils.gpu_utils import is_gpu_enabled
        
        if is_gpu_enabled():
            # GPU模式的詳細日誌
            gpu_logger = logging.getLogger('gpu')
            gpu_logger.info("🚀 GPU加速模式啟用")
            
            # 設置CUPY日誌
            try:
                import cupy
                cupy_logger = logging.getLogger('cupy')
                cupy_logger.setLevel(logging.WARNING)
            except ImportError:
                pass
        else:
            # CPU模式日誌
            cpu_logger = logging.getLogger('cpu')
            cpu_logger.info("🖥️ CPU計算模式")
    
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️ GPU日誌配置失敗: {e}")

def setup_debug_logging():
    """設置除錯模式日誌"""
    debug_logger = logging.getLogger('debug')
    debug_logger.setLevel(logging.DEBUG)
    
    # 創建除錯處理器
    debug_handler = logging.StreamHandler(sys.stderr)
    debug_handler.setLevel(logging.DEBUG)
    
    debug_formatter = logging.Formatter(
        '🐛 DEBUG | %(asctime)s | %(name)s:%(lineno)d | %(message)s',
        datefmt='%H:%M:%S'
    )
    debug_handler.setFormatter(debug_formatter)
    debug_logger.addHandler(debug_handler)
    
    return debug_logger

class LogContext:
    """日誌上下文管理器"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"🔄 開始: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"✅ 完成: {self.operation} (耗時: {duration.total_seconds():.2f}s)")
        else:
            self.logger.error(f"❌ 失敗: {self.operation} (耗時: {duration.total_seconds():.2f}s)")
            self.logger.error(f"錯誤: {exc_val}")

def log_operation(operation_name: str):
    """日誌操作裝飾器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            with LogContext(logger, operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# 範例使用
if __name__ == "__main__":
    # 基本設置
    setup_logging(log_level="DEBUG")
    
    # 配置GPU日誌
    configure_gpu_logging()
    
    # 測試日誌
    logger = logging.getLogger(__name__)
    logger.debug("這是除錯訊息")
    logger.info("這是資訊訊息")
    logger.warning("這是警告訊息")
    logger.error("這是錯誤訊息")
    
    # 測試效能日誌
    perf_logger = get_performance_logger()
    perf_logger.log_fps(60.0, "GPU")
    perf_logger.log_gpu_memory(1024.0, 8192.0)
    perf_logger.log_drone_count(10)
    
    # 測試上下文管理器
    with LogContext(logger, "測試操作"):
        import time
        time.sleep(0.1)
    
    logger.info("日誌系統測試完成")