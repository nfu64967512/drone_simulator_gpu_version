"""
專業日誌配置模組
提供分級、多輸出和GPU操作專用的日誌系統
"""

import os
import sys
import logging
import logging.handlers
from datetime import datetime
from typing import Dict, Any, Optional, List
import threading
from pathlib import Path

class ColoredFormatter(logging.Formatter):
    """彩色日誌格式器（適用於終端輸出）"""
    
    # ANSI顏色代碼
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 綠色
        'WARNING': '\033[33m',  # 黃色
        'ERROR': '\033[31m',    # 紅色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重設
    }
    
    def format(self, record):
        """格式化日誌記錄"""
        # 獲取顏色
        color = self.COLORS.get(record.levelname, '')
        reset = self.COLORS['RESET']
        
        # 格式化消息
        formatted = super().format(record)
        
        # 只在支援顏色的終端中使用顏色
        if sys.stderr.isatty() and os.name != 'nt':
            return f"{color}{formatted}{reset}"
        else:
            return formatted


class GPUMemoryFilter(logging.Filter):
    """GPU記憶體相關日誌過濾器"""
    
    def __init__(self, enable_gpu_logs: bool = True):
        super().__init__()
        self.enable_gpu_logs = enable_gpu_logs
        
    def filter(self, record):
        """過濾日誌記錄"""
        # 如果禁用GPU日誌，過濾掉CuPy相關消息
        if not self.enable_gpu_logs:
            if hasattr(record, 'name'):
                if 'cupy' in record.name.lower() or 'cuda' in record.name.lower():
                    return False
        
        return True


class PerformanceLogger:
    """性能監控日誌器"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"performance.{name}")
        self.start_time = None
        
    def start(self):
        """開始計時"""
        self.start_time = datetime.now()
        self.logger.debug(f"{self.name} 開始")
        
    def end(self, extra_info: str = ""):
        """結束計時並記錄"""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"{self.name} 完成 - 用時: {elapsed:.3f}s {extra_info}")
            self.start_time = None
            return elapsed
        return 0
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.end()


class ThreadSafeHandler(logging.Handler):
    """線程安全的日誌處理器"""
    
    def __init__(self, handler):
        super().__init__()
        self.handler = handler
        self.lock = threading.RLock()
        
    def emit(self, record):
        """發送日誌記錄"""
        with self.lock:
            self.handler.emit(record)
            
    def flush(self):
        """刷新緩衝區"""
        with self.lock:
            self.handler.flush()
            
    def close(self):
        """關閉處理器"""
        with self.lock:
            self.handler.close()


class DroneSimulatorLogger:
    """無人機模擬器專用日誌系統"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化日誌系統
        
        Args:
            config: 日誌配置字典
        """
        self.config = self._merge_default_config(config or {})
        self.loggers = {}
        self.handlers = {}
        self.performance_loggers = {}
        
        self._setup_log_directory()
        self._setup_formatters()
        self._setup_handlers()
        self._setup_loggers()
        
    def _merge_default_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """合併預設配置"""
        default_config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'date_format': '%Y-%m-%d %H:%M:%S',
            
            # 檔案輸出設定
            'log_to_file': True,
            'log_directory': 'logs',
            'log_filename': 'simulator.log',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            
            # 控制台輸出設定
            'log_to_console': True,
            'console_level': 'INFO',
            'use_colors': True,
            
            # 特殊模組日誌設定
            'performance_logging': True,
            'gpu_logging': True,
            'collision_logging': True,
            
            # 日誌分類
            'categories': {
                'main': 'INFO',
                'gpu': 'INFO',
                'collision': 'INFO',
                'trajectory': 'INFO',
                'visualization': 'INFO',
                'performance': 'DEBUG'
            }
        }
        
        # 遞歸合併配置
        def merge_dict(default, custom):
            result = default.copy()
            for key, value in custom.items():
                if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result
        
        return merge_dict(default_config, config)
    
    def _setup_log_directory(self):
        """設置日誌目錄"""
        log_dir = Path(self.config['log_directory'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 創建子目錄
        for subdir in ['performance', 'errors', 'gpu']:
            (log_dir / subdir).mkdir(exist_ok=True)
    
    def _setup_formatters(self):
        """設置格式器"""
        # 標準格式器
        self.formatters = {
            'standard': logging.Formatter(
                fmt=self.config['format'],
                datefmt=self.config['date_format']
            ),
            'detailed': logging.Formatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt=self.config['date_format']
            ),
            'performance': logging.Formatter(
                fmt='%(asctime)s - PERF - %(name)s - %(message)s',
                datefmt=self.config['date_format']
            )
        }
        
        # 彩色格式器（控制台用）
        if self.config['use_colors']:
            self.formatters['colored'] = ColoredFormatter(
                fmt=self.config['format'],
                datefmt=self.config['date_format']
            )
    
    def _setup_handlers(self):
        """設置處理器"""
        self.handlers = {}
        
        # 主日誌檔案處理器
        if self.config['log_to_file']:
            main_log_path = Path(self.config['log_directory']) / self.config['log_filename']
            main_handler = logging.handlers.RotatingFileHandler(
                main_log_path,
                maxBytes=self.config['max_file_size'],
                backupCount=self.config['backup_count'],
                encoding='utf-8'
            )
            main_handler.setFormatter(self.formatters['standard'])
            self.handlers['main_file'] = ThreadSafeHandler(main_handler)
        
        # 錯誤日誌檔案處理器
        if self.config['log_to_file']:
            error_log_path = Path(self.config['log_directory']) / 'errors' / 'errors.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_path,
                maxBytes=self.config['max_file_size'] // 2,
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setFormatter(self.formatters['detailed'])
            error_handler.setLevel(logging.ERROR)
            self.handlers['error_file'] = ThreadSafeHandler(error_handler)
        
        # 性能日誌檔案處理器
        if self.config['performance_logging']:
            perf_log_path = Path(self.config['log_directory']) / 'performance' / 'performance.log'
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_path,
                maxBytes=self.config['max_file_size'] // 4,
                backupCount=2,
                encoding='utf-8'
            )
            perf_handler.setFormatter(self.formatters['performance'])
            self.handlers['performance_file'] = ThreadSafeHandler(perf_handler)
        
        # GPU日誌檔案處理器
        if self.config['gpu_logging']:
            gpu_log_path = Path(self.config['log_directory']) / 'gpu' / 'gpu_operations.log'
            gpu_handler = logging.handlers.RotatingFileHandler(
                gpu_log_path,
                maxBytes=self.config['max_file_size'] // 4,
                backupCount=2,
                encoding='utf-8'
            )
            gpu_handler.setFormatter(self.formatters['standard'])
            self.handlers['gpu_file'] = ThreadSafeHandler(gpu_handler)
        
        # 控制台處理器
        if self.config['log_to_console']:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter_name = 'colored' if self.config['use_colors'] else 'standard'
            console_handler.setFormatter(self.formatters[formatter_name])
            console_handler.setLevel(getattr(logging, self.config['console_level']))
            
            # GPU過濾器
            if not self.config['gpu_logging']:
                console_handler.addFilter(GPUMemoryFilter(False))
            
            self.handlers['console'] = console_handler
    
    def _setup_loggers(self):
        """設置日誌器"""
        # 根日誌器設定
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config['level']))
        
        # 移除預設處理器
        root_logger.handlers.clear()
        
        # 主要模組日誌器
        for category, level in self.config['categories'].items():
            logger = logging.getLogger(category)
            logger.setLevel(getattr(logging, level))
            logger.propagate = False
            
            # 添加適當的處理器
            if 'main_file' in self.handlers:
                logger.addHandler(self.handlers['main_file'])
            
            if 'console' in self.handlers:
                logger.addHandler(self.handlers['console'])
            
            if 'error_file' in self.handlers:
                logger.addHandler(self.handlers['error_file'])
            
            # 特殊處理器
            if category == 'performance' and 'performance_file' in self.handlers:
                logger.addHandler(self.handlers['performance_file'])
            
            if category == 'gpu' and 'gpu_file' in self.handlers:
                logger.addHandler(self.handlers['gpu_file'])
            
            self.loggers[category] = logger
        
        # 設定第三方庫日誌級別
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        # 設定CuPy日誌級別
        if self.config['gpu_logging']:
            logging.getLogger('cupy').setLevel(logging.INFO)
        else:
            logging.getLogger('cupy').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        獲取指定名稱的日誌器
        
        Args:
            name: 日誌器名稱
            
        Returns:
            日誌器實例
        """
        if name in self.loggers:
            return self.loggers[name]
        else:
            # 創建新的日誌器
            logger = logging.getLogger(name)
            logger.setLevel(getattr(logging, self.config['level']))
            
            # 添加基本處理器
            if 'main_file' in self.handlers:
                logger.addHandler(self.handlers['main_file'])
            if 'console' in self.handlers:
                logger.addHandler(self.handlers['console'])
            
            self.loggers[name] = logger
            return logger
    
    def get_performance_logger(self, name: str) -> PerformanceLogger:
        """
        獲取性能日誌器
        
        Args:
            name: 性能日誌器名稱
            
        Returns:
            性能日誌器實例
        """
        if name not in self.performance_loggers:
            self.performance_loggers[name] = PerformanceLogger(name)
        
        return self.performance_loggers[name]
    
    def log_system_info(self):
        """記錄系統信息"""
        main_logger = self.get_logger('main')
        
        main_logger.info("=== 系統信息 ===")
        main_logger.info(f"Python版本: {sys.version}")
        main_logger.info(f"平台: {os.name}")
        main_logger.info(f"工作目錄: {os.getcwd()}")
        
        # GPU信息
        try:
            from utils.gpu_utils import GPUSystemChecker
            checker = GPUSystemChecker()
            gpu_info = checker.get_gpu_info()
            
            if gpu_info['available']:
                main_logger.info(f"GPU: {gpu_info['name']} ({gpu_info['total_memory_mb']:.0f}MB)")
            else:
                main_logger.info("GPU: 不可用")
        except Exception as e:
            main_logger.warning(f"GPU信息獲取失敗: {e}")
    
    def log_performance_summary(self):
        """記錄性能摘要"""
        perf_logger = self.get_logger('performance')
        
        if self.performance_loggers:
            perf_logger.info("=== 性能摘要 ===")
            for name, logger in self.performance_loggers.items():
                perf_logger.info(f"模組: {name}")
        else:
            perf_logger.info("沒有性能數據")
    
    def flush_all(self):
        """刷新所有處理器"""
        for handler in self.handlers.values():
            try:
                handler.flush()
            except Exception:
                pass
    
    def close_all(self):
        """關閉所有處理器"""
        for handler in self.handlers.values():
            try:
                handler.close()
            except Exception:
                pass
        
        # 清理
        self.handlers.clear()
        self.loggers.clear()
        self.performance_loggers.clear()


# 全域日誌系統實例
_logger_system: Optional[DroneSimulatorLogger] = None

def setup_logging(config: Optional[Dict[str, Any]] = None):
    """
    設置全域日誌系統
    
    Args:
        config: 日誌配置
    """
    global _logger_system
    
    if _logger_system is not None:
        _logger_system.close_all()
    
    _logger_system = DroneSimulatorLogger(config)
    _logger_system.log_system_info()

def get_logger(name: str) -> logging.Logger:
    """
    獲取日誌器
    
    Args:
        name: 日誌器名稱
        
    Returns:
        日誌器實例
    """
    global _logger_system
    
    if _logger_system is None:
        setup_logging()
    
    return _logger_system.get_logger(name)

def get_performance_logger(name: str) -> PerformanceLogger:
    """
    獲取性能日誌器
    
    Args:
        name: 性能日誌器名稱
        
    Returns:
        性能日誌器實例
    """
    global _logger_system
    
    if _logger_system is None:
        setup_logging()
    
    return _logger_system.get_performance_logger(name)

def log_gpu_operation(operation: str, duration: float, memory_mb: float = 0):
    """
    記錄GPU操作
    
    Args:
        operation: 操作名稱
        duration: 持續時間（秒）
        memory_mb: 記憶體使用量（MB）
    """
    gpu_logger = get_logger('gpu')
    
    if memory_mb > 0:
        gpu_logger.info(f"{operation} - 時間: {duration:.3f}s, 記憶體: {memory_mb:.1f}MB")
    else:
        gpu_logger.info(f"{operation} - 時間: {duration:.3f}s")

def log_collision_event(event_type: str, details: Dict[str, Any]):
    """
    記錄碰撞事件
    
    Args:
        event_type: 事件類型
        details: 事件詳情
    """
    collision_logger = get_logger('collision')
    
    detail_str = ", ".join([f"{k}: {v}" for k, v in details.items()])
    collision_logger.warning(f"{event_type} - {detail_str}")

def cleanup_logging():
    """清理日誌系統"""
    global _logger_system
    
    if _logger_system:
        _logger_system.close_all()
        _logger_system = None

# 自動設置
def _auto_setup():
    """自動設置日誌系統"""
    if _logger_system is None:
        setup_logging()

# 當模組被導入時自動設置
_auto_setup()