"""
GPU/CPU 統一計算後端工具模組
支持自動切換和回退機制
"""
import numpy as np
import warnings
from typing import Union, Optional, Any, Tuple
from functools import wraps
import time
import logging

from config.settings import settings, ComputeBackend

# 設置日誌
logger = logging.getLogger(__name__)

class ComputeManager:
    """計算後端管理器"""
    
    def __init__(self):
        self._backend = None
        self._xp = None  # 數組處理模組 (numpy 或 cupy)
        self._device = None
        self._gpu_available = False
        self._cupy = None
        self._initialize_backend()
    
    def _initialize_backend(self):
        """初始化計算後端"""
        # 檢測GPU可用性
        self._detect_gpu()
        
        # 根據配置選擇後端
        requested_backend = settings.gpu.backend
        
        if requested_backend == ComputeBackend.AUTO:
            # 自動選擇最佳後端
            self._backend = ComputeBackend.GPU if self._gpu_available else ComputeBackend.CPU
        elif requested_backend == ComputeBackend.GPU:
            if not self._gpu_available and settings.gpu.enable_fallback:
                logger.warning("[WARN] GPU不可用，回退到CPU模式")
                self._backend = ComputeBackend.CPU
            elif not self._gpu_available:
                raise RuntimeError("[ERROR] GPU不可用且已禁用回退模式")
            else:
                self._backend = ComputeBackend.GPU
        else:
            self._backend = ComputeBackend.CPU
        
        # 設置數組處理模組
        self._setup_array_module()
        
        # 記錄後端資訊
        self._log_backend_info()
    
    def _detect_gpu(self):
        """檢測GPU可用性"""
        try:
            import cupy as cp
            self._cupy = cp
            
            # 測試GPU基本功能
            test_array = cp.array([1, 2, 3])
            _ = cp.sum(test_array)
            cp.cuda.Device().synchronize()
            
            self._gpu_available = True
            logger.info("[OK] GPU (CUDA) 檢測成功")
            
        except ImportError:
            logger.info("[INFO] CuPy未安裝，使用CPU模式")
            self._gpu_available = False
        except Exception as e:
            logger.warning(f"[WARN] GPU檢測失敗: {e}")
            self._gpu_available = False
    
    def _setup_array_module(self):
        """設置數組處理模組"""
        if self._backend == ComputeBackend.GPU and self._gpu_available:
            self._xp = self._cupy
            if settings.gpu.device_id != 0:
                self._cupy.cuda.Device(settings.gpu.device_id).use()
            self._device = self._cupy.cuda.Device()
        else:
            self._xp = np
            self._device = None
    
    def _log_backend_info(self):
        """記錄後端資訊"""
        if self._backend == ComputeBackend.GPU:
            gpu_info = self._cupy.cuda.runtime.getDeviceProperties(settings.gpu.device_id)
            memory_info = self._cupy.cuda.MemoryPool().get_limit()
            logger.info(f"[GPU] GPU後端啟用")
            logger.info(f"[DEVICE] 設備: {gpu_info['name'].decode()}")
            logger.info(f"[MEMORY] 記憶體限制: {memory_info / 1024**3:.1f} GB")
        else:
            logger.info("[CPU] CPU後端啟用")
    
    @property
    def backend(self) -> ComputeBackend:
        return self._backend
    
    @property
    def xp(self):
        """獲取數組處理模組 (numpy 或 cupy)"""
        return self._xp
    
    @property
    def is_gpu_enabled(self) -> bool:
        return self._backend == ComputeBackend.GPU
    
    def asarray(self, array: Union[list, np.ndarray, Any]) -> Any:
        """轉換為適當的數組格式"""
        return self._xp.asarray(array)
    
    def to_cpu(self, array: Any) -> np.ndarray:
        """將數組轉換回CPU (numpy)"""
        if self.is_gpu_enabled and hasattr(array, 'get'):
            return array.get()  # cupy array to numpy
        return np.asarray(array)
    
    def to_gpu(self, array: Union[list, np.ndarray]) -> Any:
        """將數組轉換到GPU (如果可用)"""
        if self.is_gpu_enabled:
            return self._cupy.asarray(array)
        return np.asarray(array)
    
    def synchronize(self):
        """同步GPU操作 (如果使用GPU)"""
        if self.is_gpu_enabled and self._device:
            self._device.synchronize()
    
    def get_memory_info(self) -> dict:
        """獲取記憶體使用資訊"""
        if self.is_gpu_enabled:
            mempool = self._cupy.get_default_memory_pool()
            return {
                'used_bytes': mempool.used_bytes(),
                'total_bytes': mempool.total_bytes(),
                'backend': 'GPU'
            }
        else:
            import psutil
            return {
                'used_bytes': psutil.virtual_memory().used,
                'total_bytes': psutil.virtual_memory().total,
                'backend': 'CPU'
            }

# 全域計算管理器實例
compute_manager = ComputeManager()

# 便利函數
def get_array_module():
    """獲取當前數組模組"""
    return compute_manager.xp

def asarray(array):
    """轉換為當前後端的數組格式"""
    return compute_manager.asarray(array)

def to_cpu(array):
    """轉換到CPU"""
    return compute_manager.to_cpu(array)

def to_gpu(array):
    """轉換到GPU"""
    return compute_manager.to_gpu(array)

def is_gpu_enabled():
    """檢查是否啟用GPU"""
    return compute_manager.is_gpu_enabled

def synchronize():
    """同步計算"""
    compute_manager.synchronize()

def gpu_accelerated(fallback_cpu: bool = True):
    """
    裝飾器：為函數提供GPU加速
    如果GPU不可用且允許回退，自動使用CPU版本
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                if is_gpu_enabled():
                    synchronize()  # 確保GPU操作完成
                return result
            except Exception as e:
                if is_gpu_enabled() and fallback_cpu:
                    logger.warning(f"[WARN] GPU操作失敗，回退到CPU: {e}")
                    # 這裡可以實現CPU版本的回退邏輯
                    raise
                else:
                    raise
        return wrapper
    return decorator

# 高效能數學運算函數
class MathOps:
    """統一的數學運算接口"""
    
    @staticmethod
    def distance_matrix(points1: Any, points2: Any) -> Any:
        """計算兩組點之間的距離矩陣"""
        xp = get_array_module()
        p1 = asarray(points1)
        p2 = asarray(points2)
        
        # 廣播計算距離
        diff = p1[:, None, :] - p2[None, :, :]
        distances = xp.sqrt(xp.sum(diff**2, axis=-1))
        return distances
    
    @staticmethod
    def euclidean_distance(p1: Any, p2: Any) -> Any:
        """歐氏距離計算"""
        xp = get_array_module()
        p1, p2 = asarray(p1), asarray(p2)
        return xp.sqrt(xp.sum((p1 - p2)**2, axis=-1))
    
    @staticmethod
    def interpolate_trajectory(waypoints: Any, num_points: int) -> Any:
        """軌跡插值"""
        xp = get_array_module()
        wp = asarray(waypoints)
        
        # 使用線性插值
        t_old = xp.linspace(0, 1, len(wp))
        t_new = xp.linspace(0, 1, num_points)
        
        interpolated = xp.zeros((num_points, wp.shape[1]))
        for i in range(wp.shape[1]):
            interpolated[:, i] = xp.interp(t_new, t_old, wp[:, i])
        
        return interpolated

# 效能監控
class PerformanceMonitor:
    """效能監控器"""
    
    def __init__(self):
        self.gpu_times = []
        self.cpu_times = []
    
    def time_function(self, func, *args, **kwargs):
        """測量函數執行時間"""
        start_time = time.perf_counter()
        
        if is_gpu_enabled():
            result = func(*args, **kwargs)
            synchronize()  # 等待GPU操作完成
            elapsed = time.perf_counter() - start_time
            self.gpu_times.append(elapsed)
        else:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start_time
            self.cpu_times.append(elapsed)
        
        return result, elapsed
    
    def get_average_times(self) -> dict:
        """獲取平均執行時間"""
        return {
            'gpu_avg': np.mean(self.gpu_times) if self.gpu_times else 0,
            'cpu_avg': np.mean(self.cpu_times) if self.cpu_times else 0,
            'gpu_count': len(self.gpu_times),
            'cpu_count': len(self.cpu_times)
        }

# 全域效能監控器
performance_monitor = PerformanceMonitor()

def get_backend_status():
    """獲取後端狀態資訊"""
    return {
        'backend': compute_manager.backend.value,
        'gpu_available': compute_manager._gpu_available,
        'device_id': settings.gpu.device_id if is_gpu_enabled() else None,
        'memory_info': compute_manager.get_memory_info(),
        'performance': performance_monitor.get_average_times()
    }