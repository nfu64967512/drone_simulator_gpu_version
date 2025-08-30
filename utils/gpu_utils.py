"""
GPU工具模組 - 完整版本
提供GPU/CPU後端管理、性能監控和系統檢測功能
"""

import os
import sys
import time
import platform
import logging
import psutil
from typing import Dict, Any, Optional, Union, Callable
import numpy as np
from functools import wraps

# GPU支援檢測
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy GPU支援已載入")
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to NumPy
    print("CuPy 不可用，將使用NumPy作為回退")

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """GPU記憶體管理器"""
    
    def __init__(self):
        self.use_memory_pool = True
        self.memory_limit = 0.8  # 使用80%的GPU記憶體
        
    def setup_memory_pool(self):
        """設置GPU記憶體池"""
        if GPU_AVAILABLE:
            try:
                # 設置記憶體池
                mempool = cp.get_default_memory_pool()
                
                # 獲取GPU總記憶體
                device = cp.cuda.Device()
                with device:
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    limit = int(total_mem * self.memory_limit)
                    mempool.set_limit(size=limit)
                
                logger.info(f"GPU記憶體池設置完成，限制: {limit / (1024**2):.1f} MB")
                
            except Exception as e:
                logger.warning(f"GPU記憶體池設置失敗: {e}")
    
    def cleanup(self):
        """清理GPU記憶體"""
        if GPU_AVAILABLE:
            try:
                # 清理記憶體池
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                
                logger.info("GPU記憶體清理完成")
                
            except Exception as e:
                logger.warning(f"GPU記憶體清理失敗: {e}")
    
    def get_memory_info(self) -> Dict[str, float]:
        """獲取GPU記憶體使用信息"""
        if not GPU_AVAILABLE:
            return {'available': False}
        
        try:
            device = cp.cuda.Device()
            with device:
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                used_mem = total_mem - free_mem
                
                # 獲取記憶體池信息
                mempool = cp.get_default_memory_pool()
                pool_used = mempool.used_bytes()
                
                return {
                    'available': True,
                    'total_mb': total_mem / (1024**2),
                    'free_mb': free_mem / (1024**2),
                    'used_mb': used_mem / (1024**2),
                    'pool_used_mb': pool_used / (1024**2),
                    'usage_percent': (used_mem / total_mem) * 100
                }
                
        except Exception as e:
            logger.error(f"獲取GPU記憶體信息失敗: {e}")
            return {'available': False, 'error': str(e)}


class GPUSystemChecker:
    """GPU系統檢查器"""
    
    def __init__(self):
        self.gpu_info = self._detect_gpu_info()
    
    def _safe_decode_gpu_name(self, name_value) -> str:
        """
        安全地處理GPU名稱，兼容不同的CuPy版本
        
        Args:
            name_value: GPU名稱值，可能是bytes、str或其他類型
            
        Returns:
            解碼後的字符串
        """
        try:
            if isinstance(name_value, bytes):
                return name_value.decode('utf-8')
            elif isinstance(name_value, str):
                return name_value
            else:
                return str(name_value)
        except Exception as e:
            logger.warning(f"GPU名稱解碼失敗: {e}")
            return "Unknown GPU"
        
    def _detect_gpu_info(self) -> Dict[str, Any]:
        """檢測GPU信息"""
        if not GPU_AVAILABLE:
            return {'available': False, 'reason': 'CuPy not installed'}
        
        try:
            # 檢查CUDA運行時
            runtime_version = cp.cuda.runtime.runtimeGetVersion()
            driver_version = cp.cuda.runtime.driverGetVersion()
            
            # 檢查GPU設備
            device_count = cp.cuda.runtime.getDeviceCount()
            
            if device_count == 0:
                return {'available': False, 'reason': 'No CUDA devices found'}
            
            # 獲取第一個GPU的詳細信息
            device = cp.cuda.Device(0)
            with device:
                attributes = device.attributes
                free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                
                # 安全地獲取GPU名稱
                gpu_name = self._safe_decode_gpu_name(attributes.get('Name', 'Unknown'))
                
                return {
                    'available': True,
                    'device_count': device_count,
                    'name': gpu_name,
                    'compute_capability': f"{attributes['ComputeCapabilityMajor']}.{attributes['ComputeCapabilityMinor']}",
                    'total_memory_mb': total_mem / (1024**2),
                    'free_memory_mb': free_mem / (1024**2),
                    'multiprocessor_count': attributes['MultiprocessorCount'],
                    'max_threads_per_block': attributes['MaxThreadsPerBlock'],
                    'runtime_version': runtime_version,
                    'driver_version': driver_version
                }
                
        except Exception as e:
            logger.error(f"GPU檢測失敗: {e}")
            return {'available': False, 'reason': str(e)}
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """獲取GPU信息"""
        return self.gpu_info.copy()
    
    def get_system_info(self) -> Dict[str, Any]:
        """獲取完整系統信息"""
        info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'python_version': platform.python_version(),
            'numpy_version': np.__version__,
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'gpu_available': self.gpu_info['available']
        }
        
        if GPU_AVAILABLE:
            info['cupy_version'] = cp.__version__
            info['cuda_available'] = True
        else:
            info['cuda_available'] = False
        
        info.update(self.gpu_info)
        return info
    
    def is_gpu_suitable(self, min_memory_gb: float = 2.0, 
                       min_compute: tuple = (3, 5)) -> bool:
        """檢查GPU是否適合使用"""
        if not self.gpu_info['available']:
            return False
        
        try:
            # 檢查記憶體
            total_memory_gb = self.gpu_info['total_memory_mb'] / 1024
            if total_memory_gb < min_memory_gb:
                logger.warning(f"GPU記憶體不足: {total_memory_gb:.1f}GB < {min_memory_gb}GB")
                return False
            
            # 檢查計算能力
            compute_capability = self.gpu_info['compute_capability']
            major, minor = map(int, compute_capability.split('.'))
            if (major, minor) < min_compute:
                logger.warning(f"GPU計算能力不足: {compute_capability} < {min_compute[0]}.{min_compute[1]}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"GPU適用性檢查失敗: {e}")
            return False


def get_array_module(use_gpu: bool = True) -> Any:
    """
    獲取陣列運算模組 (CuPy或NumPy)
    
    Args:
        use_gpu: 是否使用GPU
        
    Returns:
        CuPy或NumPy模組
    """
    if use_gpu and GPU_AVAILABLE:
        return cp
    else:
        return np


def ensure_gpu_compatibility(func: Callable) -> Callable:
    """
    確保GPU相容性的裝飾器
    自動處理GPU/CPU數據轉換和錯誤回退
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            # 檢查是否使用GPU
            if hasattr(self, 'use_gpu') and self.use_gpu and GPU_AVAILABLE:
                return func(self, *args, **kwargs)
            else:
                # CPU模式：確保使用NumPy
                original_xp = getattr(self, 'xp', np)
                self.xp = np
                result = func(self, *args, **kwargs)
                self.xp = original_xp
                return result
                
        except Exception as e:
            # GPU錯誤時自動回退到CPU
            if hasattr(self, 'use_gpu') and self.use_gpu:
                logger.warning(f"GPU操作失敗，回退到CPU: {e}")
                self.xp = np
                return func(self, *args, **kwargs)
            else:
                raise e
    
    return wrapper


def setup_gpu_environment(device_id: int = 0, memory_pool: bool = True):
    """
    設置GPU環境
    
    Args:
        device_id: GPU設備ID
        memory_pool: 是否啟用記憶體池
    """
    if not GPU_AVAILABLE:
        logger.info("GPU不可用，跳過GPU環境設置")
        return
    
    try:
        # 設置默認設備
        cp.cuda.Device(device_id).use()
        logger.info(f"設置GPU設備: {device_id}")
        
        # 設置記憶體池
        if memory_pool:
            memory_manager = GPUMemoryManager()
            memory_manager.setup_memory_pool()
        
        # 預熱GPU（執行簡單操作）
        _ = cp.array([1, 2, 3])
        logger.info("GPU環境設置完成")
        
    except Exception as e:
        logger.error(f"GPU環境設置失敗: {e}")
        raise e


class PerformanceBenchmark:
    """性能基準測試"""
    
    def __init__(self):
        self.results = {}
        
    def benchmark_array_operations(self, size: int = 10000, iterations: int = 100):
        """基準測試陣列操作"""
        logger.info(f"開始陣列操作基準測試: size={size}, iterations={iterations}")
        
        # 準備測試數據
        data_np = np.random.random((size, size)).astype(np.float32)
        
        # CPU測試
        cpu_times = []
        for _ in range(iterations):
            start = time.time()
            result_cpu = np.dot(data_np, data_np.T)
            cpu_times.append(time.time() - start)
        
        cpu_avg = np.mean(cpu_times)
        
        # GPU測試
        gpu_avg = float('inf')
        if GPU_AVAILABLE:
            try:
                data_gpu = cp.asarray(data_np)
                cp.cuda.Stream.null.synchronize()  # 確保數據已傳輸
                
                gpu_times = []
                for _ in range(iterations):
                    start = time.time()
                    result_gpu = cp.dot(data_gpu, data_gpu.T)
                    cp.cuda.Stream.null.synchronize()  # 等待計算完成
                    gpu_times.append(time.time() - start)
                
                gpu_avg = np.mean(gpu_times)
                
            except Exception as e:
                logger.warning(f"GPU測試失敗: {e}")
        
        speedup = cpu_avg / gpu_avg if gpu_avg != float('inf') else 0
        
        self.results['array_operations'] = {
            'cpu_time_ms': cpu_avg * 1000,
            'gpu_time_ms': gpu_avg * 1000 if gpu_avg != float('inf') else None,
            'speedup': speedup,
            'size': size,
            'iterations': iterations
        }
        
        logger.info(f"陣列操作測試完成: CPU={cpu_avg*1000:.2f}ms, GPU={gpu_avg*1000:.2f}ms, 加速={speedup:.2f}x")
    
    def benchmark_distance_matrix(self, num_points: int = 1000, iterations: int = 10):
        """基準測試距離矩陣計算"""
        logger.info(f"開始距離矩陣基準測試: points={num_points}, iterations={iterations}")
        
        # 準備3D點數據
        points_np = np.random.random((num_points, 3)).astype(np.float32)
        
        # CPU測試
        cpu_times = []
        for _ in range(iterations):
            start = time.time()
            diff = points_np[:, np.newaxis, :] - points_np[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff**2, axis=2))
            cpu_times.append(time.time() - start)
        
        cpu_avg = np.mean(cpu_times)
        
        # GPU測試
        gpu_avg = float('inf')
        if GPU_AVAILABLE:
            try:
                points_gpu = cp.asarray(points_np)
                
                gpu_times = []
                for _ in range(iterations):
                    start = time.time()
                    diff = points_gpu[:, None, :] - points_gpu[None, :, :]
                    distances = cp.sqrt(cp.sum(diff**2, axis=2))
                    cp.cuda.Stream.null.synchronize()
                    gpu_times.append(time.time() - start)
                
                gpu_avg = np.mean(gpu_times)
                
            except Exception as e:
                logger.warning(f"GPU距離矩陣測試失敗: {e}")
        
        speedup = cpu_avg / gpu_avg if gpu_avg != float('inf') else 0
        
        self.results['distance_matrix'] = {
            'cpu_time_ms': cpu_avg * 1000,
            'gpu_time_ms': gpu_avg * 1000 if gpu_avg != float('inf') else None,
            'speedup': speedup,
            'num_points': num_points,
            'iterations': iterations
        }
        
        logger.info(f"距離矩陣測試完成: CPU={cpu_avg*1000:.2f}ms, GPU={gpu_avg*1000:.2f}ms, 加速={speedup:.2f}x")
    
    def benchmark_memory_bandwidth(self, size_mb: int = 100):
        """基準測試記憶體頻寬"""
        logger.info(f"開始記憶體頻寬測試: size={size_mb}MB")
        
        # 計算陣列大小
        elements = (size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        data_np = np.random.random(elements).astype(np.float32)
        
        # CPU記憶體頻寬測試
        start = time.time()
        for _ in range(10):
            result = np.copy(data_np)
        cpu_time = (time.time() - start) / 10
        cpu_bandwidth = (size_mb * 2) / cpu_time  # 讀取+寫入
        
        # GPU記憶體頻寬測試
        gpu_bandwidth = 0
        if GPU_AVAILABLE:
            try:
                # CPU -> GPU 傳輸
                start = time.time()
                data_gpu = cp.asarray(data_np)
                cp.cuda.Stream.null.synchronize()
                h2d_time = time.time() - start
                h2d_bandwidth = size_mb / h2d_time
                
                # GPU -> CPU 傳輸
                start = time.time()
                result_cpu = cp.asnumpy(data_gpu)
                d2h_time = time.time() - start
                d2h_bandwidth = size_mb / d2h_time
                
                # GPU內部複製
                start = time.time()
                for _ in range(10):
                    result = cp.copy(data_gpu)
                    cp.cuda.Stream.null.synchronize()
                gpu_time = (time.time() - start) / 10
                gpu_bandwidth = (size_mb * 2) / gpu_time
                
                self.results['memory_bandwidth'] = {
                    'cpu_bandwidth_mb_s': cpu_bandwidth,
                    'gpu_bandwidth_mb_s': gpu_bandwidth,
                    'h2d_bandwidth_mb_s': h2d_bandwidth,
                    'd2h_bandwidth_mb_s': d2h_bandwidth,
                    'size_mb': size_mb
                }
                
            except Exception as e:
                logger.warning(f"GPU記憶體頻寬測試失敗: {e}")
                self.results['memory_bandwidth'] = {
                    'cpu_bandwidth_mb_s': cpu_bandwidth,
                    'gpu_bandwidth_mb_s': None,
                    'size_mb': size_mb
                }
        
        logger.info(f"記憶體頻寬測試完成: CPU={cpu_bandwidth:.1f}MB/s, GPU={gpu_bandwidth:.1f}MB/s")
    
    def get_results(self) -> Dict[str, Any]:
        """獲取測試結果"""
        return self.results.copy()
    
    def print_summary(self):
        """打印測試摘要"""
        if not self.results:
            print("沒有測試結果")
            return
        
        print("\n=== 性能基準測試結果 ===")
        
        for test_name, result in self.results.items():
            print(f"\n{test_name.upper()}:")
            for key, value in result.items():
                if isinstance(value, float):
                    if 'time' in key:
                        print(f"  {key}: {value:.2f}")
                    elif 'bandwidth' in key:
                        print(f"  {key}: {value:.1f}")
                    elif 'speedup' in key:
                        print(f"  {key}: {value:.2f}x")
                    else:
                        print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")


def run_performance_benchmark():
    """運行完整的性能基準測試"""
    print("開始性能基準測試...")
    
    benchmark = PerformanceBenchmark()
    
    # 檢查系統
    checker = GPUSystemChecker()
    system_info = checker.get_system_info()
    
    print(f"\n系統信息:")
    print(f"  平台: {system_info['platform']}")
    print(f"  CPU: {system_info['cpu_count']} 核心")
    print(f"  記憶體: {system_info['memory_gb']:.1f} GB")
    print(f"  GPU: {'可用' if system_info['gpu_available'] else '不可用'}")
    if system_info['gpu_available']:
        print(f"  GPU名稱: {system_info.get('name', 'Unknown')}")
        print(f"  GPU記憶體: {system_info.get('total_memory_mb', 0)/1024:.1f} GB")
    
    # 運行測試
    try:
        benchmark.benchmark_array_operations(size=2000, iterations=50)
        benchmark.benchmark_distance_matrix(num_points=500, iterations=20)
        benchmark.benchmark_memory_bandwidth(size_mb=50)
        
        benchmark.print_summary()
        
    except Exception as e:
        logger.error(f"基準測試失敗: {e}")
        print(f"測試失敗: {e}")


# 便利函數
def auto_detect_backend() -> str:
    """自動檢測最佳後端"""
    checker = GPUSystemChecker()
    
    if checker.is_gpu_suitable():
        return "gpu"
    else:
        return "cpu"


def get_optimal_batch_size(base_size: int = 1000, use_gpu: bool = True) -> int:
    """
    獲取最佳批次大小
    
    Args:
        base_size: 基礎批次大小
        use_gpu: 是否使用GPU
        
    Returns:
        推薦的批次大小
    """
    if not use_gpu or not GPU_AVAILABLE:
        return base_size
    
    try:
        # 根據GPU記憶體調整批次大小
        manager = GPUMemoryManager()
        memory_info = manager.get_memory_info()
        
        if memory_info['available']:
            free_mb = memory_info['free_mb']
            
            # 簡單的啟發式規則
            if free_mb > 4000:  # > 4GB
                return base_size * 4
            elif free_mb > 2000:  # > 2GB
                return base_size * 2
            else:
                return base_size
        else:
            return base_size
            
    except Exception:
        return base_size


# 模組初始化
_gpu_initialized = False
_memory_manager = None

def initialize_gpu_module():
    """初始化GPU模組"""
    global _gpu_initialized, _memory_manager
    
    if _gpu_initialized:
        return
    
    if GPU_AVAILABLE:
        _memory_manager = GPUMemoryManager()
        logger.info("GPU模組初始化完成")
    else:
        logger.info("GPU不可用，使用CPU模式")
    
    _gpu_initialized = True


def cleanup_gpu_module():
    """清理GPU模組"""
    global _memory_manager
    
    if _memory_manager:
        _memory_manager.cleanup()
        logger.info("GPU模組清理完成")


# 自動初始化
initialize_gpu_module()