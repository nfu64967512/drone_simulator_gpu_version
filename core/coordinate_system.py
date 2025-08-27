"""
GPU加速座標轉換系統
支持大批量經緯度到公尺座標系統的高效轉換
"""
import numpy as np
import math
import logging
from typing import Union, Tuple, List, Any, Optional
from dataclasses import dataclass
from numba import cuda
import time

# 導入GPU工具
from utils.gpu_utils import (
    get_array_module, asarray, to_cpu, to_gpu, is_gpu_enabled,
    synchronize, gpu_accelerated, performance_monitor
)
from config.settings import settings

logger = logging.getLogger(__name__)

# 地球相關常數
EARTH_RADIUS = 6378137.0  # WGS84橢球半長軸 (公尺)
EARTH_FLATTENING = 1.0 / 298.257223563  # WGS84扁率
EARTH_ECCENTRICITY_SQ = 2 * EARTH_FLATTENING - EARTH_FLATTENING ** 2

@dataclass
class ReferencePoint:
    """參考點座標"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    utm_x: float = 0.0
    utm_y: float = 0.0
    utm_zone: int = 0

class GPUCoordinateConverter:
    """GPU加速座標轉換器"""
    
    def __init__(self, reference_lat: float = 0.0, reference_lon: float = 0.0):
        self.xp = get_array_module()
        
        # 設置參考點
        self.reference_point = ReferencePoint(
            latitude=reference_lat,
            longitude=reference_lon
        )
        
        # GPU優化參數
        self.batch_size = settings.gpu.batch_size
        self.use_high_precision = True  # 使用雙精度浮點
        
        # CUDA kernels可用性
        self.cuda_kernels_available = False
        self._init_cuda_kernels()
        
        # 效能統計
        self.conversion_count = 0
        self.total_conversion_time = 0.0
        
        logger.info(f"🗺️ GPU座標轉換器初始化 (參考點: {reference_lat:.6f}, {reference_lon:.6f})")

    def _init_cuda_kernels(self):
        """初始化CUDA kernels"""
        try:
            if is_gpu_enabled() and cuda.is_available():
                self._compile_coordinate_kernels()
                self.cuda_kernels_available = True
                logger.info("🚀 座標轉換 CUDA kernels 已編譯")
            else:
                logger.info("⚡ 使用GPU陣列運算進行座標轉換")
        except Exception as e:
            logger.warning(f"⚠️ 座標轉換 CUDA kernels 編譯失敗: {e}")

    def _compile_coordinate_kernels(self):
        """編譯座標轉換CUDA kernels"""
        
        @cuda.jit
        def lat_lon_to_meters_kernel(
            latitudes, longitudes, altitudes,
            ref_lat, ref_lon,
            result_x, result_y, result_z,
            n_points
        ):
            """批次經緯度到公尺轉換kernel"""
            idx = cuda.grid(1)
            
            if idx < n_points:
                lat = math.radians(latitudes[idx])
                lon = math.radians(longitudes[idx])
                alt = altitudes[idx]
                
                ref_lat_rad = math.radians(ref_lat)
                ref_lon_rad = math.radians(ref_lon)
                
                # 計算相對位移
                delta_lat = lat - ref_lat_rad
                delta_lon = lon - ref_lon_rad
                
                # 使用墨卡托投影的近似計算
                cos_ref_lat = math.cos(ref_lat_rad)
                
                # X方向 (東)
                result_x[idx] = delta_lon * EARTH_RADIUS * cos_ref_lat
                
                # Y方向 (北)
                result_y[idx] = delta_lat * EARTH_RADIUS
                
                # Z方向 (高度)
                result_z[idx] = alt

        @cuda.jit
        def meters_to_lat_lon_kernel(
            x_coords, y_coords, z_coords,
            ref_lat, ref_lon,
            result_lat, result_lon, result_alt,
            n_points
        ):
            """批次公尺到經緯度轉換kernel"""
            idx = cuda.grid(1)
            
            if idx < n_points:
                x = x_coords[idx]
                y = y_coords[idx]
                z = z_coords[idx]
                
                ref_lat_rad = math.radians(ref_lat)
                ref_lon_rad = math.radians(ref_lon)
                cos_ref_lat = math.cos(ref_lat_rad)
                
                # 反向轉換
                delta_lon = x / (EARTH_RADIUS * cos_ref_lat)
                delta_lat = y / EARTH_RADIUS
                
                # 轉回經緯度
                result_lat[idx] = math.degrees(ref_lat_rad + delta_lat)
                result_lon[idx] = math.degrees(ref_lon_rad + delta_lon)
                result_alt[idx] = z

        self.lat_lon_to_meters_kernel = lat_lon_to_meters_kernel
        self.meters_to_lat_lon_kernel = meters_to_lat_lon_kernel

    def set_reference_point(self, latitude: float, longitude: float, altitude: float = 0.0):
        """設置參考點"""
        self.reference_point.latitude = latitude
        self.reference_point.longitude = longitude  
        self.reference_point.altitude = altitude
        
        logger.info(f"📍 參考點已更新: ({latitude:.6f}, {longitude:.6f}, {altitude:.1f}m)")

    @gpu_accelerated()
    def batch_convert_to_meters(
        self, 
        latitudes: Union[List, np.ndarray, Any],
        longitudes: Union[List, np.ndarray, Any], 
        altitudes: Union[List, np.ndarray, Any]
    ) -> Any:
        """
        批次轉換經緯度到公尺座標系統
        
        Args:
            latitudes: 緯度陣列
            longitudes: 經度陣列  
            altitudes: 高度陣列
            
        Returns:
            形狀為 (n_points, 3) 的位置陣列 [x, y, z]
        """
        start_time = time.perf_counter()
        
        # 轉換為GPU陣列
        lats = asarray(latitudes, dtype=self.xp.float64 if self.use_high_precision else self.xp.float32)
        lons = asarray(longitudes, dtype=self.xp.float64 if self.use_high_precision else self.xp.float32)
        alts = asarray(altitudes, dtype=self.xp.float64 if self.use_high_precision else self.xp.float32)
        
        n_points = len(lats)
        
        if n_points == 0:
            return self.xp.zeros((0, 3))
        
        try:
            if self.cuda_kernels_available and n_points >= 1000:
                # 使用CUDA kernels進行大批量轉換
                result = self._convert_to_meters_cuda(lats, lons, alts)
            else:
                # 使用GPU陣列運算
                result = self._convert_to_meters_vectorized(lats, lons, alts)
            
            # 同步GPU操作
            if is_gpu_enabled():
                synchronize()
            
            # 更新效能統計
            conversion_time = time.perf_counter() - start_time
            self.conversion_count += n_points
            self.total_conversion_time += conversion_time
            
            logger.debug(f"📊 批次轉換 {n_points} 個點 (耗時: {conversion_time*1000:.2f}ms)")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ GPU座標轉換失敗: {e}")
            # 回退到CPU轉換
            return self._convert_to_meters_cpu_fallback(
                to_cpu(lats), to_cpu(lons), to_cpu(alts)
            )

    def _convert_to_meters_cuda(self, lats: Any, lons: Any, alts: Any) -> Any:
        """使用CUDA kernels進行轉換"""
        n_points = len(lats)
        
        # 分配輸出陣列
        result_x = self.xp.zeros(n_points, dtype=lats.dtype)
        result_y = self.xp.zeros(n_points, dtype=lats.dtype)
        result_z = self.xp.zeros(n_points, dtype=lats.dtype)
        
        # 設置CUDA執行參數
        threads_per_block = 256
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
        
        # 執行kernel
        self.lat_lon_to_meters_kernel[blocks_per_grid, threads_per_block](
            lats, lons, alts,
            self.reference_point.latitude,
            self.reference_point.longitude,
            result_x, result_y, result_z,
            n_points
        )
        
        # 組合結果
        result = self.xp.column_stack([result_x, result_y, result_z])
        return result

    @gpu_accelerated()
    def _convert_to_meters_vectorized(self, lats: Any, lons: Any, alts: Any) -> Any:
        """使用GPU陣列運算進行向量化轉換"""
        
        # 轉換為弧度
        lats_rad = self.xp.radians(lats)
        lons_rad = self.xp.radians(lons)
        ref_lat_rad = self.xp.radians(self.reference_point.latitude)
        ref_lon_rad = self.xp.radians(self.reference_point.longitude)
        
        # 計算相對位移
        delta_lat = lats_rad - ref_lat_rad
        delta_lon = lons_rad - ref_lon_rad
        
        # 考慮地球曲率的更精確計算
        cos_ref_lat = self.xp.cos(ref_lat_rad)
        cos_avg_lat = self.xp.cos((lats_rad + ref_lat_rad) / 2.0)  # 平均緯度的餘弦值
        
        # 高精度座標轉換
        if self.use_high_precision:
            # 使用橢球面計算
            N = EARTH_RADIUS / self.xp.sqrt(1 - EARTH_ECCENTRICITY_SQ * self.xp.sin(ref_lat_rad)**2)
            
            # X方向 (東)
            x_coords = delta_lon * N * cos_ref_lat
            
            # Y方向 (北) - 考慮子午線曲率半徑  
            M = EARTH_RADIUS * (1 - EARTH_ECCENTRICITY_SQ) / (1 - EARTH_ECCENTRICITY_SQ * self.xp.sin(ref_lat_rad)**2)**(3/2)
            y_coords = delta_lat * M
        else:
            # 簡化計算（更快但精度稍低）
            x_coords = delta_lon * EARTH_RADIUS * cos_avg_lat
            y_coords = delta_lat * EARTH_RADIUS
        
        # Z方向 (高度)
        z_coords = alts - self.reference_point.altitude
        
        # 組合結果
        result = self.xp.column_stack([x_coords, y_coords, z_coords])
        return result

    def _convert_to_meters_cpu_fallback(self, lats: np.ndarray, lons: np.ndarray, alts: np.ndarray) -> np.ndarray:
        """CPU回退轉換"""
        result = np.zeros((len(lats), 3))
        
        ref_lat_rad = math.radians(self.reference_point.latitude)
        ref_lon_rad = math.radians(self.reference_point.longitude)
        
        for i in range(len(lats)):
            lat_rad = math.radians(lats[i])
            lon_rad = math.radians(lons[i])
            
            delta_lat = lat_rad - ref_lat_rad
            delta_lon = lon_rad - ref_lon_rad
            
            cos_ref_lat = math.cos(ref_lat_rad)
            
            result[i, 0] = delta_lon * EARTH_RADIUS * cos_ref_lat  # X (東)
            result[i, 1] = delta_lat * EARTH_RADIUS  # Y (北)
            result[i, 2] = alts[i] - self.reference_point.altitude  # Z (高度)
        
        return result

    @gpu_accelerated()
    def batch_convert_to_lat_lon(
        self, 
        x_coords: Union[List, np.ndarray, Any],
        y_coords: Union[List, np.ndarray, Any],
        z_coords: Union[List, np.ndarray, Any]
    ) -> Tuple[Any, Any, Any]:
        """
        批次轉換公尺座標到經緯度
        
        Args:
            x_coords: X座標陣列 (東)
            y_coords: Y座標陣列 (北)
            z_coords: Z座標陣列 (高度)
            
        Returns:
            (緯度陣列, 經度陣列, 高度陣列)
        """
        start_time = time.perf_counter()
        
        # 轉換為GPU陣列
        x_gpu = asarray(x_coords, dtype=self.xp.float64 if self.use_high_precision else self.xp.float32)
        y_gpu = asarray(y_coords, dtype=self.xp.float64 if self.use_high_precision else self.xp.float32)  
        z_gpu = asarray(z_coords, dtype=self.xp.float64 if self.use_high_precision else self.xp.float32)
        
        n_points = len(x_gpu)
        
        try:
            if self.cuda_kernels_available and n_points >= 1000:
                # 使用CUDA kernels
                lats, lons, alts = self._convert_to_lat_lon_cuda(x_gpu, y_gpu, z_gpu)
            else:
                # 使用GPU陣列運算
                lats, lons, alts = self._convert_to_lat_lon_vectorized(x_gpu, y_gpu, z_gpu)
            
            # 同步GPU操作
            if is_gpu_enabled():
                synchronize()
            
            # 更新效能統計
            conversion_time = time.perf_counter() - start_time
            self.conversion_count += n_points
            self.total_conversion_time += conversion_time
            
            return lats, lons, alts
            
        except Exception as e:
            logger.error(f"❌ GPU反向座標轉換失敗: {e}")
            # 回退到CPU轉換
            return self._convert_to_lat_lon_cpu_fallback(
                to_cpu(x_gpu), to_cpu(y_gpu), to_cpu(z_gpu)
            )

    def _convert_to_lat_lon_cuda(self, x_coords: Any, y_coords: Any, z_coords: Any) -> Tuple[Any, Any, Any]:
        """使用CUDA kernels進行反向轉換"""
        n_points = len(x_coords)
        
        # 分配輸出陣列
        result_lat = self.xp.zeros(n_points, dtype=x_coords.dtype)
        result_lon = self.xp.zeros(n_points, dtype=x_coords.dtype)
        result_alt = self.xp.zeros(n_points, dtype=x_coords.dtype)
        
        # 設置CUDA執行參數
        threads_per_block = 256
        blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
        
        # 執行kernel
        self.meters_to_lat_lon_kernel[blocks_per_grid, threads_per_block](
            x_coords, y_coords, z_coords,
            self.reference_point.latitude,
            self.reference_point.longitude,
            result_lat, result_lon, result_alt,
            n_points
        )
        
        return result_lat, result_lon, result_alt

    @gpu_accelerated()
    def _convert_to_lat_lon_vectorized(self, x_coords: Any, y_coords: Any, z_coords: Any) -> Tuple[Any, Any, Any]:
        """使用GPU陣列運算進行反向轉換"""
        ref_lat_rad = self.xp.radians(self.reference_point.latitude)
        ref_lon_rad = self.xp.radians(self.reference_point.longitude)
        
        cos_ref_lat = self.xp.cos(ref_lat_rad)
        
        # 反向轉換
        if self.use_high_precision:
            # 高精度轉換
            N = EARTH_RADIUS / self.xp.sqrt(1 - EARTH_ECCENTRICITY_SQ * self.xp.sin(ref_lat_rad)**2)
            M = EARTH_RADIUS * (1 - EARTH_ECCENTRICITY_SQ) / (1 - EARTH_ECCENTRICITY_SQ * self.xp.sin(ref_lat_rad)**2)**(3/2)
            
            delta_lon = x_coords / (N * cos_ref_lat)
            delta_lat = y_coords / M
        else:
            # 簡化轉換
            delta_lon = x_coords / (EARTH_RADIUS * cos_ref_lat)
            delta_lat = y_coords / EARTH_RADIUS
        
        # 轉回經緯度
        lats = self.xp.degrees(ref_lat_rad + delta_lat)
        lons = self.xp.degrees(ref_lon_rad + delta_lon)
        alts = z_coords + self.reference_point.altitude
        
        return lats, lons, alts

    def _convert_to_lat_lon_cpu_fallback(self, x_coords: np.ndarray, y_coords: np.ndarray, z_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CPU回退反向轉換"""
        ref_lat_rad = math.radians(self.reference_point.latitude)
        ref_lon_rad = math.radians(self.reference_point.longitude)
        cos_ref_lat = math.cos(ref_lat_rad)
        
        lats = np.degrees(ref_lat_rad + y_coords / EARTH_RADIUS)
        lons = np.degrees(ref_lon_rad + x_coords / (EARTH_RADIUS * cos_ref_lat))
        alts = z_coords + self.reference_point.altitude
        
        return lats, lons, alts

    @gpu_accelerated()
    def calculate_distances_batch(
        self, 
        positions1: Any, 
        positions2: Any
    ) -> Any:
        """批次計算兩組位置間的距離"""
        pos1 = asarray(positions1)
        pos2 = asarray(positions2)
        
        # 使用GPU向量化計算
        diff = pos1 - pos2
        distances = self.xp.sqrt(self.xp.sum(diff**2, axis=-1))
        
        return distances

    @gpu_accelerated()
    def calculate_bearing_batch(
        self,
        lat1: Any, lon1: Any,
        lat2: Any, lon2: Any
    ) -> Any:
        """批次計算方位角"""
        # 轉為GPU陣列和弧度
        lat1_rad = self.xp.radians(asarray(lat1))
        lon1_rad = self.xp.radians(asarray(lon1))
        lat2_rad = self.xp.radians(asarray(lat2))
        lon2_rad = self.xp.radians(asarray(lon2))
        
        # 計算方位角
        dlon = lon2_rad - lon1_rad
        
        y = self.xp.sin(dlon) * self.xp.cos(lat2_rad)
        x = (self.xp.cos(lat1_rad) * self.xp.sin(lat2_rad) - 
             self.xp.sin(lat1_rad) * self.xp.cos(lat2_rad) * self.xp.cos(dlon))
        
        bearing = self.xp.degrees(self.xp.arctan2(y, x))
        
        # 標準化到 0-360 度
        bearing = (bearing + 360) % 360
        
        return bearing

    def get_performance_stats(self) -> dict:
        """獲取效能統計"""
        avg_time_per_conversion = (
            self.total_conversion_time / self.conversion_count 
            if self.conversion_count > 0 else 0
        )
        
        return {
            "backend": "GPU" if is_gpu_enabled() else "CPU",
            "cuda_kernels": self.cuda_kernels_available,
            "high_precision": self.use_high_precision,
            "total_conversions": self.conversion_count,
            "total_time": self.total_conversion_time,
            "avg_time_per_conversion": avg_time_per_conversion,
            "conversions_per_second": self.conversion_count / (self.total_conversion_time + 1e-6),
            "reference_point": {
                "latitude": self.reference_point.latitude,
                "longitude": self.reference_point.longitude,
                "altitude": self.reference_point.altitude
            }
        }

    def auto_set_reference_from_waypoints(
        self, 
        latitudes: Union[List, np.ndarray], 
        longitudes: Union[List, np.ndarray]
    ):
        """從航點自動設置最優參考點"""
        if len(latitudes) == 0 or len(longitudes) == 0:
            return
        
        # 轉為numpy陣列進行計算
        lats = np.array(latitudes)
        lons = np.array(longitudes)
        
        # 使用航點的中心作為參考點
        center_lat = np.mean(lats)
        center_lon = np.mean(lons)
        
        # 設置參考點
        self.set_reference_point(center_lat, center_lon, 0.0)
        
        logger.info(f"🎯 自動設置參考點為航點中心: ({center_lat:.6f}, {center_lon:.6f})")

    def cleanup(self):
        """清理資源"""
        if is_gpu_enabled():
            from utils.gpu_utils import compute_manager
            if hasattr(compute_manager, '_cupy'):
                compute_manager._cupy.get_default_memory_pool().free_all_blocks()
        
        logger.info("🧹 座標轉換器資源已清理")

# 向後相容的別名
CoordinateConverter = GPUCoordinateConverter