"""
地球坐標系統模組 - GPU加速版本
支援地理坐標與笛卡爾坐標的高效能轉換
"""

import math
import numpy as np
from typing import Tuple, Optional, List, Union
from dataclasses import dataclass

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np  # Fallback to NumPy

from utils.gpu_utils import get_array_module, ensure_gpu_compatibility

# 地球常數
EARTH_RADIUS_KM = 6371.0
METERS_PER_DEGREE_LAT = 111111.0

@dataclass
class CoordinateTransform:
    """坐標轉換參數"""
    origin_lat: float
    origin_lon: float
    meters_per_degree_lon: float


class EarthCoordinateSystem:
    """
    地球坐標系統 - GPU加速版本
    
    提供地理坐標(緯度/經度)與笛卡爾坐標(米)之間的高效能轉換
    支援批量處理和GPU加速計算
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        初始化地球坐標系統
        
        Args:
            use_gpu: 是否使用GPU加速
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        
        # 坐標系參數
        self.origin_lat: Optional[float] = None
        self.origin_lon: Optional[float] = None
        self.meters_per_degree_lon: Optional[float] = None
        self.transform_params: Optional[CoordinateTransform] = None
        
    def set_origin(self, lat: float, lon: float) -> None:
        """
        設置坐標原點
        
        Args:
            lat: 原點緯度
            lon: 原點經度
        """
        self.origin_lat = lat
        self.origin_lon = lon
        
        # 計算經度轉換係數（考慮緯度修正）
        self.meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(lat))
        
        # 創建轉換參數對象（便於GPU傳遞）
        self.transform_params = CoordinateTransform(
            origin_lat=lat,
            origin_lon=lon,
            meters_per_degree_lon=self.meters_per_degree_lon
        )
        
        print(f"坐標原點設置: ({lat:.6f}, {lon:.6f})")
        print(f"經度轉換係數: {self.meters_per_degree_lon:.2f} m/degree")
    
    @ensure_gpu_compatibility
    def lat_lon_to_meters(self, lat: Union[float, np.ndarray], 
                         lon: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], 
                                                                Union[float, np.ndarray]]:
        """
        將地理坐標轉換為米制坐標（支援批量處理）
        
        Args:
            lat: 緯度（度）- 可以是單個值或數組
            lon: 經度（度）- 可以是單個值或數組
            
        Returns:
            (x, y): 東向和北向距離（米）
        """
        if self.transform_params is None:
            return 0.0, 0.0
        
        # 確保輸入是正確的數組類型
        lat_array = self.xp.asarray(lat)
        lon_array = self.xp.asarray(lon)
        
        # 批量計算坐標轉換
        y = (lat_array - self.transform_params.origin_lat) * METERS_PER_DEGREE_LAT
        x = (lon_array - self.transform_params.origin_lon) * self.transform_params.meters_per_degree_lon
        
        # 如果輸入是標量，返回標量
        if lat_array.ndim == 0 and lon_array.ndim == 0:
            return float(x), float(y)
        
        return x, y
    
    @ensure_gpu_compatibility  
    def meters_to_lat_lon(self, x: Union[float, np.ndarray], 
                         y: Union[float, np.ndarray]) -> Tuple[Union[float, np.ndarray], 
                                                              Union[float, np.ndarray]]:
        """
        將米制坐標轉換為地理坐標（支援批量處理）
        
        Args:
            x: 東向距離（米）- 可以是單個值或數組
            y: 北向距離（米）- 可以是單個值或數組
            
        Returns:
            (lat, lon): 緯度和經度（度）
        """
        if self.transform_params is None:
            return 0.0, 0.0
        
        # 確保輸入是正確的數組類型
        x_array = self.xp.asarray(x)
        y_array = self.xp.asarray(y)
        
        # 批量計算坐標轉換
        lat = self.transform_params.origin_lat + y_array / METERS_PER_DEGREE_LAT
        lon = self.transform_params.origin_lon + x_array / self.transform_params.meters_per_degree_lon
        
        # 如果輸入是標量，返回標量
        if x_array.ndim == 0 and y_array.ndim == 0:
            return float(lat), float(lon)
        
        return lat, lon
    
    @ensure_gpu_compatibility
    def calculate_distance_haversine(self, lat1: Union[float, np.ndarray], lon1: Union[float, np.ndarray],
                                   lat2: Union[float, np.ndarray], lon2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        使用Haversine公式計算地球表面兩點間距離（支援批量處理）
        
        Args:
            lat1, lon1: 第一個點的緯度和經度（度）
            lat2, lon2: 第二個點的緯度和經度（度）
            
        Returns:
            距離（米）
        """
        # 轉換為弧度
        lat1_rad = self.xp.radians(self.xp.asarray(lat1))
        lon1_rad = self.xp.radians(self.xp.asarray(lon1))
        lat2_rad = self.xp.radians(self.xp.asarray(lat2))
        lon2_rad = self.xp.radians(self.xp.asarray(lon2))
        
        # Haversine公式
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (self.xp.sin(dlat/2)**2 + 
             self.xp.cos(lat1_rad) * self.xp.cos(lat2_rad) * self.xp.sin(dlon/2)**2)
        
        c = 2 * self.xp.arcsin(self.xp.sqrt(a))
        distance = EARTH_RADIUS_KM * 1000 * c  # 轉換為米
        
        return distance
    
    @ensure_gpu_compatibility
    def calculate_distance_3d(self, pos1: Union[np.ndarray, List], 
                            pos2: Union[np.ndarray, List]) -> Union[float, np.ndarray]:
        """
        計算3D空間中兩點間的歐幾里德距離（支援批量處理）
        
        Args:
            pos1: 第一個點的坐標 [x, y, z]
            pos2: 第二個點的坐標 [x, y, z]
            
        Returns:
            3D距離（米）
        """
        pos1_array = self.xp.asarray(pos1)
        pos2_array = self.xp.asarray(pos2)
        
        # 計算各軸向差值
        diff = pos2_array - pos1_array
        
        # 計算歐幾里德距離
        if pos1_array.ndim == 1:  # 單個點對
            distance = self.xp.sqrt(self.xp.sum(diff**2))
        else:  # 批量處理
            distance = self.xp.sqrt(self.xp.sum(diff**2, axis=-1))
        
        return distance
    
    def batch_coordinate_transform(self, coordinates: List[Tuple[float, float]], 
                                 to_meters: bool = True) -> List[Tuple[float, float]]:
        """
        批量坐標轉換（高效能版本）
        
        Args:
            coordinates: 坐標列表 [(lat/x, lon/y), ...]
            to_meters: True表示轉為米制，False表示轉為地理坐標
            
        Returns:
            轉換後的坐標列表
        """
        if not coordinates:
            return []
        
        # 分離坐標為兩個數組
        coords_array = self.xp.array(coordinates)
        first_coords = coords_array[:, 0]
        second_coords = coords_array[:, 1]
        
        # 批量轉換
        if to_meters:
            x_coords, y_coords = self.lat_lon_to_meters(first_coords, second_coords)
        else:
            x_coords, y_coords = self.meters_to_lat_lon(first_coords, second_coords)
        
        # 重新組合結果
        if self.use_gpu and hasattr(x_coords, 'get'):
            x_coords = x_coords.get()
            y_coords = y_coords.get()
        
        return list(zip(x_coords.tolist(), y_coords.tolist()))
    
    def create_formation_pattern(self, center_lat: float, center_lon: float, 
                               formation_type: str = "diamond", 
                               spacing: float = 10.0, 
                               num_drones: int = 4) -> List[Tuple[float, float]]:
        """
        創建編隊模式的位置點
        
        Args:
            center_lat: 編隊中心緯度
            center_lon: 編隊中心經度
            formation_type: 編隊類型 ("diamond", "square", "line", "circle")
            spacing: 間距（米）
            num_drones: 無人機數量
            
        Returns:
            編隊位置列表 [(lat, lon), ...]
        """
        # 轉換中心點為米制坐標
        center_x, center_y = self.lat_lon_to_meters(center_lat, center_lon)
        
        positions = []
        
        if formation_type == "square" and num_drones == 4:
            # 2x2方陣
            offsets = [
                (-spacing/2, -spacing/2),  # 左下
                (spacing/2, -spacing/2),   # 右下  
                (-spacing/2, spacing/2),   # 左上
                (spacing/2, spacing/2)     # 右上
            ]
        elif formation_type == "diamond" and num_drones == 4:
            # 菱形編隊
            offsets = [
                (0, -spacing),      # 南
                (spacing, 0),       # 東
                (0, spacing),       # 北
                (-spacing, 0)       # 西
            ]
        elif formation_type == "line":
            # 一字排開
            offsets = [(i * spacing - (num_drones-1) * spacing/2, 0) 
                      for i in range(num_drones)]
        elif formation_type == "circle":
            # 圓形編隊
            angles = self.xp.linspace(0, 2*math.pi, num_drones, endpoint=False)
            offsets = [(spacing * self.xp.cos(angle), spacing * self.xp.sin(angle)) 
                      for angle in angles]
        else:
            # 默認方陣
            rows = int(math.sqrt(num_drones))
            cols = math.ceil(num_drones / rows)
            offsets = []
            for i in range(num_drones):
                row = i // cols
                col = i % cols
                x_offset = (col - (cols-1)/2) * spacing
                y_offset = (row - (rows-1)/2) * spacing
                offsets.append((x_offset, y_offset))
        
        # 轉換回地理坐標
        for x_offset, y_offset in offsets:
            pos_x = center_x + x_offset
            pos_y = center_y + y_offset
            lat, lon = self.meters_to_lat_lon(pos_x, pos_y)
            positions.append((lat, lon))
        
        return positions
    
    def validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        驗證地理坐標的合法性
        
        Args:
            lat: 緯度
            lon: 經度
            
        Returns:
            坐標是否合法
        """
        return (-90 <= lat <= 90) and (-180 <= lon <= 180)
    
    def get_coordinate_bounds(self, positions: List[Tuple[float, float]]) -> dict:
        """
        計算坐標列表的邊界
        
        Args:
            positions: 坐標位置列表
            
        Returns:
            邊界信息字典
        """
        if not positions:
            return {}
        
        lats, lons = zip(*positions)
        
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons), 
            'max_lon': max(lons),
            'center_lat': (min(lats) + max(lats)) / 2,
            'center_lon': (min(lons) + max(lons)) / 2,
            'span_lat': max(lats) - min(lats),
            'span_lon': max(lons) - min(lons)
        }
    
    def optimize_memory_usage(self):
        """優化GPU記憶體使用"""
        if self.use_gpu:
            try:
                # 清理GPU記憶體
                cp.get_default_memory_pool().free_all_blocks()
                print("GPU記憶體已優化")
            except:
                pass
    
    def get_system_info(self) -> dict:
        """獲取坐標系統狀態信息"""
        return {
            'origin_set': self.origin_lat is not None,
            'origin_lat': self.origin_lat,
            'origin_lon': self.origin_lon,
            'gpu_enabled': self.use_gpu,
            'gpu_available': GPU_AVAILABLE,
            'meters_per_degree_lon': self.meters_per_degree_lon
        }


# 便利函數
def create_coordinate_system(origin_lat: float = None, origin_lon: float = None, 
                           use_gpu: bool = True) -> EarthCoordinateSystem:
    """
    快速創建坐標系統實例
    
    Args:
        origin_lat: 原點緯度  
        origin_lon: 原點經度
        use_gpu: 是否使用GPU
        
    Returns:
        配置好的坐標系統實例
    """
    coord_system = EarthCoordinateSystem(use_gpu=use_gpu)
    
    if origin_lat is not None and origin_lon is not None:
        coord_system.set_origin(origin_lat, origin_lon)
    
    return coord_system