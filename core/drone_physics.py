#!/usr/bin/env python3
"""
無人機物理系統和座標轉換
整合GPU/CPU加速功能，支援大規模無人機群模擬
"""

import numpy as np
import math
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 嘗試導入GPU加速模組
try:
    from utils.gpu_utils import compute_manager, asarray, to_cpu, synchronize
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    compute_manager = None

logger = logging.getLogger(__name__)

# 地球物理常數
EARTH_RADIUS_KM = 6371.0
METERS_PER_DEGREE_LAT = 111111.0
GRAVITY = 9.81  # m/s²

@dataclass
class DronePhysicsConfig:
    """無人機物理參數配置"""
    max_speed: float = 15.0          # 最大飛行速度 (m/s)
    max_acceleration: float = 3.0     # 最大加速度 (m/s²)
    max_climb_rate: float = 3.0      # 最大爬升速度 (m/s)
    max_turn_rate: float = 45.0      # 最大轉彎速度 (度/秒)
    cruise_speed: float = 8.0        # 巡航速度 (m/s)
    takeoff_speed: float = 2.0       # 起飛速度 (m/s)
    landing_speed: float = 1.0       # 降落速度 (m/s)
    wind_resistance: float = 0.1     # 風阻係數

@dataclass
class FlightConstraints:
    """飛行約束參數"""
    min_altitude: float = 0.5        # 最小飛行高度 (m)
    max_altitude: float = 120.0      # 最大飛行高度 (m)
    safety_margin: float = 2.0       # 安全邊界 (m)
    no_fly_zones: List[Dict] = None  # 禁飛區域

    def __post_init__(self):
        if self.no_fly_zones is None:
            self.no_fly_zones = []

class EarthCoordinateSystem:
    """地球座標系統 - 支援GPU加速的高精度轉換"""
    
    def __init__(self):
        self.origin_lat: Optional[float] = None
        self.origin_lon: Optional[float] = None
        self.origin_alt: Optional[float] = None
        self.use_gpu = GPU_AVAILABLE and compute_manager is not None
        
        logger.info(f"座標系統初始化 - GPU加速: {'啟用' if self.use_gpu else '停用'}")
        
    def set_origin(self, lat: float, lon: float, alt: float = 0.0):
        """設置座標原點"""
        self.origin_lat = lat
        self.origin_lon = lon
        self.origin_alt = alt
        
        logger.info(f"設置座標原點: 緯度={lat:.8f}°, 經度={lon:.8f}°, 高度={alt:.2f}m")
        
    def lat_lon_alt_to_meters(self, lat: float, lon: float, alt: float) -> Tuple[float, float, float]:
        """將GPS座標轉換為本地米制座標系統"""
        if self.origin_lat is None or self.origin_lon is None:
            raise ValueError("座標原點尚未設置")
            
        # 緯度轉換 (北向為Y正方向)
        y = (lat - self.origin_lat) * METERS_PER_DEGREE_LAT
        
        # 經度轉換 (東向為X正方向，考慮緯度修正)
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
        x = (lon - self.origin_lon) * meters_per_degree_lon
        
        # 高度轉換 (相對高度)
        z = alt - self.origin_alt
        
        return x, y, z
    
    def meters_to_lat_lon_alt(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        """將本地米制座標轉換為GPS座標"""
        if self.origin_lat is None or self.origin_lon is None:
            raise ValueError("座標原點尚未設置")
            
        # 緯度轉換
        lat = self.origin_lat + y / METERS_PER_DEGREE_LAT
        
        # 經度轉換
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
        lon = self.origin_lon + x / meters_per_degree_lon
        
        # 高度轉換
        alt = z + self.origin_alt
        
        return lat, lon, alt
    
    def batch_convert_to_meters(self, positions: np.ndarray) -> np.ndarray:
        """批次座標轉換 - GPU加速版本
        
        Args:
            positions: shape (N, 3) 的陣列，包含 [lat, lon, alt]
            
        Returns:
            shape (N, 3) 的陣列，包含 [x, y, z] 米制座標
        """
        if self.origin_lat is None or self.origin_lon is None:
            raise ValueError("座標原點尚未設置")
            
        if self.use_gpu and positions.shape[0] > 100:  # 大於100個點才使用GPU
            try:
                return self._gpu_batch_convert_to_meters(positions)
            except Exception as e:
                logger.warning(f"GPU座標轉換失敗，回退到CPU: {e}")
                self.use_gpu = False
        
        return self._cpu_batch_convert_to_meters(positions)
    
    def _gpu_batch_convert_to_meters(self, positions: np.ndarray) -> np.ndarray:
        """GPU加速批次座標轉換"""
        # 轉換為GPU陣列
        pos_gpu = asarray(positions.astype(np.float32))
        
        # 分離座標
        lats = pos_gpu[:, 0]
        lons = pos_gpu[:, 1]
        alts = pos_gpu[:, 2]
        
        # GPU計算
        y_gpu = (lats - self.origin_lat) * METERS_PER_DEGREE_LAT
        
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
        x_gpu = (lons - self.origin_lon) * meters_per_degree_lon
        
        z_gpu = alts - self.origin_alt
        
        # 組合結果
        result_gpu = compute_manager.xp.stack([x_gpu, y_gpu, z_gpu], axis=1)
        
        # 同步並轉回CPU
        synchronize()
        return to_cpu(result_gpu)
    
    def _cpu_batch_convert_to_meters(self, positions: np.ndarray) -> np.ndarray:
        """CPU批次座標轉換"""
        result = np.zeros_like(positions)
        
        # 緯度轉換 (Y軸)
        result[:, 1] = (positions[:, 0] - self.origin_lat) * METERS_PER_DEGREE_LAT
        
        # 經度轉換 (X軸)
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
        result[:, 0] = (positions[:, 1] - self.origin_lon) * meters_per_degree_lon
        
        # 高度轉換 (Z軸)
        result[:, 2] = positions[:, 2] - self.origin_alt
        
        return result

class DronePhysics:
    """無人機物理模型 - 支援真實飛行動力學"""
    
    def __init__(self, config: DronePhysicsConfig = None):
        self.config = config or DronePhysicsConfig()
        self.use_gpu = GPU_AVAILABLE and compute_manager is not None
        
    def calculate_flight_time(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                             speed: float = None) -> float:
        """計算飛行時間，考慮加速和減速"""
        if speed is None:
            speed = self.config.cruise_speed
            
        distance = np.linalg.norm(end_pos - start_pos)
        
        # 簡化的飛行時間計算（考慮加速階段）
        acceleration_time = speed / self.config.max_acceleration
        acceleration_distance = 0.5 * self.config.max_acceleration * (acceleration_time ** 2)
        
        if distance <= 2 * acceleration_distance:
            # 短距離：只有加速和減速階段
            return 2 * math.sqrt(distance / self.config.max_acceleration)
        else:
            # 長距離：加速 + 等速 + 減速
            cruise_distance = distance - 2 * acceleration_distance
            cruise_time = cruise_distance / speed
            return 2 * acceleration_time + cruise_time
    
    def calculate_realistic_trajectory(self, waypoints: List[Dict], drone_id: str = None) -> List[Dict]:
        """計算真實的飛行軌跡，考慮物理約束"""
        if len(waypoints) < 2:
            return []
            
        trajectory = []
        total_time = 0.0
        
        # 轉換航點為米制座標（假設已有座標系統）
        positions = []
        for wp in waypoints:
            if 'x' in wp and 'y' in wp and 'z' in wp:
                positions.append([wp['x'], wp['y'], wp['z']])
            else:
                # 如果只有GPS座標，需要外部座標系統轉換
                positions.append([0, 0, wp.get('alt', 0)])
        
        positions = np.array(positions)
        
        # 計算每段的飛行時間和中間點
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]
            
            # 計算這一段的飛行時間
            segment_time = self.calculate_flight_time(start_pos, end_pos)
            
            # 生成中間軌跡點（考慮加速度曲線）
            num_points = max(5, int(segment_time * 2))  # 每秒至少2個點
            
            for j in range(num_points):
                t_ratio = j / (num_points - 1)
                # 使用S曲線插值模擬真實的加速度變化
                smooth_ratio = self._smooth_step(t_ratio)
                
                interp_pos = start_pos + smooth_ratio * (end_pos - start_pos)
                point_time = total_time + t_ratio * segment_time
                
                trajectory.append({
                    'x': float(interp_pos[0]),
                    'y': float(interp_pos[1]), 
                    'z': float(interp_pos[2]),
                    'time': point_time,
                    'waypoint_index': i,
                    'speed': self._calculate_instantaneous_speed(t_ratio, segment_time),
                    'phase': 'cruise'
                })
            
            total_time += segment_time
        
        return trajectory
    
    def _smooth_step(self, t: float) -> float:
        """S曲線插值，模擬真實的加速度變化"""
        # 使用smoothstep函數：3t² - 2t³
        if t <= 0:
            return 0.0
        elif t >= 1:
            return 1.0
        else:
            return t * t * (3.0 - 2.0 * t)
    
    def _calculate_instantaneous_speed(self, t_ratio: float, segment_time: float) -> float:
        """計算瞬時速度"""
        # 簡化的速度計算
        if t_ratio < 0.1:  # 加速階段
            return self.config.cruise_speed * (t_ratio / 0.1)
        elif t_ratio > 0.9:  # 減速階段
            return self.config.cruise_speed * ((1.0 - t_ratio) / 0.1)
        else:  # 等速階段
            return self.config.cruise_speed
    
    def check_flight_constraints(self, position: np.ndarray, 
                               constraints: FlightConstraints) -> Dict[str, bool]:
        """檢查飛行約束"""
        checks = {
            'altitude_ok': constraints.min_altitude <= position[2] <= constraints.max_altitude,
            'in_bounds': True,  # 可以添加邊界檢查
            'no_fly_zone_clear': True  # 可以添加禁飛區檢查
        }
        
        # 檢查禁飛區域
        for no_fly_zone in constraints.no_fly_zones:
            if self._point_in_zone(position, no_fly_zone):
                checks['no_fly_zone_clear'] = False
                break
        
        return checks
    
    def _point_in_zone(self, point: np.ndarray, zone: Dict) -> bool:
        """檢查點是否在禁飛區域內"""
        # 簡單的圓形禁飛區檢查
        if zone.get('type') == 'circle':
            center = np.array([zone['x'], zone['y'], zone.get('z', 0)])
            radius = zone['radius']
            distance = np.linalg.norm(point[:2] - center[:2])  # 只檢查水平距離
            return distance <= radius
        
        return False

class TakeoffLandingManager:
    """起飛降落管理器"""
    
    def __init__(self, physics_config: DronePhysicsConfig = None):
        self.config = physics_config or DronePhysicsConfig()
        
    def generate_takeoff_sequence(self, start_pos: np.ndarray, target_altitude: float,
                                formation_offset: Tuple[float, float] = None) -> List[Dict]:
        """生成起飛序列"""
        sequence = []
        
        # 如果有編隊偏移，調整起飛位置
        if formation_offset:
            takeoff_x = start_pos[0] + formation_offset[0]
            takeoff_y = start_pos[1] + formation_offset[1]
        else:
            takeoff_x, takeoff_y = start_pos[0], start_pos[1]
        
        takeoff_z = start_pos[2]
        
        # 階段1: 地面準備 (0-2秒)
        sequence.append({
            'x': takeoff_x, 'y': takeoff_y, 'z': takeoff_z,
            'time': 0.0, 'phase': 'taxi',
            'speed': 0.0
        })
        
        sequence.append({
            'x': takeoff_x, 'y': takeoff_y, 'z': takeoff_z,
            'time': 2.0, 'phase': 'taxi',
            'speed': 0.0
        })
        
        # 階段2: 垂直起飛 (2-7秒)
        climb_time = target_altitude / self.config.takeoff_speed
        climb_end_time = 2.0 + climb_time
        
        # 生成爬升軌跡點
        num_climb_points = int(climb_time * 2)  # 每秒2個點
        for i in range(1, num_climb_points + 1):
            t_ratio = i / num_climb_points
            current_time = 2.0 + t_ratio * climb_time
            current_altitude = takeoff_z + t_ratio * target_altitude
            
            sequence.append({
                'x': takeoff_x, 'y': takeoff_y, 'z': current_altitude,
                'time': current_time, 'phase': 'takeoff',
                'speed': self.config.takeoff_speed
            })
        
        # 階段3: 懸停穩定 (2秒)
        hover_end_time = climb_end_time + 2.0
        sequence.append({
            'x': takeoff_x, 'y': takeoff_y, 'z': takeoff_z + target_altitude,
            'time': hover_end_time, 'phase': 'hover',
            'speed': 0.0
        })
        
        return sequence
    
    def generate_landing_sequence(self, approach_pos: np.ndarray, 
                                landing_pos: np.ndarray) -> List[Dict]:
        """生成降落序列"""
        sequence = []
        
        # 階段1: 進場下降
        approach_time = 10.0  # 進場時間
        descent_rate = (approach_pos[2] - landing_pos[2]) / approach_time
        
        num_approach_points = int(approach_time)
        for i in range(num_approach_points + 1):
            t_ratio = i / num_approach_points
            current_time = t_ratio * approach_time
            
            current_pos = approach_pos + t_ratio * (landing_pos - approach_pos)
            
            sequence.append({
                'x': float(current_pos[0]),
                'y': float(current_pos[1]),
                'z': float(current_pos[2]),
                'time': current_time, 
                'phase': 'landing',
                'speed': self.config.landing_speed
            })
        
        return sequence

# 工廠函數
def create_coordinate_system(origin_lat: float, origin_lon: float, 
                           origin_alt: float = 0.0) -> EarthCoordinateSystem:
    """創建並初始化座標系統"""
    coord_system = EarthCoordinateSystem()
    coord_system.set_origin(origin_lat, origin_lon, origin_alt)
    return coord_system

def create_physics_engine(max_speed: float = 15.0, 
                         cruise_speed: float = 8.0) -> DronePhysics:
    """創建物理引擎"""
    config = DronePhysicsConfig(max_speed=max_speed, cruise_speed=cruise_speed)
    return DronePhysics(config)