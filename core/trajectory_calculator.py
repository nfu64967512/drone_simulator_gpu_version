"""
GPU加速軌跡計算模組
提供高效能的無人機飛行軌跡計算、插值和優化功能
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

from utils.gpu_utils import get_array_module, ensure_gpu_compatibility
from config.settings import FlightConfig
from core.coordinate_system import EarthCoordinateSystem
logger = logging.getLogger(__name__)

class FlightPhase(Enum):
    """飛行階段"""
    TAXI = "taxi"           # 地面滑行
    TAKEOFF = "takeoff"     # 起飛爬升
    HOVER = "hover"         # 懸停等待
    CRUISE = "cruise"       # 巡航飛行
    AUTO = "auto"           # 自動任務
    LOITER = "loiter"       # 等待避讓
    LANDING = "landing"     # 降落
    RTL = "rtl"            # 返航

@dataclass
class TrajectoryPoint:
    """軌跡點數據結構"""
    x: float                    # 東向距離 (m)
    y: float                    # 北向距離 (m)  
    z: float                    # 高度 (m)
    time: float                 # 時間戳 (s)
    phase: FlightPhase         # 飛行階段
    latitude: float            # 緯度
    longitude: float           # 經度
    altitude: float            # 高度 (m)
    waypoint_index: Optional[int] = None  # 對應航點索引
    velocity: Optional[float] = None      # 速度 (m/s)
    heading: Optional[float] = None       # 航向角 (度)

@dataclass 
class FlightParameters:
    """飛行參數配置"""
    cruise_speed: float = 8.0          # 巡航速度 (m/s)
    climb_rate: float = 2.0            # 爬升率 (m/s)
    descent_rate: float = 1.5          # 下降率 (m/s)
    takeoff_altitude: float = 10.0     # 起飛高度 (m)
    hover_time: float = 2.0            # 懸停時間 (s)
    acceleration: float = 2.0          # 加速度 (m/s²)
    turn_radius: float = 5.0           # 轉彎半徑 (m)


class GPUTrajectoryCalculator:
    """
    GPU加速軌跡計算器
    使用並行計算生成平滑、真實的飛行軌跡
    """
    
    def __init__(self, coordinate_system: EarthCoordinateSystem, 
                 flight_params: FlightParameters, use_gpu: bool = True):
        """
        初始化軌跡計算器
        
        Args:
            coordinate_system: 坐標系統實例
            flight_params: 飛行參數
            use_gpu: 是否使用GPU加速
        """
        self.coordinate_system = coordinate_system
        self.flight_params = flight_params
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        
        logger.info(f"軌跡計算器初始化: GPU={'啟用' if self.use_gpu else '禁用'}")
    
    def calculate_complete_trajectory(self, waypoints: List[Dict], 
                                    drone_id: str) -> List[TrajectoryPoint]:
        """
        計算完整的無人機飛行軌跡
        
        Args:
            waypoints: 航點列表 [{'lat': float, 'lon': float, 'alt': float, 'cmd': int}, ...]
            drone_id: 無人機識別碼
            
        Returns:
            完整軌跡點列表
        """
        if not waypoints:
            logger.warning(f"{drone_id}: 沒有航點數據")
            return []
        
        logger.info(f"{drone_id}: 開始計算軌跡，包含 {len(waypoints)} 個航點")
        
        trajectory = []
        total_time = 0.0
        
        # 階段1：地面滑行 (0-2秒)
        taxi_points = self._generate_taxi_phase(waypoints[0], total_time)
        trajectory.extend(taxi_points)
        total_time = taxi_points[-1].time if taxi_points else 0.0
        
        # 階段2：起飛爬升 (2-7秒)
        takeoff_points = self._generate_takeoff_phase(waypoints[0], total_time)
        trajectory.extend(takeoff_points)
        total_time = takeoff_points[-1].time if takeoff_points else total_time
        
        # 階段3：起飛後懸停 (7-9秒)
        hover_points = self._generate_hover_phase(waypoints[0], total_time)
        trajectory.extend(hover_points)
        total_time = hover_points[-1].time if hover_points else total_time
        
        # 階段4：自動任務執行
        if len(waypoints) > 1:
            mission_points = self._generate_mission_trajectory(
                waypoints, total_time, drone_id
            )
            trajectory.extend(mission_points)
        
        logger.info(f"{drone_id}: 軌跡計算完成，總時長 {trajectory[-1].time:.1f}s，"
                   f"包含 {len(trajectory)} 個軌跡點")
        
        return trajectory
    
    def _generate_taxi_phase(self, home_waypoint: Dict, start_time: float) -> List[TrajectoryPoint]:
        """
        生成地面滑行階段軌跡
        
        Args:
            home_waypoint: HOME航點
            start_time: 開始時間
            
        Returns:
            滑行軌跡點列表
        """
        home_x, home_y = self.coordinate_system.lat_lon_to_meters(
            home_waypoint['lat'], home_waypoint['lon']
        )
        
        # 地面滑行2秒
        taxi_duration = 2.0
        
        point = TrajectoryPoint(
            x=home_x, y=home_y, z=0.0,
            time=start_time,
            phase=FlightPhase.TAXI,
            latitude=home_waypoint['lat'],
            longitude=home_waypoint['lon'],
            altitude=0.0,
            waypoint_index=0,
            velocity=0.0,
            heading=0.0
        )
        
        return [point]
    
    def _generate_takeoff_phase(self, home_waypoint: Dict, start_time: float) -> List[TrajectoryPoint]:
        """
        生成起飛爬升階段軌跡
        
        Args:
            home_waypoint: HOME航點  
            start_time: 開始時間
            
        Returns:
            起飛軌跡點列表
        """
        home_x, home_y = self.coordinate_system.lat_lon_to_meters(
            home_waypoint['lat'], home_waypoint['lon']
        )
        
        takeoff_duration = 5.0  # 起飛爬升5秒
        num_points = 20
        
        points = []
        for i in range(num_points):
            progress = i / (num_points - 1)
            time = start_time + takeoff_duration * progress
            altitude = progress * self.flight_params.takeoff_altitude
            
            # 使用平滑的爬升曲線
            smooth_progress = self._smooth_curve(progress)
            smooth_altitude = smooth_progress * self.flight_params.takeoff_altitude
            
            point = TrajectoryPoint(
                x=home_x, y=home_y, z=smooth_altitude,
                time=time,
                phase=FlightPhase.TAKEOFF,
                latitude=home_waypoint['lat'],
                longitude=home_waypoint['lon'], 
                altitude=smooth_altitude,
                waypoint_index=0 if i < 10 else 1,  # 漸進過渡到航點1
                velocity=self.flight_params.climb_rate,
                heading=0.0
            )
            points.append(point)
        
        return points
    
    def _generate_hover_phase(self, home_waypoint: Dict, start_time: float) -> List[TrajectoryPoint]:
        """
        生成懸停等待階段軌跡
        
        Args:
            home_waypoint: HOME航點
            start_time: 開始時間
            
        Returns:
            懸停軌跡點列表
        """
        home_x, home_y = self.coordinate_system.lat_lon_to_meters(
            home_waypoint['lat'], home_waypoint['lon']
        )
        
        hover_end_time = start_time + self.flight_params.hover_time
        
        point = TrajectoryPoint(
            x=home_x, y=home_y, z=self.flight_params.takeoff_altitude,
            time=hover_end_time,
            phase=FlightPhase.HOVER,
            latitude=home_waypoint['lat'],
            longitude=home_waypoint['lon'],
            altitude=self.flight_params.takeoff_altitude,
            waypoint_index=1,
            velocity=0.0,
            heading=0.0
        )
        
        return [point]
    
    @ensure_gpu_compatibility
    def _generate_mission_trajectory(self, waypoints: List[Dict], start_time: float, 
                                   drone_id: str) -> List[TrajectoryPoint]:
        """
        生成任務執行階段軌跡（GPU加速）
        
        Args:
            waypoints: 航點列表
            start_time: 開始時間
            drone_id: 無人機ID
            
        Returns:
            任務軌跡點列表
        """
        points = []
        current_time = start_time
        
        # 當前位置（起飛點）
        home_wp = waypoints[0]
        prev_x, prev_y = self.coordinate_system.lat_lon_to_meters(
            home_wp['lat'], home_wp['lon']
        )
        prev_z = self.flight_params.takeoff_altitude
        
        # 處理任務航點（從waypoints[1:]開始）
        for wp_idx, wp in enumerate(waypoints[1:], start=2):
            # 轉換航點坐標
            target_x, target_y = self.coordinate_system.lat_lon_to_meters(
                wp['lat'], wp['lon']
            )
            target_z = wp.get('alt', 15.0)
            
            # 計算飛行距離和時間
            distance_3d = np.sqrt((target_x - prev_x)**2 + 
                                (target_y - prev_y)**2 + 
                                (target_z - prev_z)**2)
            
            flight_time = distance_3d / self.flight_params.cruise_speed
            
            # 生成航段軌跡點
            segment_points = self._generate_flight_segment(
                (prev_x, prev_y, prev_z), 
                (target_x, target_y, target_z),
                current_time, flight_time, wp_idx
            )
            
            points.extend(segment_points)
            current_time += flight_time
            
            # 添加航點到達點
            arrival_point = TrajectoryPoint(
                x=target_x, y=target_y, z=target_z,
                time=current_time,
                phase=FlightPhase.AUTO,
                latitude=wp['lat'], longitude=wp['lon'], altitude=target_z,
                waypoint_index=wp_idx,
                velocity=self.flight_params.cruise_speed,
                heading=self._calculate_heading(prev_x, prev_y, target_x, target_y)
            )
            points.append(arrival_point)
            
            # 更新當前位置
            prev_x, prev_y, prev_z = target_x, target_y, target_z
            
            logger.debug(f"{drone_id}: 航點 {wp_idx} 軌跡計算完成，"
                        f"距離 {distance_3d:.1f}m，飛行時間 {flight_time:.1f}s")
        
        return points
    
    @ensure_gpu_compatibility
    def _generate_flight_segment(self, start_pos: Tuple[float, float, float], 
                               end_pos: Tuple[float, float, float],
                               start_time: float, duration: float, 
                               waypoint_idx: int) -> List[TrajectoryPoint]:
        """
        生成單個飛行航段的詳細軌跡（GPU加速）
        
        Args:
            start_pos: 起始位置 (x, y, z)
            end_pos: 結束位置 (x, y, z)  
            start_time: 開始時間
            duration: 飛行時長
            waypoint_idx: 目標航點索引
            
        Returns:
            航段軌跡點列表
        """
        if duration <= 0:
            return []
        
        # 計算插值點數量（基於距離，每10米至少一個點）
        distance = np.sqrt(sum((end_pos[i] - start_pos[i])**2 for i in range(3)))
        num_points = max(2, int(distance / 10))
        
        # 使用GPU並行計算插值點
        t_values = self.xp.linspace(0, 1, num_points, endpoint=False)
        
        # 批量計算位置
        start_array = self.xp.array(start_pos)
        end_array = self.xp.array(end_pos)
        
        positions = start_array[None, :] + t_values[:, None] * (end_array - start_array)[None, :]
        times = start_time + t_values * duration
        
        # 轉換回CPU數據（如果使用GPU）
        if self.use_gpu and hasattr(positions, 'get'):
            positions = positions.get()
            times = times.get()
        
        points = []
        heading = self._calculate_heading(start_pos[0], start_pos[1], end_pos[0], end_pos[1])
        
        for i, (pos, time) in enumerate(zip(positions, times)):
            lat, lon = self.coordinate_system.meters_to_lat_lon(pos[0], pos[1])
            
            point = TrajectoryPoint(
                x=pos[0], y=pos[1], z=pos[2],
                time=time,
                phase=FlightPhase.AUTO,
                latitude=lat, longitude=lon, altitude=pos[2],
                waypoint_index=waypoint_idx - 1,  # 指向前一個航點
                velocity=self.flight_params.cruise_speed,
                heading=heading
            )
            points.append(point)
        
        return points
    
    def _smooth_curve(self, t: float) -> float:
        """
        生成平滑曲線（S型）
        
        Args:
            t: 參數 [0, 1]
            
        Returns:
            平滑化後的值
        """
        # 使用三次平滑函數
        return t * t * (3 - 2 * t)
    
    def _calculate_heading(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        計算兩點間的航向角
        
        Args:
            x1, y1: 起點坐標
            x2, y2: 終點坐標
            
        Returns:
            航向角（度，北為0度，順時針）
        """
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return 0.0
        
        # 計算角度（弧度）
        angle_rad = np.arctan2(dx, dy)  # 注意：x對應東向，y對應北向
        
        # 轉換為度並確保在[0, 360)範圍內
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        return angle_deg


class TrajectoryInterpolator:
    """
    軌跡插值器 - GPU加速版本  
    提供高效能的軌跡插值和查詢功能
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        初始化插值器
        
        Args:
            use_gpu: 是否使用GPU加速
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
    
    @ensure_gpu_compatibility
    def interpolate_position(self, trajectory: List[TrajectoryPoint], 
                           query_time: float) -> Optional[TrajectoryPoint]:
        """
        在軌跡中插值計算指定時間的位置（GPU加速）
        
        Args:
            trajectory: 軌跡點列表
            query_time: 查詢時間
            
        Returns:
            插值得到的軌跡點，如果無法插值則返回None
        """
        if not trajectory:
            return None
        
        # 邊界條件處理
        if query_time >= trajectory[-1].time:
            return trajectory[-1]
        if query_time <= trajectory[0].time:
            return trajectory[0]
        
        # 找到插值區間
        for i in range(len(trajectory) - 1):
            t1, t2 = trajectory[i].time, trajectory[i + 1].time
            
            if t1 <= query_time <= t2:
                if t2 - t1 == 0:
                    return trajectory[i]
                
                # 線性插值比例
                ratio = (query_time - t1) / (t2 - t1)
                
                # 插值計算各個屬性
                p1, p2 = trajectory[i], trajectory[i + 1]
                
                interpolated_point = TrajectoryPoint(
                    x=p1.x + ratio * (p2.x - p1.x),
                    y=p1.y + ratio * (p2.y - p1.y),
                    z=p1.z + ratio * (p2.z - p1.z),
                    time=query_time,
                    phase=p1.phase,  # 使用前一點的飛行階段
                    latitude=p1.latitude + ratio * (p2.latitude - p1.latitude),
                    longitude=p1.longitude + ratio * (p2.longitude - p1.longitude),
                    altitude=p1.altitude + ratio * (p2.altitude - p1.altitude),
                    waypoint_index=p1.waypoint_index,
                    velocity=p1.velocity,
                    heading=p1.heading
                )
                
                return interpolated_point
        
        return None
    
    @ensure_gpu_compatibility
    def batch_interpolate_positions(self, trajectory: List[TrajectoryPoint], 
                                  query_times: Union[List[float], np.ndarray]) -> List[Optional[TrajectoryPoint]]:
        """
        批量插值多個時間點的位置（GPU加速）
        
        Args:
            trajectory: 軌跡點列表
            query_times: 查詢時間列表或數組
            
        Returns:
            插值結果列表
        """
        if not trajectory or not query_times:
            return []
        
        results = []
        query_times = self.xp.asarray(query_times)
        
        # 批量處理查詢
        for query_time in query_times:
            result = self.interpolate_position(trajectory, float(query_time))
            results.append(result)
        
        return results
    
    def get_trajectory_segment(self, trajectory: List[TrajectoryPoint], 
                             start_time: float, end_time: float) -> List[TrajectoryPoint]:
        """
        獲取指定時間範圍內的軌跡段
        
        Args:
            trajectory: 完整軌跡
            start_time: 開始時間
            end_time: 結束時間
            
        Returns:
            軌跡段
        """
        if not trajectory:
            return []
        
        segment = []
        
        for point in trajectory:
            if start_time <= point.time <= end_time:
                segment.append(point)
            elif point.time > end_time:
                break
        
        # 添加邊界插值點
        if segment:
            # 添加開始時間的插值點
            if segment[0].time > start_time:
                start_point = self.interpolate_position(trajectory, start_time)
                if start_point:
                    segment.insert(0, start_point)
            
            # 添加結束時間的插值點  
            if segment[-1].time < end_time:
                end_point = self.interpolate_position(trajectory, end_time)
                if end_point:
                    segment.append(end_point)
        
        return segment


class TrajectoryOptimizer:
    """
    軌跡優化器
    提供軌跡平滑、速度優化等高級功能
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        初始化優化器
        
        Args:
            use_gpu: 是否使用GPU加速
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
    
    @ensure_gpu_compatibility
    def smooth_trajectory(self, trajectory: List[TrajectoryPoint], 
                         smoothing_factor: float = 0.1) -> List[TrajectoryPoint]:
        """
        平滑軌跡（GPU加速）
        
        Args:
            trajectory: 原始軌跡
            smoothing_factor: 平滑因子 [0, 1]，越大越平滑
            
        Returns:
            平滑後的軌跡
        """
        if len(trajectory) < 3:
            return trajectory
        
        # 提取位置數據
        positions = self.xp.array([[p.x, p.y, p.z] for p in trajectory])
        
        # 應用移動平均平滑
        smoothed_positions = positions.copy()
        
        for i in range(1, len(positions) - 1):
            # 三點移動平均
            neighbor_avg = (positions[i-1] + positions[i+1]) / 2
            smoothed_positions[i] = (1 - smoothing_factor) * positions[i] + smoothing_factor * neighbor_avg
        
        # 轉換回CPU（如果使用GPU）
        if self.use_gpu and hasattr(smoothed_positions, 'get'):
            smoothed_positions = smoothed_positions.get()
        
        # 更新軌跡點位置
        smoothed_trajectory = []
        for i, point in enumerate(trajectory):
            new_point = TrajectoryPoint(
                x=smoothed_positions[i, 0],
                y=smoothed_positions[i, 1], 
                z=smoothed_positions[i, 2],
                time=point.time,
                phase=point.phase,
                latitude=point.latitude,  # 需要重新計算
                longitude=point.longitude,  # 需要重新計算
                altitude=smoothed_positions[i, 2],
                waypoint_index=point.waypoint_index,
                velocity=point.velocity,
                heading=point.heading
            )
            smoothed_trajectory.append(new_point)
        
        logger.info(f"軌跡平滑完成，平滑因子: {smoothing_factor}")
        return smoothed_trajectory
    
    def optimize_speed_profile(self, trajectory: List[TrajectoryPoint], 
                             max_acceleration: float = 2.0) -> List[TrajectoryPoint]:
        """
        優化速度剖面，確保加速度限制
        
        Args:
            trajectory: 原始軌跡
            max_acceleration: 最大加速度 (m/s²)
            
        Returns:
            優化後的軌跡
        """
        if len(trajectory) < 2:
            return trajectory
        
        optimized_trajectory = trajectory.copy()
        
        for i in range(1, len(optimized_trajectory)):
            prev_point = optimized_trajectory[i-1]
            current_point = optimized_trajectory[i]
            
            dt = current_point.time - prev_point.time
            if dt <= 0:
                continue
            
            # 計算當前速度變化
            dv = current_point.velocity - prev_point.velocity if prev_point.velocity else 0
            acceleration = dv / dt
            
            # 限制加速度
            if abs(acceleration) > max_acceleration:
                max_dv = max_acceleration * dt * (1 if acceleration > 0 else -1)
                new_velocity = prev_point.velocity + max_dv
                
                # 更新速度
                optimized_trajectory[i] = TrajectoryPoint(
                    x=current_point.x, y=current_point.y, z=current_point.z,
                    time=current_point.time,
                    phase=current_point.phase,
                    latitude=current_point.latitude,
                    longitude=current_point.longitude,
                    altitude=current_point.altitude,
                    waypoint_index=current_point.waypoint_index,
                    velocity=new_velocity,
                    heading=current_point.heading
                )
        
        logger.info(f"速度剖面優化完成，最大加速度: {max_acceleration} m/s²")
        return optimized_trajectory


# 便利函數
def create_trajectory_calculator(coordinate_system: EarthCoordinateSystem,
                               cruise_speed: float = 8.0,
                               takeoff_altitude: float = 10.0,
                               use_gpu: bool = True) -> GPUTrajectoryCalculator:
    """
    快速創建軌跡計算器
    
    Args:
        coordinate_system: 坐標系統
        cruise_speed: 巡航速度
        takeoff_altitude: 起飛高度
        use_gpu: 是否使用GPU
        
    Returns:
        配置好的軌跡計算器
    """
    flight_params = FlightParameters(
        cruise_speed=cruise_speed,
        takeoff_altitude=takeoff_altitude
    )
    
    return GPUTrajectoryCalculator(coordinate_system, flight_params, use_gpu)
