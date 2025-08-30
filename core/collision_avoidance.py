"""
GPU加速碰撞檢測系統
提供高效能的多無人機碰撞檢測和軌跡衝突分析
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

from utils.gpu_utils import get_array_module, ensure_gpu_compatibility
from config.settings import SafetyConfig

logger = logging.getLogger(__name__)

class ConflictSeverity(Enum):
    """衝突嚴重程度"""
    SAFE = "safe"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class CollisionWarning:
    """碰撞警告數據結構"""
    drone1_id: str
    drone2_id: str
    distance: float
    time: float
    position: Tuple[float, float, float]
    severity: ConflictSeverity
    relative_velocity: float
    estimated_collision_time: Optional[float] = None

@dataclass
class TrajectoryConflict:
    """軌跡衝突數據結構"""
    drone1_id: str
    drone2_id: str
    conflict_time: float
    conflict_position1: Tuple[float, float, float]
    conflict_position2: Tuple[float, float, float]
    minimum_distance: float
    severity: ConflictSeverity
    priority_drone: str
    waiting_drone: str
    recommended_wait_time: float
    waypoint1_index: int
    waypoint2_index: int


class GPUCollisionDetector:
    """
    GPU加速碰撞檢測器
    使用並行計算檢測多無人機間的潛在碰撞
    """
    
    def __init__(self, use_gpu: bool = True, batch_size: int = 1000):
        """
        初始化GPU碰撞檢測器
        
        Args:
            use_gpu: 是否使用GPU加速
            batch_size: 批次處理大小
        """
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        self.batch_size = batch_size
        
        logger.info(f"碰撞檢測器初始化: GPU={'啟用' if self.use_gpu else '禁用'}")
    
    @ensure_gpu_compatibility
    def calculate_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """
        計算所有無人機間的距離矩陣（GPU加速）
        
        Args:
            positions: 位置數組 shape: (n_drones, 3) [x, y, z]
            
        Returns:
            距離矩陣 shape: (n_drones, n_drones)
        """
        positions = self.xp.asarray(positions)
        n_drones = positions.shape[0]
        
        # 使用廣播計算所有配對的距離
        # positions[:, None, :] - positions[None, :, :] 生成差值張量
        diff = positions[:, None, :] - positions[None, :, :]
        
        # 計算歐幾里德距離
        distance_matrix = self.xp.sqrt(self.xp.sum(diff**2, axis=2))
        
        return distance_matrix
    
    @ensure_gpu_compatibility
    def detect_collisions_vectorized(self, positions: Dict[str, np.ndarray], 
                                   safety_distance: float) -> List[CollisionWarning]:
        """
        向量化碰撞檢測（GPU並行處理）
        
        Args:
            positions: 無人機位置字典 {drone_id: [x, y, z]}
            safety_distance: 安全距離（米）
            
        Returns:
            碰撞警告列表
        """
        if len(positions) < 2:
            return []
        
        # 準備數據
        drone_ids = list(positions.keys())
        position_array = self.xp.array([positions[drone_id] for drone_id in drone_ids])
        
        # 計算距離矩陣
        distance_matrix = self.calculate_distance_matrix(position_array)
        
        # 創建上三角遮罩（避免重複檢查和自己與自己的距離）
        n = len(drone_ids)
        triu_mask = self.xp.triu(self.xp.ones((n, n)), k=1).astype(bool)
        
        # 找出危險距離的配對
        danger_mask = (distance_matrix < safety_distance) & triu_mask
        danger_indices = self.xp.where(danger_mask)
        
        # 轉換為CPU進行後處理（如果使用GPU）
        if self.use_gpu and hasattr(danger_indices[0], 'get'):
            i_indices = danger_indices[0].get()
            j_indices = danger_indices[1].get()
            distances = distance_matrix[danger_mask].get()
        else:
            i_indices = danger_indices[0]
            j_indices = danger_indices[1]  
            distances = distance_matrix[danger_mask]
        
        # 生成警告
        warnings = []
        for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            distance = float(distances[idx])
            pos1 = positions[drone_ids[i]]
            pos2 = positions[drone_ids[j]]
            
            # 計算中點位置
            mid_pos = tuple((pos1 + pos2) / 2)
            
            # 確定嚴重程度
            severity = self._determine_severity(distance, safety_distance)
            
            warning = CollisionWarning(
                drone1_id=drone_ids[i],
                drone2_id=drone_ids[j],
                distance=distance,
                time=0.0,  # 需要外部提供當前時間
                position=mid_pos,
                severity=severity,
                relative_velocity=0.0  # 需要速度信息計算
            )
            warnings.append(warning)
        
        return warnings
    
    def _determine_severity(self, distance: float, safety_distance: float) -> ConflictSeverity:
        """
        根據距離確定衝突嚴重程度
        
        Args:
            distance: 實際距離
            safety_distance: 安全距離
            
        Returns:
            衝突嚴重程度
        """
        ratio = distance / safety_distance
        
        if ratio > 1.0:
            return ConflictSeverity.SAFE
        elif ratio > 0.8:
            return ConflictSeverity.WARNING
        elif ratio > 0.5:
            return ConflictSeverity.CRITICAL
        else:
            return ConflictSeverity.EMERGENCY


class TrajectoryAnalyzer:
    """
    軌跡分析器 - GPU加速版本
    分析多無人機軌跡的潛在衝突點
    """
    
    def __init__(self, safety_config: SafetyConfig, use_gpu: bool = True):
        """
        初始化軌跡分析器
        
        Args:
            safety_config: 安全配置
            use_gpu: 是否使用GPU加速
        """
        self.config = safety_config
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        self.collision_detector = GPUCollisionDetector(use_gpu)
        
    @ensure_gpu_compatibility
    def analyze_trajectory_conflicts(self, drones_data: Dict) -> List[TrajectoryConflict]:
        """
        分析整個軌跡期間的潛在衝突點（GPU加速）
        
        Args:
            drones_data: 無人機數據字典
            
        Returns:
            軌跡衝突列表
        """
        conflicts = []
        drone_ids = sorted(drones_data.keys())  # 數字小的優先權高
        
        # 批量處理無人機配對
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                drone1, drone2 = drone_ids[i], drone_ids[j]
                
                trajectory1 = drones_data[drone1].get('trajectory', [])
                trajectory2 = drones_data[drone2].get('trajectory', [])
                
                if not trajectory1 or not trajectory2:
                    continue
                
                # 分析兩條軌跡的衝突點
                pair_conflicts = self._analyze_trajectory_pair(
                    drone1, trajectory1, drone2, trajectory2
                )
                
                conflicts.extend(pair_conflicts)
                
        logger.info(f"軌跡分析完成: 發現 {len(conflicts)} 個潛在衝突")
        return conflicts
    
    def _analyze_trajectory_pair(self, drone1: str, traj1: List[Dict],
                               drone2: str, traj2: List[Dict]) -> List[TrajectoryConflict]:
        """
        分析一對無人機軌跡的衝突點
        
        Args:
            drone1, drone2: 無人機ID
            traj1, traj2: 軌跡數據
            
        Returns:
            衝突列表
        """
        conflicts = []
        
        # 確定分析時間範圍
        max_time = max(traj1[-1]['time'], traj2[-1]['time'])
        time_step = 0.5  # 0.5秒間隔採樣
        
        # 準備時間陣列進行批量計算
        time_points = self.xp.arange(0, max_time, time_step)
        
        # 批量插值位置
        positions1 = self._batch_interpolate_positions(traj1, time_points)
        positions2 = self._batch_interpolate_positions(traj2, time_points)
        
        # 計算距離陣列
        if positions1.size > 0 and positions2.size > 0:
            distances = self.xp.sqrt(self.xp.sum((positions1 - positions2)**2, axis=1))
            
            # 找出危險時刻
            danger_mask = distances < self.config.safety_distance
            danger_indices = self.xp.where(danger_mask)[0]
            
            if self.use_gpu and hasattr(danger_indices, 'get'):
                danger_indices = danger_indices.get()
                distances_cpu = distances.get()
            else:
                distances_cpu = distances
            
            # 處理危險時刻
            for idx in danger_indices:
                time_point = float(time_points[idx])
                distance = float(distances_cpu[idx])
                
                # 找到對應的航點索引
                wp1_idx = self._find_waypoint_index(traj1, time_point)
                wp2_idx = self._find_waypoint_index(traj2, time_point)
                
                # 計算等待時間
                wait_time = self._calculate_wait_time(traj1, traj2, time_point, distance)
                
                conflict = TrajectoryConflict(
                    drone1_id=drone1,
                    drone2_id=drone2,
                    conflict_time=time_point,
                    conflict_position1=tuple(positions1[idx]),
                    conflict_position2=tuple(positions2[idx]),
                    minimum_distance=distance,
                    severity=self.collision_detector._determine_severity(distance, self.config.safety_distance),
                    priority_drone=drone1,  # 數字小的優先
                    waiting_drone=drone2,
                    recommended_wait_time=wait_time,
                    waypoint1_index=wp1_idx,
                    waypoint2_index=wp2_idx
                )
                
                conflicts.append(conflict)
                
                logger.warning(f"發現衝突: {drone1}(WP{wp1_idx}) vs {drone2}(WP{wp2_idx}) "
                             f"在 {time_point:.1f}s 距離 {distance:.2f}m")
        
        return conflicts
    
    @ensure_gpu_compatibility
    def _batch_interpolate_positions(self, trajectory: List[Dict], 
                                   time_points: np.ndarray) -> np.ndarray:
        """
        批量插值軌跡位置（GPU加速）
        
        Args:
            trajectory: 軌跡數據
            time_points: 時間點陣列
            
        Returns:
            插值位置陣列 shape: (n_times, 3)
        """
        if not trajectory:
            return self.xp.array([])
        
        # 提取軌跡數據
        traj_times = self.xp.array([p['time'] for p in trajectory])
        traj_positions = self.xp.array([[p['x'], p['y'], p['z']] for p in trajectory])
        
        # 批量線性插值
        interpolated_positions = []
        
        for time_point in time_points:
            if time_point <= traj_times[0]:
                interpolated_positions.append(traj_positions[0])
            elif time_point >= traj_times[-1]:
                interpolated_positions.append(traj_positions[-1])
            else:
                # 找到插值區間
                idx = self.xp.searchsorted(traj_times, time_point) - 1
                idx = max(0, min(idx, len(traj_times) - 2))
                
                t1, t2 = traj_times[idx], traj_times[idx + 1]
                if t2 - t1 > 0:
                    ratio = (time_point - t1) / (t2 - t1)
                    pos = traj_positions[idx] + ratio * (traj_positions[idx + 1] - traj_positions[idx])
                else:
                    pos = traj_positions[idx]
                
                interpolated_positions.append(pos)
        
        return self.xp.array(interpolated_positions)
    
    def _find_waypoint_index(self, trajectory: List[Dict], time: float) -> int:
        """
        找到指定時間最接近的航點索引
        
        Args:
            trajectory: 軌跡數據
            time: 時間點
            
        Returns:
            航點索引
        """
        if not trajectory:
            return 0
        
        min_diff = float('inf')
        nearest_idx = 0
        
        for i, point in enumerate(trajectory):
            if 'waypoint_index' in point:
                time_diff = abs(point['time'] - time)
                if time_diff < min_diff:
                    min_diff = time_diff
                    nearest_idx = point.get('waypoint_index', i)
        
        return nearest_idx
    
    def _calculate_wait_time(self, traj1: List[Dict], traj2: List[Dict], 
                           conflict_time: float, distance: float) -> float:
        """
        計算精確的等待時間
        
        Args:
            traj1, traj2: 軌跡數據
            conflict_time: 衝突時間
            distance: 衝突距離
            
        Returns:
            建議等待時間（秒）
        """
        # 基本等待時間計算
        base_wait = 3.0  # 最少等待3秒
        
        # 根據衝突嚴重程度調整
        if distance < self.config.critical_distance:
            severity_multiplier = 2.0
        elif distance < self.config.warning_distance:
            severity_multiplier = 1.5
        else:
            severity_multiplier = 1.0
        
        # 計算前一架飛機飛出安全距離需要的時間
        safety_buffer = 2.0  # 額外安全緩衝
        check_interval = 0.1
        
        for t in np.arange(conflict_time, traj1[-1]['time'], check_interval):
            # 這裡需要更詳細的位置計算邏輯
            # 簡化版本：基於距離和速度估算
            estimated_wait = (self.config.safety_distance + safety_buffer) / 8.0  # 假設8m/s速度
            return max(base_wait * severity_multiplier, estimated_wait)
        
        # 如果無法計算，返回保守估計
        return base_wait * severity_multiplier + 5.0


class CollisionAvoidanceSystem:
    """
    完整的碰撞避免系統 - GPU加速版本
    整合實時碰撞檢測和軌跡衝突分析
    """
    
    def __init__(self, safety_config: SafetyConfig, use_gpu: bool = True):
        """
        初始化碰撞避免系統
        
        Args:
            safety_config: 安全配置
            use_gpu: 是否使用GPU加速
        """
        self.config = safety_config
        self.use_gpu = use_gpu
        
        # 初始化子系統
        self.collision_detector = GPUCollisionDetector(use_gpu)
        self.trajectory_analyzer = TrajectoryAnalyzer(safety_config, use_gpu)
        
        # 狀態追踪
        self.collision_warnings: List[CollisionWarning] = []
        self.trajectory_conflicts: List[TrajectoryConflict] = []
        self.last_collision_check = 0.0
        
        logger.info("碰撞避免系統初始化完成")
    
    def check_realtime_collisions(self, positions: Dict[str, np.ndarray], 
                                current_time: float) -> Tuple[List[CollisionWarning], Dict[str, float]]:
        """
        實時碰撞檢測（用於動畫播放期間）
        
        Args:
            positions: 當前位置字典 {drone_id: [x, y, z]}
            current_time: 當前時間
            
        Returns:
            (碰撞警告列表, 新的LOITER延遲字典)
        """
        self.collision_warnings.clear()
        new_loiters = {}
        
        # 只在指定間隔檢查（避免過度計算）
        if current_time - self.last_collision_check >= self.config.collision_check_interval:
            
            # 執行GPU加速碰撞檢測
            warnings = self.collision_detector.detect_collisions_vectorized(
                positions, self.config.safety_distance
            )
            
            # 更新時間信息
            for warning in warnings:
                warning.time = current_time
                
                # 計算相對速度（如果有歷史數據）
                warning.relative_velocity = self._estimate_relative_velocity(
                    warning.drone1_id, warning.drone2_id, positions
                )
            
            self.collision_warnings = warnings
            self.last_collision_check = current_time
            
            # 生成緊急LOITER命令（如果需要）
            for warning in warnings:
                if warning.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.EMERGENCY]:
                    # 讓ID較大的無人機進行緊急等待
                    waiting_drone = max(warning.drone1_id, warning.drone2_id)
                    wait_time = 5.0 if warning.severity == ConflictSeverity.CRITICAL else 10.0
                    new_loiters[waiting_drone] = wait_time
                    
                    logger.warning(f"緊急避讓: {waiting_drone} 等待 {wait_time}s")
        
        return self.collision_warnings, new_loiters
    
    def _estimate_relative_velocity(self, drone1_id: str, drone2_id: str, 
                                  positions: Dict[str, np.ndarray]) -> float:
        """
        估算兩架無人機的相對速度
        
        Args:
            drone1_id, drone2_id: 無人機ID
            positions: 當前位置
            
        Returns:
            相對速度 (m/s)
        """
        # 這裡需要歷史位置數據來計算速度
        # 簡化版本返回固定值
        return 8.0  # 假設巡航速度8m/s
    
    def get_system_status(self) -> dict:
        """獲取系統狀態信息"""
        return {
            'gpu_enabled': self.use_gpu,
            'active_warnings': len(self.collision_warnings),
            'trajectory_conflicts': len(self.trajectory_conflicts),
            'last_check_time': self.last_collision_check,
            'safety_distance': self.config.safety_distance,
            'warning_distance': self.config.warning_distance,
            'critical_distance': self.config.critical_distance
        }
    
    def optimize_performance(self):
        """優化系統性能"""
        if self.use_gpu:
            try:
                # 清理GPU記憶體
                cp.get_default_memory_pool().free_all_blocks()
                logger.info("碰撞檢測系統GPU記憶體已優化")
            except:
                pass


# 便利函數
def create_collision_system(safety_distance: float = 5.0, 
                          warning_distance: float = 8.0,
                          critical_distance: float = 3.0,
                          use_gpu: bool = True) -> CollisionAvoidanceSystem:
    """
    快速創建碰撞避免系統
    
    Args:
        safety_distance: 安全距離
        warning_distance: 警告距離  
        critical_distance: 危險距離
        use_gpu: 是否使用GPU
        
    Returns:
        配置好的碰撞避免系統
    """
    from config.settings import SafetyConfig
    
    config = SafetyConfig(
        safety_distance=safety_distance,
        warning_distance=warning_distance,
        critical_distance=critical_distance
    )
    
    return CollisionAvoidanceSystem(config, use_gpu)