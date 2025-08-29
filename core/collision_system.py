#!/usr/bin/env python3
"""
先進碰撞避免系統 - GPU加速版本
支援即時碰撞檢測、軌跡分析和自動避讓策略
"""

import numpy as np
import math
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time

# 嘗試導入GPU加速模組
try:
    from utils.gpu_utils import compute_manager, MathOps, asarray, to_cpu, synchronize
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    compute_manager = None
    MathOps = None

logger = logging.getLogger(__name__)

@dataclass
class SafetyConfig:
    """安全配置參數"""
    safety_distance: float = 5.0        # 安全距離 (公尺)
    warning_distance: float = 8.0       # 警告距離 (公尺)
    critical_distance: float = 3.0      # 緊急距離 (公尺)
    collision_check_interval: float = 0.1  # 碰撞檢查間隔 (秒)
    prediction_horizon: float = 10.0    # 預測時間範圍 (秒)
    avoidance_margin: float = 2.0       # 避讓餘量 (公尺)

@dataclass
class CollisionWarning:
    """碰撞警告數據結構"""
    drone1_id: str
    drone2_id: str
    distance: float
    time_to_collision: float
    collision_point: np.ndarray
    severity: str  # 'warning', 'critical', 'imminent'
    timestamp: float
    avoidance_suggestion: Dict = None

class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        """重置統計數據"""
        self.collision_checks = 0
        self.gpu_operations = 0
        self.cpu_fallbacks = 0
        self.total_time = 0.0
        self.start_time = time.time()
    
    def record_operation(self, operation_type: str, duration: float, using_gpu: bool = False):
        """記錄操作統計"""
        self.total_time += duration
        
        if operation_type == 'collision_check':
            self.collision_checks += 1
            
        if using_gpu:
            self.gpu_operations += 1
        else:
            self.cpu_fallbacks += 1
    
    def get_stats(self) -> Dict:
        """獲取統計信息"""
        runtime = time.time() - self.start_time
        return {
            'collision_checks': self.collision_checks,
            'gpu_operations': self.gpu_operations,
            'cpu_fallbacks': self.cpu_fallbacks,
            'total_computation_time': self.total_time,
            'runtime': runtime,
            'gpu_utilization': self.gpu_operations / max(1, self.gpu_operations + self.cpu_fallbacks),
            'checks_per_second': self.collision_checks / max(0.001, runtime)
        }

class CollisionDetector:
    """核心碰撞檢測引擎"""
    
    def __init__(self, safety_config: SafetyConfig):
        self.config = safety_config
        self.use_gpu = GPU_AVAILABLE and compute_manager is not None
        self.performance = PerformanceMonitor()
        
        if self.use_gpu:
            logger.info(f"碰撞檢測器初始化 - 使用 {compute_manager.backend.name} 後端")
        else:
            logger.info("碰撞檢測器初始化 - 使用 CPU 後端")
    
    def check_immediate_collisions(self, positions: Dict[str, np.ndarray]) -> List[CollisionWarning]:
        """檢查即時碰撞風險 - GPU加速版本"""
        start_time = time.time()
        warnings = []
        
        if len(positions) < 2:
            return warnings
        
        try:
            drone_ids = list(positions.keys())
            position_array = np.array([positions[drone_id] for drone_id in drone_ids])
            
            # 使用GPU計算距離矩陣
            if self.use_gpu and len(drone_ids) > 4:
                distances = self._gpu_distance_matrix(position_array)
                using_gpu = True
            else:
                distances = self._cpu_distance_matrix(position_array)
                using_gpu = False
            
            # 檢查所有配對的距離
            for i in range(len(drone_ids)):
                for j in range(i + 1, len(drone_ids)):
                    distance = distances[i, j]
                    
                    if distance < self.config.warning_distance:
                        severity = self._determine_severity(distance)
                        
                        warning = CollisionWarning(
                            drone1_id=drone_ids[i],
                            drone2_id=drone_ids[j],
                            distance=distance,
                            time_to_collision=0.0,  # 即時碰撞
                            collision_point=(position_array[i] + position_array[j]) / 2,
                            severity=severity,
                            timestamp=time.time()
                        )
                        warnings.append(warning)
            
            # 記錄性能
            duration = time.time() - start_time
            self.performance.record_operation('collision_check', duration, using_gpu)
            
        except Exception as e:
            logger.error(f"碰撞檢測失敗: {e}")
            duration = time.time() - start_time
            self.performance.record_operation('collision_check', duration, False)
        
        return warnings
    
    def predict_trajectory_collisions(self, trajectories: Dict[str, List[Dict]], 
                                    current_time: float) -> List[CollisionWarning]:
        """預測軌跡碰撞 - 高精度分析"""
        warnings = []
        drone_ids = list(trajectories.keys())
        
        # 時間採樣點
        time_horizon = min(self.config.prediction_horizon, 30.0)  # 最多預測30秒
        time_steps = np.arange(current_time, current_time + time_horizon, 0.5)
        
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                drone1_id, drone2_id = drone_ids[i], drone_ids[j]
                traj1, traj2 = trajectories[drone1_id], trajectories[drone2_id]
                
                # 預測每個時間點的位置
                for t in time_steps:
                    pos1 = self._interpolate_position(traj1, t)
                    pos2 = self._interpolate_position(traj2, t)
                    
                    if pos1 is not None and pos2 is not None:
                        distance = np.linalg.norm(pos1 - pos2)
                        
                        if distance < self.config.safety_distance:
                            time_to_collision = t - current_time
                            severity = self._determine_severity(distance)
                            
                            warning = CollisionWarning(
                                drone1_id=drone1_id,
                                drone2_id=drone2_id,
                                distance=distance,
                                time_to_collision=time_to_collision,
                                collision_point=(pos1 + pos2) / 2,
                                severity=severity,
                                timestamp=time.time(),
                                avoidance_suggestion=self._suggest_avoidance(
                                    drone1_id, drone2_id, pos1, pos2, time_to_collision
                                )
                            )
                            warnings.append(warning)
                            break  # 只報告第一個碰撞點
        
        return warnings
    
    def _gpu_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """GPU加速距離矩陣計算"""
        if MathOps is not None:
            try:
                return MathOps.distance_matrix(positions, positions)
            except Exception as e:
                logger.warning(f"GPU距離計算失敗，回退到CPU: {e}")
                return self._cpu_distance_matrix(positions)
        else:
            return self._cpu_distance_matrix(positions)
    
    def _cpu_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """CPU距離矩陣計算"""
        n = len(positions)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = distances[j, i] = dist
        
        return distances
    
    def _interpolate_position(self, trajectory: List[Dict], time: float) -> Optional[np.ndarray]:
        """軌跡位置插值"""
        if not trajectory:
            return None
        
        # 邊界條件
        if time >= trajectory[-1]['time']:
            last_point = trajectory[-1]
            return np.array([last_point['x'], last_point['y'], last_point['z']])
        
        if time <= trajectory[0]['time']:
            first_point = trajectory[0]
            return np.array([first_point['x'], first_point['y'], first_point['z']])
        
        # 線性插值
        for i in range(len(trajectory) - 1):
            t1, t2 = trajectory[i]['time'], trajectory[i + 1]['time']
            if t1 <= time <= t2:
                if t2 - t1 == 0:
                    point = trajectory[i]
                    return np.array([point['x'], point['y'], point['z']])
                
                ratio = (time - t1) / (t2 - t1)
                p1 = trajectory[i]
                p2 = trajectory[i + 1]
                
                pos = np.array([
                    p1['x'] + ratio * (p2['x'] - p1['x']),
                    p1['y'] + ratio * (p2['y'] - p1['y']),
                    p1['z'] + ratio * (p2['z'] - p1['z'])
                ])
                return pos
        
        return None
    
    def _determine_severity(self, distance: float) -> str:
        """判斷碰撞嚴重程度"""
        if distance < self.config.critical_distance:
            return 'critical'
        elif distance < self.config.safety_distance:
            return 'warning'
        else:
            return 'notice'
    
    def _suggest_avoidance(self, drone1_id: str, drone2_id: str, 
                          pos1: np.ndarray, pos2: np.ndarray, 
                          time_to_collision: float) -> Dict:
        """建議避讓策略"""
        # 簡單的避讓建議：數字小的無人機有優先權
        drone1_priority = int(drone1_id.split('_')[-1]) if '_' in drone1_id else 0
        drone2_priority = int(drone2_id.split('_')[-1]) if '_' in drone2_id else 1
        
        if drone1_priority < drone2_priority:
            # drone1 有優先權，drone2 需要避讓
            waiting_drone = drone2_id
            priority_drone = drone1_id
        else:
            waiting_drone = drone1_id
            priority_drone = drone2_id
        
        # 計算避讓時間
        avoidance_time = max(5.0, time_to_collision + 3.0)
        
        return {
            'strategy': 'loiter',
            'waiting_drone': waiting_drone,
            'priority_drone': priority_drone,
            'wait_time': avoidance_time,
            'avoidance_altitude': max(pos1[2], pos2[2]) + self.config.avoidance_margin
        }

class AdvancedCollisionAvoidanceSystem:
    """先進碰撞避免系統 - 完整解決方案"""
    
    def __init__(self, safety_config: SafetyConfig = None):
        self.config = safety_config or SafetyConfig()
        self.detector = CollisionDetector(self.config)
        self.active_warnings: List[CollisionWarning] = []
        self.avoidance_commands: Dict[str, Dict] = {}
        self.last_check_time = 0.0
        
        logger.info("先進碰撞避免系統已啟動")
    
    def update(self, current_positions: Dict[str, np.ndarray], 
              trajectories: Dict[str, List[Dict]] = None,
              current_time: float = None) -> Tuple[List[CollisionWarning], Dict[str, Dict]]:
        """系統更新 - 主要處理入口"""
        if current_time is None:
            current_time = time.time()
        
        # 檢查是否需要更新
        if current_time - self.last_check_time < self.config.collision_check_interval:
            return self.active_warnings, self.avoidance_commands
        
        self.last_check_time = current_time
        
        # 清空舊的警告
        self.active_warnings.clear()
        
        # 即時碰撞檢測
        immediate_warnings = self.detector.check_immediate_collisions(current_positions)
        self.active_warnings.extend(immediate_warnings)
        
        # 軌跡預測碰撞檢測
        if trajectories:
            predicted_warnings = self.detector.predict_trajectory_collisions(trajectories, current_time)
            self.active_warnings.extend(predicted_warnings)
        
        # 生成避讓命令
        self._generate_avoidance_commands()
        
        # 記錄統計
        if len(self.active_warnings) > 0:
            logger.warning(f"檢測到 {len(self.active_warnings)} 個碰撞警告")
        
        return self.active_warnings, self.avoidance_commands
    
    def _generate_avoidance_commands(self):
        """根據警告生成避讓命令"""
        self.avoidance_commands.clear()
        
        for warning in self.active_warnings:
            if warning.avoidance_suggestion and warning.severity in ['critical', 'warning']:
                suggestion = warning.avoidance_suggestion
                waiting_drone = suggestion['waiting_drone']
                
                # 避免重複命令
                if waiting_drone not in self.avoidance_commands:
                    self.avoidance_commands[waiting_drone] = {
                        'command': suggestion['strategy'],
                        'parameters': {
                            'wait_time': suggestion['wait_time'],
                            'reason': f"避讓 {suggestion['priority_drone']}",
                            'altitude': suggestion.get('avoidance_altitude')
                        },
                        'priority': 1 if warning.severity == 'critical' else 2,
                        'timestamp': warning.timestamp
                    }
    
    def get_system_status(self) -> Dict:
        """獲取系統狀態"""
        detector_stats = self.detector.performance.get_stats()
        
        return {
            'active_warnings': len(self.active_warnings),
            'avoidance_commands': len(self.avoidance_commands),
            'last_check_time': self.last_check_time,
            'safety_distance': self.config.safety_distance,
            'using_gpu': self.detector.use_gpu,
            'performance': detector_stats
        }
    
    def update_safety_config(self, new_config: SafetyConfig):
        """更新安全配置"""
        self.config = new_config
        self.detector.config = new_config
        logger.info(f"安全配置已更新 - 安全距離: {new_config.safety_distance}m")
    
    def reset_warnings(self):
        """重置所有警告"""
        self.active_warnings.clear()
        self.avoidance_commands.clear()
        logger.info("碰撞警告已重置")
    
    def export_collision_report(self, filename: str = None) -> Dict:
        """匯出碰撞報告"""
        if filename is None:
            filename = f"collision_report_{int(time.time())}.json"
        
        report = {
            'timestamp': time.time(),
            'config': {
                'safety_distance': self.config.safety_distance,
                'warning_distance': self.config.warning_distance,
                'critical_distance': self.config.critical_distance
            },
            'current_warnings': [
                {
                    'drone1': w.drone1_id,
                    'drone2': w.drone2_id,
                    'distance': w.distance,
                    'severity': w.severity,
                    'time_to_collision': w.time_to_collision
                }
                for w in self.active_warnings
            ],
            'avoidance_commands': self.avoidance_commands,
            'system_status': self.get_system_status()
        }
        
        try:
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"碰撞報告已匯出: {filename}")
        except Exception as e:
            logger.error(f"匯出碰撞報告失敗: {e}")
        
        return report

# 便利函數
def create_collision_system(safety_distance: float = 5.0,
                           warning_distance: float = 8.0) -> AdvancedCollisionAvoidanceSystem:
    """創建碰撞避免系統"""
    config = SafetyConfig(
        safety_distance=safety_distance,
        warning_distance=warning_distance,
        critical_distance=safety_distance * 0.6
    )
    return AdvancedCollisionAvoidanceSystem(config)

def quick_collision_check(positions: Dict[str, np.ndarray], 
                         safety_distance: float = 5.0) -> List[Tuple[str, str, float]]:
    """快速碰撞檢查 - 返回簡化結果"""
    detector = CollisionDetector(SafetyConfig(safety_distance=safety_distance))
    warnings = detector.check_immediate_collisions(positions)
    
    return [(w.drone1_id, w.drone2_id, w.distance) for w in warnings]