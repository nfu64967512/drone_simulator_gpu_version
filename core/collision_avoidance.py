"""
GPU加速碰撞檢測和避免系統
支持大規模無人機群的高效能碰撞檢測
"""
import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import json
from numba import cuda
import warnings

# 導入GPU工具
from utils.gpu_utils import (
    get_array_module, asarray, to_cpu, to_gpu, is_gpu_enabled,
    synchronize, gpu_accelerated, MathOps, performance_monitor
)
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class CollisionEvent:
    """碰撞事件記錄"""
    timestamp: str
    simulation_time: float
    drone1: str
    drone2: str
    distance: float
    severity: str  # "warning", "critical", "collision"
    position1: List[float]
    position2: List[float]
    relative_velocity: float = 0.0
    resolution_action: str = ""

@dataclass
class CollisionMetrics:
    """碰撞統計指標"""
    total_events: int = 0
    warning_events: int = 0
    critical_events: int = 0
    collision_events: int = 0
    unique_drone_pairs: Set[Tuple[str, str]] = field(default_factory=set)
    min_distance: float = float('inf')
    avg_distance: float = 0.0

class GPUCollisionDetector:
    """GPU加速碰撞檢測器"""
    
    def __init__(self):
        self.xp = get_array_module()
        
        # 碰撞事件記錄
        self.collision_events: List[CollisionEvent] = []
        self.metrics = CollisionMetrics()
        
        # GPU優化的檢測參數
        self.safety_distance = settings.safety.safety_distance
        self.warning_distance = settings.safety.warning_distance  
        self.critical_distance = settings.safety.critical_distance
        self.check_interval = settings.safety.collision_check_interval
        
        # 批次處理參數
        self.batch_size = settings.gpu.batch_size
        self.max_drones = 1000  # GPU記憶體限制
        
        # 效能優化
        self.last_check_time = 0.0
        self.detection_cache = {}
        self.distance_matrices = {}  # 快取距離矩陣
        
        # GPU CUDA kernels (如果可用)
        self._init_cuda_kernels()
        
        logger.info(f"🛡️ GPU碰撞檢測器初始化 (後端: {('GPU' if is_gpu_enabled() else 'CPU')})")

    def _init_cuda_kernels(self):
        """初始化CUDA kernels（如果可用）"""
        try:
            if is_gpu_enabled() and cuda.is_available():
                self.cuda_kernels_available = True
                logger.info("🚀 CUDA kernels 可用，啟用高效能碰撞檢測")
                self._compile_cuda_kernels()
            else:
                self.cuda_kernels_available = False
                logger.info("⚡ 使用GPU陣列運算進行碰撞檢測")
        except Exception as e:
            self.cuda_kernels_available = False
            logger.warning(f"⚠️ CUDA kernels 初始化失敗: {e}")

    def _compile_cuda_kernels(self):
        """編譯自定義CUDA kernels"""
        if not self.cuda_kernels_available:
            return
        
        # 距離計算kernel
        @cuda.jit
        def distance_matrix_kernel(positions, distances, n_drones):
            """並行計算所有無人機對的距離"""
            i, j = cuda.grid(2)
            
            if i < n_drones and j < n_drones and i != j:
                # 計算3D歐氏距離
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1] 
                dz = positions[i, 2] - positions[j, 2]
                
                dist = (dx*dx + dy*dy + dz*dz) ** 0.5
                distances[i, j] = dist
        
        # 碰撞檢測kernel
        @cuda.jit
        def collision_detection_kernel(distances, collision_flags, safety_dist, n_drones):
            """並行檢測碰撞"""
            i, j = cuda.grid(2)
            
            if i < n_drones and j < n_drones and i < j:  # 避免重複檢測
                if distances[i, j] < safety_dist:
                    collision_flags[i, j] = 1
                    collision_flags[j, i] = 1
        
        self.distance_matrix_kernel = distance_matrix_kernel
        self.collision_detection_kernel = collision_detection_kernel

    @gpu_accelerated()
    def detect_collisions_batch(
        self, 
        positions: Any, 
        velocities: Optional[Any] = None,
        drone_names: Optional[List[str]] = None,
        simulation_time: float = 0.0
    ) -> List[CollisionEvent]:
        """
        批次碰撞檢測
        
        Args:
            positions: 形狀為 (n_drones, 3) 的位置陣列
            velocities: 可選的速度陣列，用於預測性碰撞檢測
            drone_names: 無人機名稱列表
            simulation_time: 當前模擬時間
        
        Returns:
            檢測到的碰撞事件列表
        """
        start_time = time.perf_counter()
        
        # 轉換輸入為GPU陣列
        positions_gpu = asarray(positions)
        n_drones = positions_gpu.shape[0]
        
        if n_drones < 2:
            return []
        
        # 檢查是否需要進行檢測
        if simulation_time - self.last_check_time < self.check_interval:
            return []
        
        collision_events = []
        
        try:
            if self.cuda_kernels_available and n_drones >= 32:
                # 使用CUDA kernels進行大規模檢測
                collision_events = self._detect_collisions_cuda(
                    positions_gpu, velocities, drone_names, simulation_time
                )
            else:
                # 使用GPU陣列運算
                collision_events = self._detect_collisions_gpu_arrays(
                    positions_gpu, velocities, drone_names, simulation_time  
                )
            
            # 更新統計資訊
            self._update_metrics(collision_events)
            
            # 記錄效能
            detection_time = time.perf_counter() - start_time
            performance_monitor.time_function(lambda: None)  # 記錄時間
            
            if len(collision_events) > 0:
                logger.warning(f"⚠️ 檢測到 {len(collision_events)} 個碰撞事件 (耗時: {detection_time*1000:.2f}ms)")
            
            self.last_check_time = simulation_time
            
        except Exception as e:
            logger.error(f"❌ 碰撞檢測失敗: {e}")
            # 回退到CPU檢測
            if is_gpu_enabled():
                logger.info("🔄 回退到CPU碰撞檢測...")
                positions_cpu = to_cpu(positions_gpu)
                collision_events = self._detect_collisions_cpu_fallback(
                    positions_cpu, drone_names, simulation_time
                )
        
        return collision_events

    def _detect_collisions_cuda(
        self, 
        positions: Any, 
        velocities: Optional[Any],
        drone_names: Optional[List[str]], 
        simulation_time: float
    ) -> List[CollisionEvent]:
        """使用CUDA kernels進行碰撞檢測"""
        n_drones = positions.shape[0]
        
        # 分配GPU記憶體
        distances = self.xp.zeros((n_drones, n_drones), dtype=self.xp.float32)
        collision_flags = self.xp.zeros((n_drones, n_drones), dtype=self.xp.int32)
        
        # 設置CUDA網格大小
        threads_per_block = (16, 16)
        blocks_per_grid_x = (n_drones + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (n_drones + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # 執行距離計算kernel
        self.distance_matrix_kernel[blocks_per_grid, threads_per_block](
            positions, distances, n_drones
        )
        
        # 執行碰撞檢測kernel
        self.collision_detection_kernel[blocks_per_grid, threads_per_block](
            distances, collision_flags, self.safety_distance, n_drones
        )
        
        # 同步GPU操作
        synchronize()
        
        # 提取碰撞事件
        return self._extract_collision_events(
            distances, collision_flags, drone_names, simulation_time
        )

    @gpu_accelerated()
    def _detect_collisions_gpu_arrays(
        self,
        positions: Any,
        velocities: Optional[Any], 
        drone_names: Optional[List[str]],
        simulation_time: float
    ) -> List[CollisionEvent]:
        """使用GPU陣列運算進行碰撞檢測"""
        
        # 計算距離矩陣
        distances = MathOps.distance_matrix(positions, positions)
        
        # 多級別威脅檢測
        collision_events = []
        
        # 嚴重碰撞 (< critical_distance)
        critical_mask = (distances < self.critical_distance) & (distances > 0)
        collision_events.extend(self._process_collision_mask(
            critical_mask, distances, drone_names, simulation_time, "collision"
        ))
        
        # 危險接近 (< warning_distance)
        warning_mask = (distances < self.warning_distance) & (distances >= self.critical_distance)
        collision_events.extend(self._process_collision_mask(
            warning_mask, distances, drone_names, simulation_time, "critical"
        ))
        
        # 安全警告 (< safety_distance)
        safety_mask = (distances < self.safety_distance) & (distances >= self.warning_distance)
        collision_events.extend(self._process_collision_mask(
            safety_mask, distances, drone_names, simulation_time, "warning"
        ))
        
        # 預測性碰撞檢測（如果提供速度資料）
        if velocities is not None:
            predictive_events = self._predict_future_collisions(
                positions, velocities, drone_names, simulation_time
            )
            collision_events.extend(predictive_events)
        
        return collision_events

    def _process_collision_mask(
        self, 
        mask: Any, 
        distances: Any, 
        drone_names: Optional[List[str]], 
        simulation_time: float,
        severity: str
    ) -> List[CollisionEvent]:
        """處理碰撞遮罩並生成事件"""
        events = []
        
        if self.xp.any(mask):
            # 轉回CPU進行事件處理
            mask_cpu = to_cpu(mask)
            distances_cpu = to_cpu(distances)
            collision_indices = np.where(mask_cpu)
            
            for i in range(len(collision_indices[0])):
                idx1, idx2 = collision_indices[0][i], collision_indices[1][i]
                
                # 避免重複事件 (只處理上三角)
                if idx1 >= idx2:
                    continue
                
                distance = distances_cpu[idx1, idx2]
                
                # 創建碰撞事件
                event = CollisionEvent(
                    timestamp=datetime.now().isoformat(),
                    simulation_time=simulation_time,
                    drone1=drone_names[idx1] if drone_names else f"Drone_{idx1}",
                    drone2=drone_names[idx2] if drone_names else f"Drone_{idx2}",
                    distance=float(distance),
                    severity=severity,
                    position1=[0, 0, 0],  # 將在後面填充
                    position2=[0, 0, 0]
                )
                
                events.append(event)
        
        return events

    @gpu_accelerated()
    def _predict_future_collisions(
        self,
        positions: Any,
        velocities: Any,
        drone_names: Optional[List[str]],
        simulation_time: float,
        prediction_time: float = 3.0
    ) -> List[CollisionEvent]:
        """預測未來可能的碰撞"""
        # 預測未來位置
        future_positions = positions + velocities * prediction_time
        
        # 檢測預測位置的碰撞
        future_distances = MathOps.distance_matrix(future_positions, future_positions)
        
        # 找出可能的未來碰撞
        future_collision_mask = (future_distances < self.safety_distance) & (future_distances > 0)
        
        # 檢查當前距離是否安全（避免重複警告）
        current_distances = MathOps.distance_matrix(positions, positions)
        safe_now_mask = current_distances >= self.safety_distance
        
        # 組合條件：現在安全但未來危險
        prediction_mask = future_collision_mask & safe_now_mask
        
        return self._process_collision_mask(
            prediction_mask, future_distances, drone_names, 
            simulation_time + prediction_time, "predicted"
        )

    def _detect_collisions_cpu_fallback(
        self,
        positions: np.ndarray,
        drone_names: Optional[List[str]],
        simulation_time: float
    ) -> List[CollisionEvent]:
        """CPU回退碰撞檢測"""
        events = []
        n_drones = len(positions)
        
        for i in range(n_drones):
            for j in range(i + 1, n_drones):
                # 計算距離
                pos1, pos2 = positions[i], positions[j]
                distance = np.linalg.norm(pos1 - pos2)
                
                # 判斷威脅等級
                if distance < self.critical_distance:
                    severity = "collision"
                elif distance < self.warning_distance:
                    severity = "critical"
                elif distance < self.safety_distance:
                    severity = "warning"
                else:
                    continue
                
                # 創建事件
                event = CollisionEvent(
                    timestamp=datetime.now().isoformat(),
                    simulation_time=simulation_time,
                    drone1=drone_names[i] if drone_names else f"Drone_{i}",
                    drone2=drone_names[j] if drone_names else f"Drone_{j}",
                    distance=float(distance),
                    severity=severity,
                    position1=pos1.tolist(),
                    position2=pos2.tolist()
                )
                
                events.append(event)
        
        return events

    def _extract_collision_events(
        self,
        distances: Any,
        collision_flags: Any, 
        drone_names: Optional[List[str]],
        simulation_time: float
    ) -> List[CollisionEvent]:
        """從CUDA結果提取碰撞事件"""
        events = []
        
        # 轉回CPU處理結果
        flags_cpu = to_cpu(collision_flags)
        distances_cpu = to_cpu(distances)
        
        collision_indices = np.where(flags_cpu == 1)
        
        for i in range(len(collision_indices[0])):
            idx1, idx2 = collision_indices[0][i], collision_indices[1][i]
            
            # 避免重複
            if idx1 >= idx2:
                continue
                
            distance = distances_cpu[idx1, idx2]
            
            # 判斷嚴重程度
            if distance < self.critical_distance:
                severity = "collision"
            elif distance < self.warning_distance:
                severity = "critical"
            else:
                severity = "warning"
            
            event = CollisionEvent(
                timestamp=datetime.now().isoformat(),
                simulation_time=simulation_time,
                drone1=drone_names[idx1] if drone_names else f"Drone_{idx1}",
                drone2=drone_names[idx2] if drone_names else f"Drone_{idx2}",
                distance=float(distance),
                severity=severity,
                position1=[0, 0, 0],  # 填充實際位置
                position2=[0, 0, 0]
            )
            
            events.append(event)
        
        return events

    def _update_metrics(self, events: List[CollisionEvent]):
        """更新碰撞統計指標"""
        for event in events:
            self.metrics.total_events += 1
            
            # 按嚴重程度分類
            if event.severity == "warning":
                self.metrics.warning_events += 1
            elif event.severity == "critical":
                self.metrics.critical_events += 1
            elif event.severity == "collision":
                self.metrics.collision_events += 1
            
            # 記錄無人機對
            drone_pair = tuple(sorted([event.drone1, event.drone2]))
            self.metrics.unique_drone_pairs.add(drone_pair)
            
            # 更新距離統計
            if event.distance < self.metrics.min_distance:
                self.metrics.min_distance = event.distance
            
            # 更新平均距離
            total_distances = sum(e.distance for e in self.collision_events + [event])
            self.metrics.avg_distance = total_distances / len(self.collision_events + [event])
        
        # 保存事件
        self.collision_events.extend(events)

    def log_collision_event(
        self, 
        drone1: str, 
        drone2: str, 
        distance: float,
        pos1: Any, 
        pos2: Any, 
        simulation_time: float
    ):
        """記錄單個碰撞事件"""
        # 判斷嚴重程度
        if distance < self.critical_distance:
            severity = "collision"
        elif distance < self.warning_distance:
            severity = "critical"
        else:
            severity = "warning"
        
        event = CollisionEvent(
            timestamp=datetime.now().isoformat(),
            simulation_time=simulation_time,
            drone1=drone1,
            drone2=drone2,
            distance=distance,
            severity=severity,
            position1=to_cpu(asarray(pos1)).tolist(),
            position2=to_cpu(asarray(pos2)).tolist()
        )
        
        self.collision_events.append(event)
        self._update_metrics([event])

    def export_collision_log(self, filename: str) -> bool:
        """匯出碰撞記錄"""
        try:
            export_data = {
                "metadata": {
                    "total_events": self.metrics.total_events,
                    "export_time": datetime.now().isoformat(),
                    "gpu_backend": is_gpu_enabled(),
                    "cuda_kernels": self.cuda_kernels_available,
                    "statistics": {
                        "warning_events": self.metrics.warning_events,
                        "critical_events": self.metrics.critical_events,
                        "collision_events": self.metrics.collision_events,
                        "unique_drone_pairs": len(self.metrics.unique_drone_pairs),
                        "min_distance": self.metrics.min_distance if self.metrics.min_distance != float('inf') else 0,
                        "avg_distance": self.metrics.avg_distance
                    }
                },
                "collision_events": [
                    {
                        "timestamp": event.timestamp,
                        "simulation_time": event.simulation_time,
                        "drone1": event.drone1,
                        "drone2": event.drone2,
                        "distance": event.distance,
                        "severity": event.severity,
                        "position1": event.position1,
                        "position2": event.position2,
                        "relative_velocity": event.relative_velocity,
                        "resolution_action": event.resolution_action
                    }
                    for event in self.collision_events
                ]
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 碰撞記錄已匯出: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 匯出碰撞記錄失敗: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取效能統計資訊"""
        return {
            "backend": "GPU" if is_gpu_enabled() else "CPU",
            "cuda_kernels": self.cuda_kernels_available,
            "total_detections": len(self.collision_events),
            "detection_efficiency": {
                "batch_size": self.batch_size,
                "check_interval": self.check_interval,
                "last_check_time": self.last_check_time
            },
            "metrics": {
                "total_events": self.metrics.total_events,
                "warning_events": self.metrics.warning_events,
                "critical_events": self.metrics.critical_events,
                "collision_events": self.metrics.collision_events,
                "unique_pairs": len(self.metrics.unique_drone_pairs),
                "min_distance": self.metrics.min_distance if self.metrics.min_distance != float('inf') else 0,
                "avg_distance": self.metrics.avg_distance
            }
        }

    def clear_collision_history(self):
        """清除碰撞歷史"""
        self.collision_events.clear()
        self.metrics = CollisionMetrics()
        logger.info("🧹 碰撞歷史已清除")

# 向後相容的別名
CollisionDetector = GPUCollisionDetector