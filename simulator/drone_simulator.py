"""
GPU加速無人機群模擬器
支持大規模無人機群的高效能模擬
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 導入GPU工具
from utils.gpu_utils import (
    get_array_module, asarray, to_cpu, to_gpu, is_gpu_enabled, 
    synchronize, gpu_accelerated, MathOps, performance_monitor
)
from config.settings import settings
from core.collision_avoidance import CollisionDetector
from core.coordinate_system import CoordinateConverter

logger = logging.getLogger(__name__)

@dataclass
class DroneState:
    """無人機狀態"""
    position: Any  # 3D位置 [x, y, z]
    velocity: Any  # 速度向量
    trajectory: Any  # 軌跡點序列
    current_waypoint: int = 0
    status: str = "ready"  # ready, flying, hovering, landing, landed

class GPUDroneSimulator:
    """GPU加速的無人機群模擬器"""
    
    def __init__(self, parent=None):
        self.parent = parent
        self.xp = get_array_module()  # numpy或cupy
        
        # 模擬狀態
        self.drones = {}  # Dict[str, DroneState]
        self.drone_count = 0
        self.simulation_time = 0.0
        self.is_running = False
        self.is_paused = False
        
        # GPU優化的數據結構
        self.position_matrix = None  # 形狀: (n_drones, 3)
        self.velocity_matrix = None  # 形狀: (n_drones, 3) 
        self.trajectory_batch = None  # 形狀: (n_drones, max_waypoints, 3)
        self.collision_matrix = None  # 形狀: (n_drones, n_drones)
        
        # 效能優化參數
        self.batch_size = settings.gpu.batch_size
        self.update_interval = settings.performance.update_interval
        self.max_trajectory_points = settings.performance.max_trajectory_points
        
        # 視覺化
        self.fig = None
        self.ax = None
        self.canvas = None
        self.animation = None
        
        # 效能監控
        self.frame_times = []
        self.gpu_memory_usage = []
        
        # 碰撞檢測器
        self.collision_detector = CollisionDetector()
        self.coordinate_converter = CoordinateConverter()
        
        logger.info(f"🚀 GPU加速模擬器初始化完成 (後端: {('GPU' if is_gpu_enabled() else 'CPU')})")

    @gpu_accelerated()
    def load_mission_data(self, mission_data: Dict[str, List[Dict]]):
        """載入任務資料並轉換為GPU格式"""
        logger.info("📁 載入任務資料...")
        
        self.drones.clear()
        drone_names = list(mission_data.keys())
        self.drone_count = len(drone_names)
        
        if self.drone_count == 0:
            logger.warning("⚠️ 沒有載入任何無人機資料")
            return
        
        # 預處理軌跡資料
        max_waypoints = max(len(waypoints) for waypoints in mission_data.values())
        
        # 初始化GPU數據結構
        self._initialize_gpu_arrays(self.drone_count, max_waypoints)
        
        # 處理每架無人機的資料
        for i, (drone_name, waypoints) in enumerate(mission_data.items()):
            # 轉換座標
            positions = self._convert_waypoints_to_positions(waypoints)
            
            # 創建無人機狀態
            drone_state = DroneState(
                position=asarray([positions[0][0], positions[0][1], positions[0][2]]),
                velocity=asarray([0.0, 0.0, 0.0]),
                trajectory=asarray(positions)
            )
            
            self.drones[drone_name] = drone_state
            
            # 填充GPU陣列
            self.position_matrix[i] = drone_state.position
            self.velocity_matrix[i] = drone_state.velocity
            
            # 軌跡資料（填充不足的部分）
            traj_len = len(positions)
            self.trajectory_batch[i, :traj_len] = asarray(positions)
            if traj_len < max_waypoints:
                # 用最後一個點填充
                self.trajectory_batch[i, traj_len:] = asarray(positions[-1])
        
        if is_gpu_enabled():
            synchronize()
        
        logger.info(f"✅ 已載入 {self.drone_count} 架無人機，最大航點數: {max_waypoints}")

    def _initialize_gpu_arrays(self, n_drones: int, max_waypoints: int):
        """初始化GPU陣列"""
        logger.debug("🔧 初始化GPU陣列結構...")
        
        # 位置和速度矩陣
        self.position_matrix = self.xp.zeros((n_drones, 3), dtype=self.xp.float32)
        self.velocity_matrix = self.xp.zeros((n_drones, 3), dtype=self.xp.float32)
        
        # 軌跡批次處理
        self.trajectory_batch = self.xp.zeros((n_drones, max_waypoints, 3), dtype=self.xp.float32)
        
        # 碰撞檢測矩陣
        self.collision_matrix = self.xp.zeros((n_drones, n_drones), dtype=self.xp.float32)
        
        if is_gpu_enabled():
            # 預分配GPU記憶體池
            logger.debug(f"💾 GPU記憶體分配: 位置矩陣 {self.position_matrix.nbytes/1024:.1f}KB")

    @gpu_accelerated()
    def _convert_waypoints_to_positions(self, waypoints: List[Dict]) -> List[List[float]]:
        """批次轉換航點座標"""
        positions = []
        
        # 提取座標資料
        lats = [wp.get('lat', wp.get('latitude', 0)) for wp in waypoints]
        lons = [wp.get('lon', wp.get('longitude', 0)) for wp in waypoints] 
        alts = [wp.get('alt', wp.get('altitude', 0)) for wp in waypoints]
        
        # 批次座標轉換（GPU加速）
        if len(lats) > 0:
            # 轉換為GPU陣列進行批次處理
            lat_array = asarray(lats)
            lon_array = asarray(lons) 
            alt_array = asarray(alts)
            
            # 座標轉換（在CoordinateConverter中實現GPU加速）
            positions_array = self.coordinate_converter.batch_convert_to_meters(
                lat_array, lon_array, alt_array
            )
            
            # 轉回CPU進行進一步處理
            positions = to_cpu(positions_array).tolist()
        
        return positions

    @gpu_accelerated()
    def update_simulation(self, dt: float):
        """更新模擬狀態（GPU並行處理）"""
        if not self.is_running or self.is_paused:
            return
        
        start_time = time.perf_counter()
        
        # 更新模擬時間
        self.simulation_time += dt
        
        # GPU並行更新所有無人機位置
        self._update_drone_positions_gpu(dt)
        
        # 碰撞檢測（GPU加速）
        if settings.gpu.accelerate_collision_detection:
            collision_events = self._detect_collisions_gpu()
            if collision_events:
                self._handle_collisions(collision_events)
        
        # 同步GPU操作
        if is_gpu_enabled():
            synchronize()
        
        # 記錄效能
        frame_time = time.perf_counter() - start_time
        self.frame_times.append(frame_time)
        
        # 限制記錄長度
        if len(self.frame_times) > 1000:
            self.frame_times = self.frame_times[-500:]

    @gpu_accelerated()
    def _update_drone_positions_gpu(self, dt: float):
        """GPU並行更新無人機位置"""
        if self.drone_count == 0:
            return
        
        # 批次處理軌跡跟蹤
        drone_names = list(self.drones.keys())
        
        for i, drone_name in enumerate(drone_names):
            drone = self.drones[drone_name]
            
            if drone.current_waypoint < len(to_cpu(drone.trajectory)):
                # 目標位置
                target_pos = drone.trajectory[drone.current_waypoint]
                current_pos = self.position_matrix[i]
                
                # 計算移動方向和距離
                direction = target_pos - current_pos
                distance = self.xp.linalg.norm(direction)
                
                # 移動邏輯
                speed = 5.0  # 5 m/s
                move_distance = speed * dt
                
                if distance < 0.5:  # 到達航點
                    self.position_matrix[i] = target_pos
                    drone.current_waypoint += 1
                else:  # 朝目標移動
                    normalized_direction = direction / distance
                    move_vector = normalized_direction * min(move_distance, distance)
                    self.position_matrix[i] = current_pos + move_vector
                
                # 更新無人機狀態
                drone.position = self.position_matrix[i].copy()

    @gpu_accelerated()
    def _detect_collisions_gpu(self) -> List[Tuple[str, str, float]]:
        """GPU加速碰撞檢測"""
        if self.drone_count < 2:
            return []
        
        # 計算距離矩陣
        distances = MathOps.distance_matrix(self.position_matrix, self.position_matrix)
        
        # 找出小於安全距離的無人機對
        safety_distance = settings.safety.safety_distance
        collision_mask = (distances < safety_distance) & (distances > 0)
        
        # 提取碰撞事件
        collision_events = []
        if self.xp.any(collision_mask):
            # 轉回CPU處理結果
            collision_indices = to_cpu(self.xp.where(collision_mask))
            distances_cpu = to_cpu(distances)
            drone_names = list(self.drones.keys())
            
            for i in range(len(collision_indices[0])):
                idx1, idx2 = collision_indices[0][i], collision_indices[1][i]
                if idx1 < idx2:  # 避免重複
                    distance = distances_cpu[idx1, idx2]
                    collision_events.append((
                        drone_names[idx1], 
                        drone_names[idx2], 
                        distance
                    ))
        
        return collision_events

    def _handle_collisions(self, collision_events: List[Tuple[str, str, float]]):
        """處理碰撞事件"""
        for drone1_name, drone2_name, distance in collision_events:
            # 記錄碰撞事件
            self.collision_detector.log_collision_event(
                drone1_name, drone2_name, distance,
                self.drones[drone1_name].position,
                self.drones[drone2_name].position,
                self.simulation_time
            )
            
            logger.warning(f"⚠️ 碰撞警告: {drone1_name} 與 {drone2_name} 距離 {distance:.2f}m")

    def setup_visualization(self, parent_frame):
        """設置視覺化"""
        logger.info("🎨 設置視覺化...")
        
        # 創建matplotlib圖表
        self.fig = plt.figure(figsize=(12, 8), dpi=settings.visualization.dpi)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 設置軸標籤
        self.ax.set_xlabel('東 (公尺)')
        self.ax.set_ylabel('北 (公尺)')
        self.ax.set_zlabel('高度 (公尺)')
        self.ax.set_title('無人機群模擬 (GPU加速)')
        
        # 創建tkinter畫布
        self.canvas = FigureCanvasTkAgg(self.fig, parent_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 設置動畫（避免快取警告）
        self.animation = animation.FuncAnimation(
            self.fig, 
            self._update_plot, 
            interval=int(self.update_interval * 1000),
            blit=False,
            cache_frame_data=False  # 解決你提到的警告
        )

    def _update_plot(self, frame):
        """更新繪圖"""
        if not self.is_running or self.drone_count == 0:
            return
        
        self.ax.clear()
        
        # 獲取當前位置（轉回CPU用於繪圖）
        if self.position_matrix is not None:
            positions_cpu = to_cpu(self.position_matrix)
            
            # 繪製無人機
            for i, drone_name in enumerate(self.drones.keys()):
                pos = positions_cpu[i]
                self.ax.scatter(pos[0], pos[1], pos[2], 
                              s=100, label=drone_name, alpha=0.8)
                
                # 繪製軌跡
                drone = self.drones[drone_name]
                if hasattr(drone, 'trajectory'):
                    traj_cpu = to_cpu(drone.trajectory)
                    if len(traj_cpu) > 0:
                        self.ax.plot(traj_cpu[:, 0], traj_cpu[:, 1], traj_cpu[:, 2], 
                                   alpha=0.6, linewidth=1)
        
        # 設置視角
        self.ax.legend()
        self.ax.set_xlabel('東 (公尺)')
        self.ax.set_ylabel('北 (公尺)')
        self.ax.set_zlabel('高度 (公尺)')
        
        # 顯示效能資訊
        if self.frame_times:
            avg_fps = 1.0 / (np.mean(self.frame_times[-100:]) + 1e-6)
            backend = "GPU" if is_gpu_enabled() else "CPU"
            self.ax.text2D(0.02, 0.98, f'{backend} | FPS: {avg_fps:.1f}', 
                          transform=self.ax.transAxes)

    def start_simulation(self):
        """開始模擬"""
        if self.drone_count == 0:
            messagebox.showwarning("警告", "請先載入任務資料")
            return
        
        self.is_running = True
        self.is_paused = False
        self.simulation_time = 0.0
        
        logger.info("▶️ 開始模擬...")
        
        # 啟動更新循環
        self._simulation_loop()

    def _simulation_loop(self):
        """模擬主循環"""
        if self.is_running and not self.is_paused:
            self.update_simulation(self.update_interval)
            
            # 安排下次更新
            if self.parent:
                self.parent.after(int(self.update_interval * 1000), self._simulation_loop)

    def pause_simulation(self):
        """暫停模擬"""
        self.is_paused = not self.is_paused
        logger.info("⏸️ 模擬已暫停" if self.is_paused else "▶️ 模擬已繼續")

    def stop_simulation(self):
        """停止模擬"""
        self.is_running = False
        self.is_paused = False
        logger.info("⏹️ 模擬已停止")

    def reset_simulation(self):
        """重置模擬"""
        self.stop_simulation()
        self.simulation_time = 0.0
        
        # 重置無人機狀態
        for drone in self.drones.values():
            drone.current_waypoint = 0
            if hasattr(drone, 'trajectory') and len(to_cpu(drone.trajectory)) > 0:
                drone.position = asarray(to_cpu(drone.trajectory)[0])
        
        # 重置GPU陣列
        if self.position_matrix is not None:
            for i, drone in enumerate(self.drones.values()):
                self.position_matrix[i] = drone.position
        
        logger.info("🔄 模擬已重置")

    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取效能統計"""
        stats = {
            'backend': 'GPU' if is_gpu_enabled() else 'CPU',
            'drone_count': self.drone_count,
            'simulation_time': self.simulation_time,
            'avg_frame_time': np.mean(self.frame_times[-100:]) if self.frame_times else 0,
            'fps': 1.0 / (np.mean(self.frame_times[-100:]) + 1e-6) if self.frame_times else 0,
        }
        
        if is_gpu_enabled():
            from utils.gpu_utils import compute_manager
            memory_info = compute_manager.get_memory_info()
            stats.update({
                'gpu_memory_used': memory_info['used_bytes'] / 1024**2,  # MB
                'gpu_memory_total': memory_info['total_bytes'] / 1024**2,  # MB
            })
        
        return stats

    def cleanup(self):
        """清理資源"""
        logger.info("🧹 清理模擬器資源...")
        
        self.stop_simulation()
        
        if self.animation:
            self.animation.event_source.stop()
        
        # 清理GPU記憶體
        if is_gpu_enabled():
            from utils.gpu_utils import compute_manager
            if hasattr(compute_manager, '_cupy'):
                compute_manager._cupy.get_default_memory_pool().free_all_blocks()
        
        logger.info("✅ 資源清理完成")