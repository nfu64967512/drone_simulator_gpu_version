"""
進階無人機群飛模擬器主類別 - GPU加速版本
整合所有子系統，提供完整的模擬功能
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

from core.coordinate_system import EarthCoordinateSystem, create_coordinate_system
from core.collision_avoidance import CollisionAvoidanceSystem, create_collision_system
from core.trajectory_calculator import GPUTrajectoryCalculator, create_trajectory_calculator, FlightPhase
from utils.qgc_handlers import QGCWaypointGenerator, parse_mission_file, MissionFileExporter
from utils.gpu_utils import get_array_module, GPUMemoryManager
from config.settings import SimulationConfig, SafetyConfig, FlightConfig

logger = logging.getLogger(__name__)

@dataclass
class DroneState:
    """無人機狀態數據結構"""
    drone_id: str
    waypoints: List[Dict]
    trajectory: List[Any]
    color: str
    takeoff_position: Tuple[float, float]
    phase: FlightPhase
    loiter_delays: List[Dict]
    current_position: Optional[Dict]
    file_path: Optional[str] = None
    
class PerformanceMonitor:
    """性能監控器"""
    
    def __init__(self):
        self.frame_times = []
        self.last_update = time.time()
        self.fps = 0.0
        self.gpu_memory_usage = 0.0
        self.collision_check_time = 0.0
        
    def update_frame_time(self):
        """更新幀時間"""
        current_time = time.time()
        if self.last_update > 0:
            frame_time = current_time - self.last_update
            self.frame_times.append(frame_time)
            
            # 保持最近100幀的記錄
            if len(self.frame_times) > 100:
                self.frame_times.pop(0)
            
            # 計算FPS
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        self.last_update = current_time
    
    def update_gpu_memory(self):
        """更新GPU記憶體使用情況"""
        if GPU_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                self.gpu_memory_usage = mempool.used_bytes() / (1024**2)  # MB
            except:
                self.gpu_memory_usage = 0.0


class AdvancedDroneSimulator:
    """
    進階無人機群飛模擬器
    整合GPU加速、碰撞檢測、軌跡計算等所有功能
    """
    
    def __init__(self, config: SimulationConfig):
        """
        初始化模擬器
        
        Args:
            config: 模擬配置
        """
        self.config = config
        self.use_gpu = config.backend.use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        
        # 核心系統初始化
        self._initialize_core_systems()
        
        # 數據存儲
        self.drones: Dict[str, DroneState] = {}
        self.modified_missions: Dict[str, List[str]] = {}
        
        # 模擬狀態
        self.current_time = 0.0
        self.max_time = 0.0
        self.time_scale = 1.0
        self.is_playing = False
        self.is_paused = False
        
        # 性能優化
        self.last_collision_check = 0.0
        self.update_interval = 33  # ~30fps
        self.performance_monitor = PerformanceMonitor()
        
        # GPU記憶體管理
        if self.use_gpu:
            self.memory_manager = GPUMemoryManager()
        
        logger.info(f"進階無人機模擬器初始化完成: GPU={'啟用' if self.use_gpu else '禁用'}")
        
    def _initialize_core_systems(self):
        """初始化核心子系統"""
        # 坐標系統
        self.coordinate_system = create_coordinate_system(use_gpu=self.use_gpu)
        
        # 碰撞避免系統  
        self.collision_system = create_collision_system(
            safety_distance=self.config.safety.safety_distance,
            warning_distance=self.config.safety.warning_distance,
            critical_distance=self.config.safety.critical_distance,
            use_gpu=self.use_gpu
        )
        
        # 軌跡計算器（稍後初始化，需要坐標系統先設定原點）
        self.trajectory_calculator = None
        
        # QGC檔案處理器
        self.qgc_generator = QGCWaypointGenerator()
        self.mission_exporter = MissionFileExporter()
        
        logger.info("核心子系統初始化完成")
    
    def load_mission_files(self, file_paths: List[str], file_type: str = "auto") -> bool:
        """
        載入任務檔案
        
        Args:
            file_paths: 檔案路徑列表
            file_type: 檔案類型 ("qgc", "csv", "auto")
            
        Returns:
            是否載入成功
        """
        self.drones.clear()
        self.modified_missions.clear()
        
        colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', 
                 '#FF44FF', '#44FFFF', '#FFAA44', '#AA44FF']
        
        loaded_count = 0
        
        for i, file_path in enumerate(file_paths[:8]):  # 最多8架無人機
            try:
                drone_id = f"Drone_{i+1}"
                
                # 解析檔案
                waypoints = parse_mission_file(file_path)
                
                if not waypoints:
                    logger.warning(f"檔案 {file_path} 沒有有效航點")
                    continue
                
                # 設定第一個檔案的原點
                if i == 0:
                    self.coordinate_system.set_origin(waypoints[0]['lat'], waypoints[0]['lon'])
                    
                    # 初始化軌跡計算器
                    self.trajectory_calculator = create_trajectory_calculator(
                        self.coordinate_system,
                        cruise_speed=self.config.flight.cruise_speed,
                        takeoff_altitude=self.config.flight.takeoff_altitude,
                        use_gpu=self.use_gpu
                    )
                
                # 計算軌跡
                trajectory = self.trajectory_calculator.calculate_complete_trajectory(
                    waypoints, drone_id
                )
                
                # 創建無人機狀態
                drone_state = DroneState(
                    drone_id=drone_id,
                    waypoints=waypoints,
                    trajectory=trajectory,
                    color=colors[i % len(colors)],
                    takeoff_position=(waypoints[0]['lat'], waypoints[0]['lon']),
                    phase=FlightPhase.TAXI,
                    loiter_delays=[],
                    current_position=None,
                    file_path=file_path
                )
                
                self.drones[drone_id] = drone_state
                loaded_count += 1
                
                logger.info(f"載入 {drone_id}: {len(waypoints)} 個航點，"
                           f"軌跡長度 {len(trajectory)} 點")
                
            except Exception as e:
                logger.error(f"載入檔案 {file_path} 失敗: {e}")
                continue
        
        if loaded_count > 0:
            self._calculate_max_time()
            self._analyze_trajectory_conflicts()
            
            logger.info(f"成功載入 {loaded_count} 架無人機")
            return True
        else:
            logger.warning("未成功載入任何無人機數據")
            return False
    
    def create_test_mission(self, formation_type: str = "square", num_drones: int = 4) -> bool:
        """
        創建測試任務
        
        Args:
            formation_type: 編隊類型 ("square", "diamond", "line")
            num_drones: 無人機數量
            
        Returns:
            是否創建成功
        """
        self.drones.clear()
        self.modified_missions.clear()
        
        # 設定基準座標（台灣某地）
        base_lat, base_lon = 24.0, 121.0
        self.coordinate_system.set_origin(base_lat, base_lon)
        
        # 初始化軌跡計算器
        self.trajectory_calculator = create_trajectory_calculator(
            self.coordinate_system,
            cruise_speed=self.config.flight.cruise_speed,
            takeoff_altitude=self.config.flight.takeoff_altitude,
            use_gpu=self.use_gpu
        )
        
        # 生成起飛編隊位置
        takeoff_positions = self.coordinate_system.create_formation_pattern(
            center_lat=base_lat, 
            center_lon=base_lon + 0.001,  # 東偏移約100米
            formation_type=formation_type,
            spacing=self.config.flight.formation_spacing,
            num_drones=num_drones
        )
        
        colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44', 
                 '#FF44FF', '#44FFFF', '#FFAA44', '#AA44FF']
        
        # 任務區域偏移量
        mission_offsets = [
            (-150, -100), (150, -100), (-150, 100), (150, 100),  # 四個角落
            (-150, 0), (150, 0), (0, -100), (0, 100)             # 四個邊
        ]
        
        for i in range(min(num_drones, len(takeoff_positions))):
            drone_id = f"Drone_{i+1}"
            takeoff_lat, takeoff_lon = takeoff_positions[i]
            
            # 創建任務航點
            waypoints = self._generate_test_waypoints(
                takeoff_lat, takeoff_lon, mission_offsets[i % len(mission_offsets)]
            )
            
            # 計算軌跡
            trajectory = self.trajectory_calculator.calculate_complete_trajectory(
                waypoints, drone_id
            )
            
            # 創建無人機狀態
            drone_state = DroneState(
                drone_id=drone_id,
                waypoints=waypoints,
                trajectory=trajectory,
                color=colors[i % len(colors)],
                takeoff_position=(takeoff_lat, takeoff_lon),
                phase=FlightPhase.TAXI,
                loiter_delays=[],
                current_position=None
            )
            
            self.drones[drone_id] = drone_state
        
        self._calculate_max_time()
        self._analyze_trajectory_conflicts()
        
        logger.info(f"創建測試任務完成: {len(self.drones)} 架無人機，{formation_type} 編隊")
        return True
    
    def _generate_test_waypoints(self, takeoff_lat: float, takeoff_lon: float, 
                                offset: Tuple[float, float]) -> List[Dict]:
        """生成測試航點"""
        waypoints = []
        
        # HOME點
        waypoints.append({
            'lat': takeoff_lat,
            'lon': takeoff_lon, 
            'alt': 0,
            'cmd': 179
        })
        
        # 轉換偏移到地理坐標
        base_x, base_y = self.coordinate_system.lat_lon_to_meters(takeoff_lat, takeoff_lon)
        offset_x, offset_y = offset
        
        # 生成矩形任務軌跡
        mission_points = [
            (base_x + offset_x, base_y + offset_y, 15),
            (base_x + offset_x + 100, base_y + offset_y, 15),
            (base_x + offset_x + 100, base_y + offset_y + 80, 15),
            (base_x + offset_x, base_y + offset_y + 80, 15),
            (base_x + offset_x, base_y + offset_y, 15)
        ]
        
        for mx, my, mz in mission_points:
            mlat, mlon = self.coordinate_system.meters_to_lat_lon(mx, my)
            waypoints.append({
                'lat': mlat,
                'lon': mlon,
                'alt': mz,
                'cmd': 16
            })
        
        return waypoints
    
    def _analyze_trajectory_conflicts(self):
        """分析軌跡衝突並生成修正任務"""
        if len(self.drones) < 2:
            return
        
        logger.info("開始軌跡衝突分析...")
        
        # 準備軌跡數據
        drones_data = {
            drone_id: {'trajectory': drone_state.trajectory}
            for drone_id, drone_state in self.drones.items()
        }
        
        # 執行衝突分析
        conflicts = self.collision_system.trajectory_analyzer.analyze_trajectory_conflicts(drones_data)
        
        if conflicts:
            logger.info(f"發現 {len(conflicts)} 個軌跡衝突，生成修正任務...")
            
            # 為每架受影響的無人機生成修正任務
            for drone_id, drone_state in self.drones.items():
                relevant_conflicts = [c for c in conflicts if c.waiting_drone == drone_id]
                
                if relevant_conflicts:
                    # 生成修正後的QGC任務
                    modified_mission = self.qgc_generator.generate_mission_with_conflicts(
                        drone_id, drone_state.waypoints, relevant_conflicts
                    )
                    
                    self.modified_missions[drone_id] = modified_mission
                    
                    logger.info(f"為 {drone_id} 生成修正任務，包含避讓指令")
        
        else:
            logger.info("未發現軌跡衝突")
    
    def _calculate_max_time(self):
        """計算最大模擬時間"""
        self.max_time = 0.0
        
        for drone_state in self.drones.values():
            if drone_state.trajectory:
                # 基礎軌跡時間
                base_time = drone_state.trajectory[-1].time
                
                # 加上LOITER延遲
                total_loiter = sum(delay.get('duration', 0) for delay in drone_state.loiter_delays)
                
                total_time = base_time + total_loiter
                self.max_time = max(self.max_time, total_time)
        
        logger.info(f"最大模擬時間: {self.max_time:.1f} 秒")
    
    def get_drone_position_at_time(self, drone_id: str, time: float) -> Optional[Dict]:
        """
        獲取指定時間的無人機位置
        
        Args:
            drone_id: 無人機ID
            time: 時間
            
        Returns:
            位置字典或None
        """
        if drone_id not in self.drones:
            return None
        
        drone_state = self.drones[drone_id]
        trajectory = drone_state.trajectory
        
        if not trajectory:
            return None
        
        # 應用LOITER延遲
        effective_time = time
        for delay in drone_state.loiter_delays:
            if time >= delay.get('start_time', 0):
                effective_time = max(delay.get('start_time', 0), 
                                   time - delay.get('duration', 0))
                break
        
        # 使用軌跡插值器
        if hasattr(self, 'trajectory_interpolator'):
            return self.trajectory_interpolator.interpolate_position(trajectory, effective_time)
        
        # 簡化版插值
        return self._simple_interpolate_position(trajectory, effective_time)
    
    def _simple_interpolate_position(self, trajectory: List[Any], time: float) -> Optional[Dict]:
        """簡化版位置插值"""
        if not trajectory:
            return None
        
        # 邊界條件
        if time >= trajectory[-1].time:
            last_point = trajectory[-1]
            return {
                'x': last_point.x, 'y': last_point.y, 'z': last_point.z,
                'time': time, 'phase': last_point.phase
            }
        
        if time <= trajectory[0].time:
            first_point = trajectory[0]
            return {
                'x': first_point.x, 'y': first_point.y, 'z': first_point.z,
                'time': time, 'phase': first_point.phase
            }
        
        # 線性插值
        for i in range(len(trajectory) - 1):
            p1, p2 = trajectory[i], trajectory[i + 1]
            
            if p1.time <= time <= p2.time:
                if p2.time - p1.time == 0:
                    return {
                        'x': p1.x, 'y': p1.y, 'z': p1.z,
                        'time': time, 'phase': p1.phase
                    }
                
                ratio = (time - p1.time) / (p2.time - p1.time)
                
                return {
                    'x': p1.x + ratio * (p2.x - p1.x),
                    'y': p1.y + ratio * (p2.y - p1.y),
                    'z': p1.z + ratio * (p2.z - p1.z),
                    'time': time,
                    'phase': p1.phase
                }
        
        return None
    
    def update_simulation(self, dt: float) -> Dict[str, Any]:
        """
        更新模擬狀態
        
        Args:
            dt: 時間步長
            
        Returns:
            更新結果字典
        """
        if not self.is_playing or self.is_paused:
            return {'updated': False}
        
        # 更新時間
        self.current_time += dt * self.time_scale
        
        if self.current_time > self.max_time:
            self.current_time = self.max_time
            self.is_playing = False
        
        # 獲取當前所有無人機位置
        current_positions = {}
        for drone_id in self.drones:
            pos = self.get_drone_position_at_time(drone_id, self.current_time)
            if pos:
                current_positions[drone_id] = pos
        
        # 碰撞檢測（每0.1秒檢查一次）
        collision_warnings = []
        new_loiters = {}
        
        if self.current_time - self.last_collision_check >= self.config.safety.collision_check_interval:
            collision_start = time.time()
            
            collision_warnings, new_loiters = self.collision_system.check_realtime_collisions(
                current_positions, self.current_time
            )
            
            # 應用新的LOITER延遲
            for drone_id, loiter_time in new_loiters.items():
                if drone_id in self.drones:
                    self.drones[drone_id].loiter_delays.append({
                        'start_time': self.current_time,
                        'duration': loiter_time
                    })
            
            self.last_collision_check = self.current_time
            self.performance_monitor.collision_check_time = time.time() - collision_start
        
        # 更新性能監控
        self.performance_monitor.update_frame_time()
        self.performance_monitor.update_gpu_memory()
        
        # 更新無人機當前位置
        for drone_id, pos in current_positions.items():
            self.drones[drone_id].current_position = pos
        
        return {
            'updated': True,
            'current_time': self.current_time,
            'positions': current_positions,
            'collision_warnings': collision_warnings,
            'new_loiters': new_loiters,
            'fps': self.performance_monitor.fps,
            'gpu_memory': self.performance_monitor.gpu_memory_usage
        }
    
    def export_modified_missions(self, export_directory: str) -> Dict[str, str]:
        """
        導出修正後的任務檔案
        
        Args:
            export_directory: 導出目錄
            
        Returns:
            導出結果 {drone_id: file_path}
        """
        if not self.modified_missions:
            logger.warning("沒有修正後的任務需要導出")
            return {}
        
        return self.mission_exporter.export_modified_missions(
            self.modified_missions, export_directory
        )
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """獲取模擬狀態信息"""
        return {
            'is_playing': self.is_playing,
            'is_paused': self.is_paused,
            'current_time': self.current_time,
            'max_time': self.max_time,
            'time_scale': self.time_scale,
            'num_drones': len(self.drones),
            'use_gpu': self.use_gpu,
            'gpu_available': GPU_AVAILABLE,
            'fps': self.performance_monitor.fps,
            'gpu_memory_mb': self.performance_monitor.gpu_memory_usage,
            'collision_warnings': len(self.collision_system.collision_warnings),
            'modified_missions': len(self.modified_missions)
        }
    
    def play_simulation(self):
        """開始播放模擬"""
        self.is_playing = True
        self.is_paused = False
        logger.info("模擬開始播放")
    
    def pause_simulation(self):
        """暫停模擬"""
        self.is_paused = True
        logger.info("模擬已暫停")
    
    def stop_simulation(self):
        """停止模擬"""
        self.is_playing = False
        self.is_paused = False
        logger.info("模擬已停止")
    
    def reset_simulation(self):
        """重置模擬"""
        self.stop_simulation()
        self.current_time = 0.0
        self.last_collision_check = 0.0
        
        # 清除LOITER延遲
        for drone_state in self.drones.values():
            drone_state.loiter_delays = []
            drone_state.current_position = None
        
        # 清除修正任務
        self.modified_missions.clear()
        
        logger.info("模擬已重置")
    
    def set_time_scale(self, scale: float):
        """設置時間縮放比例"""
        self.time_scale = max(0.1, min(scale, 10.0))  # 限制在0.1x到10x之間
        logger.info(f"時間縮放設置為: {self.time_scale}x")
    
    def seek_to_time(self, time: float):
        """跳轉到指定時間"""
        if not self.is_playing:  # 只在暫停時允許跳轉
            self.current_time = max(0, min(time, self.max_time))
            logger.info(f"跳轉到時間: {self.current_time:.1f}s")
    
    def optimize_performance(self):
        """優化系統性能"""
        if self.use_gpu:
            # 清理GPU記憶體
            self.coordinate_system.optimize_memory_usage()
            self.collision_system.optimize_performance()
            
            if hasattr(self, 'memory_manager'):
                self.memory_manager.cleanup()
            
            logger.info("GPU性能優化完成")
    
    def cleanup(self):
        """清理資源"""
        self.stop_simulation()
        self.optimize_performance()
        logger.info("模擬器資源清理完成")
