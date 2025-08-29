#!/usr/bin/env python3
"""
無人機群模擬器主視窗 - 完整功能版本
整合GPU/CPU後端支援，提供完整3D軌跡模擬功能
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import time
import logging
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.animation as animation
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import threading
import os
from datetime import datetime
import copy
import math

# 導入專案核心模組
try:
    from config.settings import get_compute_backend_info
    from utils.gpu_utils import compute_manager, MathOps
except ImportError as e:
    print(f"警告: 無法導入GPU模組: {e}")
    compute_manager = None

# 獲取日誌記錄器
logger = logging.getLogger(__name__)

# 地球座標常數
EARTH_RADIUS_KM = 6371.0
METERS_PER_DEGREE_LAT = 111111.0

@dataclass
class SafetyConfig:
    """安全配置"""
    safety_distance: float = 5.0
    warning_distance: float = 8.0
    critical_distance: float = 3.0
    collision_check_interval: float = 0.1

@dataclass
class TakeoffConfig:
    """起飛配置"""
    formation_spacing: float = 3.0
    takeoff_altitude: float = 10.0
    hover_time: float = 2.0
    east_offset: float = 50.0

class FlightPhase:
    """飛行階段"""
    TAXI = "taxi"
    TAKEOFF = "takeoff"
    HOVER = "hover"
    AUTO = "auto"
    LOITER = "loiter"
    LANDING = "landing"

class EarthCoordinateSystem:
    """地球座標系統 - 支援GPU加速"""
    
    def __init__(self):
        self.origin_lat: Optional[float] = None
        self.origin_lon: Optional[float] = None
        
    def set_origin(self, lat: float, lon: float):
        """設置座標原點"""
        self.origin_lat = lat
        self.origin_lon = lon
        logger.info(f"設置座標原點: ({lat:.6f}, {lon:.6f})")
        
    def lat_lon_to_meters(self, lat: float, lon: float) -> Tuple[float, float]:
        """將經緯度轉換為米制座標（考慮地球曲率）- 可GPU加速"""
        if self.origin_lat is None or self.origin_lon is None:
            return 0.0, 0.0
            
        # 緯度轉換（1度約111.111公里）
        y = (lat - self.origin_lat) * METERS_PER_DEGREE_LAT
        
        # 經度轉換（考慮緯度修正）
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
        x = (lon - self.origin_lon) * meters_per_degree_lon
        
        return x, y
    
    def meters_to_lat_lon(self, x: float, y: float) -> Tuple[float, float]:
        """將米制座標轉換為經緯度"""
        if self.origin_lat is None or self.origin_lon is None:
            return 0.0, 0.0
            
        # 緯度轉換
        lat = self.origin_lat + y / METERS_PER_DEGREE_LAT
        
        # 經度轉換
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
        lon = self.origin_lon + x / meters_per_degree_lon
        
        return lat, lon

    def batch_lat_lon_to_meters(self, lats: np.ndarray, lons: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """批次座標轉換 - GPU加速版本"""
        if compute_manager and compute_manager.backend.name == 'GPU':
            try:
                # 使用GPU加速批次轉換
                from utils.gpu_utils import asarray, to_cpu
                
                lats_gpu = asarray(lats)
                lons_gpu = asarray(lons)
                
                # GPU計算
                y_gpu = (lats_gpu - self.origin_lat) * METERS_PER_DEGREE_LAT
                meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
                x_gpu = (lons_gpu - self.origin_lon) * meters_per_degree_lon
                
                return to_cpu(x_gpu), to_cpu(y_gpu)
            except Exception as e:
                logger.warning(f"GPU座標轉換失敗，回退到CPU: {e}")
        
        # CPU版本
        y = (lats - self.origin_lat) * METERS_PER_DEGREE_LAT
        meters_per_degree_lon = METERS_PER_DEGREE_LAT * math.cos(math.radians(self.origin_lat))
        x = (lons - self.origin_lon) * meters_per_degree_lon
        
        return x, y

class CollisionAvoidanceSystem:
    """碰撞避免系統 - GPU加速版本"""
    
    def __init__(self, safety_config: SafetyConfig):
        self.config = safety_config
        self.collision_warnings: List[Dict] = []
        
    def check_collisions_gpu(self, positions: Dict[str, Dict], current_time: float) -> Tuple[List[Dict], Dict[str, float]]:
        """GPU加速碰撞檢測"""
        self.collision_warnings.clear()
        new_loiters = {}
        
        if len(positions) < 2:
            return self.collision_warnings, new_loiters
        
        drone_ids = list(positions.keys())
        position_array = np.array([[pos['x'], pos['y'], pos['z']] for pos in positions.values()])
        
        try:
            if compute_manager and compute_manager.backend.name == 'GPU':
                # 使用GPU計算距離矩陣
                distances = MathOps.distance_matrix(position_array, position_array)
            else:
                # CPU版本距離計算
                distances = self._cpu_distance_matrix(position_array)
            
            # 檢查碰撞
            for i in range(len(drone_ids)):
                for j in range(i + 1, len(drone_ids)):
                    distance = distances[i, j]
                    
                    if distance < self.config.safety_distance:
                        drone1, drone2 = drone_ids[i], drone_ids[j]
                        pos1, pos2 = positions[drone1], positions[drone2]
                        
                        warning = {
                            'drone1': drone1,
                            'drone2': drone2,
                            'distance': distance,
                            'time': current_time,
                            'position': ((pos1['x'] + pos2['x'])/2, 
                                       (pos1['y'] + pos2['y'])/2, 
                                       (pos1['z'] + pos2['z'])/2),
                            'severity': 'critical' if distance < self.config.critical_distance else 'warning'
                        }
                        self.collision_warnings.append(warning)
                        
        except Exception as e:
            logger.error(f"碰撞檢測失敗: {e}")
        
        return self.collision_warnings, new_loiters
    
    def _cpu_distance_matrix(self, positions: np.ndarray) -> np.ndarray:
        """CPU版本距離矩陣計算"""
        n = len(positions)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances[i, j] = distances[j, i] = dist
                
        return distances

class DroneSimulatorApp:
    """無人機模擬器主應用程式 - 完整功能版"""
    
    def __init__(self, root, backend_info=None):
        self.root = root
        self.backend_info = backend_info or {}
        
        # 核心系統
        self.coordinate_system = EarthCoordinateSystem()
        self.safety_config = SafetyConfig()
        self.takeoff_config = TakeoffConfig()
        self.collision_system = CollisionAvoidanceSystem(self.safety_config)
        
        # 數據
        self.drones: Dict[str, Dict] = {}
        self.current_time = 0.0
        self.max_time = 0.0
        self.time_scale = 1.0
        self.is_playing = False
        self.modified_missions: Dict[str, List[str]] = {}
        
        # 性能優化
        self.last_collision_check = 0.0
        self.update_interval = 33  # ~30fps
        
        # 設置UI
        self.setup_window()
        self.create_widgets()
        self.setup_advanced_3d_plot()
        
        # 動畫
        self.animation = None
        self.last_update_time = time.time()
        
    def setup_window(self):
        """設置主視窗"""
        self.root.title("無人機群模擬器 - GPU/CPU 加速完整版")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # 設置關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 設置樣式
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            pass
        
    def create_widgets(self):
        """創建UI元件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左側控制面板
        control_frame = tk.Frame(main_frame, bg='#2d2d2d', width=320)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # 右側3D顯示區域
        self.plot_container = ttk.Frame(main_frame)
        self.plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 創建控制面板
        self.create_control_panel(control_frame)
        
    def create_control_panel(self, parent):
        """創建控制面板"""
        # 標題和後端資訊
        title_frame = tk.Frame(parent, bg='#2d2d2d')
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(title_frame, text="無人機群模擬器", 
                font=('Arial', 16, 'bold'), fg='#00d4aa', bg='#2d2d2d').pack()
        
        # 顯示後端資訊
        if self.backend_info:
            try:
                backend_name = str(self.backend_info.get('backend', 'UNKNOWN')).upper()
                if 'device_id' in self.backend_info and self.backend_info['device_id'] is not None:
                    backend_text = f"計算後端: {backend_name} (GPU {self.backend_info['device_id']})"
                else:
                    backend_text = f"計算後端: {backend_name}"
            except:
                backend_text = "計算後端: 未知"
        else:
            backend_text = "計算後端: CPU"
            
        tk.Label(title_frame, text=backend_text, 
                font=('Arial', 10), fg='#ffffff', bg='#2d2d2d').pack(pady=5)
        
        # 檔案操作
        file_frame = tk.LabelFrame(parent, text="任務檔案", 
                                  fg='white', bg='#2d2d2d', font=('Arial', 11, 'bold'))
        file_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_frame = tk.Frame(file_frame, bg='#2d2d2d')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Button(btn_frame, text="載入 QGC", command=self.load_qgc_files,
                 bg='#28a745', fg='white', font=('Arial', 9, 'bold'), width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="載入 CSV", command=self.load_csv_files,
                 bg='#007bff', fg='white', font=('Arial', 9, 'bold'), width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="測試任務", command=self.create_test_mission,
                 bg='#6f42c1', fg='white', font=('Arial', 9, 'bold'), width=10).pack(side=tk.LEFT, padx=2)
        
        # 播放控制
        play_frame = tk.LabelFrame(parent, text="播放控制", 
                                  fg='white', bg='#2d2d2d', font=('Arial', 11, 'bold'))
        play_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 按鈕行
        btn_row = tk.Frame(play_frame, bg='#2d2d2d')
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_button = tk.Button(btn_row, text="▶", command=self.toggle_play,
                                    bg='#28a745', fg='white', 
                                    font=('Arial', 12, 'bold'), width=4)
        self.play_button.pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_row, text="■", command=self.stop_simulation,
                 bg='#dc3545', fg='white', 
                 font=('Arial', 12, 'bold'), width=4).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_row, text="⟲", command=self.reset_simulation,
                 bg='#ffc107', fg='black', 
                 font=('Arial', 12, 'bold'), width=4).pack(side=tk.LEFT, padx=2)
        
        tk.Button(btn_row, text="💾", command=self.export_modified_missions,
                 bg='#17a2b8', fg='white', 
                 font=('Arial', 12, 'bold'), width=4).pack(side=tk.LEFT, padx=2)
        
        # 時間控制
        self.time_label = tk.Label(play_frame, text="00:00 / 00:00", 
                                  fg='#00d4aa', bg='#2d2d2d', 
                                  font=('Arial', 12, 'bold'))
        self.time_label.pack(pady=5)
        
        self.time_var = tk.DoubleVar()
        self.time_slider = tk.Scale(play_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   variable=self.time_var, command=self.on_time_change,
                                   bg='#2d2d2d', fg='white', 
                                   highlightbackground='#2d2d2d',
                                   troughcolor='#404040', length=280, resolution=0.1)
        self.time_slider.pack(fill=tk.X, padx=5, pady=5)
        
        # 參數控制
        params_frame = tk.Frame(play_frame, bg='#2d2d2d')
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 無人機數量
        tk.Label(params_frame, text="無人機數量:", fg='white', bg='#2d2d2d', 
                font=('Arial', 9)).pack(anchor=tk.W)
        
        self.drone_count_var = tk.StringVar(value="4")
        self.drone_count_label = tk.Label(params_frame, textvariable=self.drone_count_var, 
                                         fg='#00d4aa', bg='#2d2d2d', font=('Arial', 11, 'bold'))
        self.drone_count_label.pack(anchor=tk.W, pady=(0, 5))
        
        # 模擬速度
        tk.Label(params_frame, text="模擬速度:", fg='white', bg='#2d2d2d', 
                font=('Arial', 9)).pack(anchor=tk.W)
        
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = tk.Scale(params_frame, from_=0.1, to=5.0, resolution=0.1,
                              orient=tk.HORIZONTAL, variable=self.speed_var,
                              command=self.on_speed_change,
                              bg='#2d2d2d', fg='white', length=280)
        speed_scale.pack(fill=tk.X, pady=(0, 5))
        
        # 安全距離
        tk.Label(params_frame, text="安全距離 (公尺):", fg='white', bg='#2d2d2d', 
                font=('Arial', 9)).pack(anchor=tk.W)
        
        self.safety_var = tk.DoubleVar(value=5.0)
        safety_scale = tk.Scale(params_frame, from_=2.0, to=15.0, resolution=0.5,
                               orient=tk.HORIZONTAL, variable=self.safety_var,
                               command=self.on_safety_change,
                               bg='#2d2d2d', fg='white', length=280)
        safety_scale.pack(fill=tk.X, pady=(0, 5))
        
        # 碰撞檢測開關
        self.collision_var = tk.BooleanVar(value=True)
        tk.Checkbutton(params_frame, text="啟用碰撞檢測", variable=self.collision_var,
                      fg='white', bg='#2d2d2d', selectcolor='#404040',
                      font=('Arial', 9)).pack(anchor=tk.W, pady=5)
        
        # 狀態資訊
        status_frame = tk.LabelFrame(parent, text="狀態資訊", 
                                    fg='white', bg='#2d2d2d', font=('Arial', 11, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 創建文字顯示區域和滾動條
        text_container = tk.Frame(status_frame, bg='#2d2d2d')
        text_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(text_container, bg='#1a1a1a', fg='#00d4aa',
                                  font=('Consolas', 10), height=8, wrap=tk.WORD)
        status_scroll = ttk.Scrollbar(text_container, orient="vertical", 
                                     command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 碰撞警告
        warning_frame = tk.LabelFrame(parent, text="碰撞警告", 
                                     fg='white', bg='#2d2d2d', font=('Arial', 11, 'bold'))
        warning_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.warning_text = tk.Text(warning_frame, height=4, bg='#1a1a1a', fg='#ff5722',
                                   font=('Consolas', 9), wrap=tk.WORD)
        self.warning_text.pack(fill=tk.X, padx=5, pady=5)
        
    def setup_advanced_3d_plot(self):
        """設置先進3D繪圖"""
        # 使用深色主題
        plt.style.use('dark_background')
        
        # 高解析度圖形
        self.fig = plt.figure(figsize=(12, 9), facecolor='#1e1e1e', dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 背景設置
        self.ax.set_facecolor('#1e1e1e')
        self.fig.patch.set_facecolor('#1e1e1e')
        
        # 軸樣式
        self.setup_axis_style()
        
        # 創建畫布容器
        canvas_frame = tk.Frame(self.plot_container, bg='#1e1e1e')
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 畫布
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        
        # 自訂工具欄
        self.setup_custom_toolbar(canvas_frame)
        
        # 打包畫布
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 啟用滑鼠縮放
        self.enable_mouse_zoom()
        
        # 設置初始視角
        self.ax.view_init(elev=30, azim=45)
        
    def setup_axis_style(self):
        """設置軸樣式"""
        self.ax.grid(True, alpha=0.3, color='#404040', linewidth=0.5)
        self.ax.set_xlabel('東向距離 (公尺)', fontsize=11, color='#00d4aa', labelpad=10)
        self.ax.set_ylabel('北向距離 (公尺)', fontsize=11, color='#00d4aa', labelpad=10)
        self.ax.set_zlabel('飛行高度 (公尺)', fontsize=11, color='#00d4aa', labelpad=10)
        self.ax.set_title('無人機群3D軌跡模擬 - GPU/CPU加速版', 
                         fontsize=13, color='#ffffff', pad=15)
        self.ax.tick_params(colors='#888888', labelsize=9)
        
        # 軸面板設置
        for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor('#404040')
            pane.set_alpha(0.1)
    
    def setup_custom_toolbar(self, parent):
        """設置自訂工具欄"""
        toolbar_frame = tk.Frame(parent, bg='#3a3a3a', height=35)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar_frame.pack_propagate(False)
        
        # 標準matplotlib工具欄
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.configure(bg='#3a3a3a')
        
        # 添加自訂按鈕
        custom_frame = tk.Frame(toolbar_frame, bg='#3a3a3a')
        custom_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Button(custom_frame, text="俯視", command=self.set_top_view,
                 bg='#007bff', fg='white', font=('Arial', 8), width=6).pack(side=tk.LEFT, padx=1)
        
        tk.Button(custom_frame, text="側視", command=self.set_side_view,
                 bg='#007bff', fg='white', font=('Arial', 8), width=6).pack(side=tk.LEFT, padx=1)
        
        tk.Button(custom_frame, text="3D視圖", command=self.set_3d_view,
                 bg='#007bff', fg='white', font=('Arial', 8), width=6).pack(side=tk.LEFT, padx=1)
    
    def enable_mouse_zoom(self):
        """啟用滑鼠滾輪縮放"""
        def on_scroll(event):
            if event.inaxes == self.ax:
                # 取得當前軸限制
                xlim = self.ax.get_xlim()
                ylim = self.ax.get_ylim()
                zlim = self.ax.get_zlim()
                
                # 縮放因子
                scale_factor = 1.1 if event.button == 'down' else 1/1.1
                
                # 計算新的限制
                x_center = (xlim[0] + xlim[1]) / 2
                y_center = (ylim[0] + ylim[1]) / 2
                z_center = (zlim[0] + zlim[1]) / 2
                
                x_range = (xlim[1] - xlim[0]) * scale_factor / 2
                y_range = (ylim[1] - ylim[0]) * scale_factor / 2
                z_range = (zlim[1] - zlim[0]) * scale_factor / 2
                
                # 設置新的軸限制
                self.ax.set_xlim(x_center - x_range, x_center + x_range)
                self.ax.set_ylim(y_center - y_range, y_center + y_range)
                self.ax.set_zlim(max(0, z_center - z_range), z_center + z_range)
                
                self.canvas.draw_idle()
        
        self.canvas.mpl_connect('scroll_event', on_scroll)
    
    def set_top_view(self):
        """俯視角"""
        self.ax.view_init(elev=90, azim=0)
        self.canvas.draw_idle()
        
    def set_side_view(self):
        """側視角"""
        self.ax.view_init(elev=0, azim=0)
        self.canvas.draw_idle()
        
    def set_3d_view(self):
        """3D視角"""
        self.ax.view_init(elev=30, azim=45)
        self.canvas.draw_idle()
    
    def create_test_mission(self):
        """創建測試任務"""
        self.drones.clear()
        self.modified_missions.clear()
        
        # 設置基準座標（台灣某地）
        base_lat, base_lon = 24.0, 121.0
        self.coordinate_system.set_origin(base_lat, base_lon)
        
        colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44']
        drone_names = ['Alpha', 'Beta', 'Charlie', 'Delta']
        
        for i in range(4):
            drone_id = f"Drone_{i+1}"
            
            # 創建簡單的矩形任務
            waypoints = []
            
            # 起飛點
            takeoff_lat = base_lat + (i * 0.0001)
            takeoff_lon = base_lon + (i * 0.0001)
            
            waypoints.append({
                'lat': takeoff_lat,
                'lon': takeoff_lon,
                'alt': 0,
                'cmd': 179  # HOME
            })
            
            # 任務區域（基於索引分散到不同區域）
            region_offsets = [
                (-0.001, -0.0005),  # 西南
                (0.001, -0.0005),   # 東南
                (-0.001, 0.0005),   # 西北
                (0.001, 0.0005)     # 東北
            ]
            
            offset_lat, offset_lon = region_offsets[i]
            
            # 生成矩形任務軌跡
            mission_points = [
                (base_lat + offset_lat, base_lon + offset_lon, 15),
                (base_lat + offset_lat + 0.0008, base_lon + offset_lon, 15),
                (base_lat + offset_lat + 0.0008, base_lon + offset_lon + 0.0008, 15),
                (base_lat + offset_lat, base_lon + offset_lon + 0.0008, 15),
                (base_lat + offset_lat, base_lon + offset_lon, 15)
            ]
            
            for mlat, mlon, malt in mission_points:
                waypoints.append({
                    'lat': mlat,
                    'lon': mlon,
                    'alt': malt,
                    'cmd': 16
                })
            
            # 計算軌跡
            trajectory = self.calculate_trajectory(waypoints, drone_id)
            
            self.drones[drone_id] = {
                'waypoints': waypoints,
                'trajectory': trajectory,
                'color': colors[i],
                'name': drone_names[i],
                'takeoff_position': (takeoff_lat, takeoff_lon),
                'phase': FlightPhase.TAXI,
                'loiter_delays': [],
                'current_position': None
            }
        
        self.drone_count_var.set(str(len(self.drones)))
        self.calculate_max_time()
        self.update_status_display()
        self.update_3d_plot()
        messagebox.showinfo("任務創建", f"已創建 {len(self.drones)} 架無人機的測試任務")
    
    def calculate_trajectory(self, waypoints: List[Dict], drone_id: str) -> List[Dict]:
        """計算真實軌跡（GPU加速版本）"""
        trajectory = []
        if len(waypoints) < 2:
            return trajectory
            
        total_time = 0.0
        speed = 8.0  # 巡航速度
        
        # 階段1: 地面滑行 (0-2秒)
        home_wp = waypoints[0]
        home_x, home_y = self.coordinate_system.lat_lon_to_meters(home_wp['lat'], home_wp['lon'])
        
        trajectory.append({
            'x': home_x, 'y': home_y, 'z': 0,
            'time': 0.0, 'phase': FlightPhase.TAXI,
            'lat': home_wp['lat'], 'lon': home_wp['lon'], 'alt': 0
        })
        
        # 階段2: 起飛 (2-7秒)
        takeoff_time = 2.0
        climb_duration = 5.0
        
        for t in np.linspace(takeoff_time, takeoff_time + climb_duration, 10):
            progress = (t - takeoff_time) / climb_duration
            altitude = progress * self.takeoff_config.takeoff_altitude
            
            trajectory.append({
                'x': home_x, 'y': home_y, 'z': altitude,
                'time': t, 'phase': FlightPhase.TAKEOFF,
                'lat': home_wp['lat'], 'lon': home_wp['lon'], 'alt': altitude
            })
        
        total_time = takeoff_time + climb_duration
        
        # 階段3: 懸停等待 (7-9秒)
        hover_end_time = total_time + self.takeoff_config.hover_time
        
        trajectory.append({
            'x': home_x, 'y': home_y, 'z': self.takeoff_config.takeoff_altitude,
            'time': hover_end_time, 'phase': FlightPhase.HOVER,
            'lat': home_wp['lat'], 'lon': home_wp['lon'], 'alt': self.takeoff_config.takeoff_altitude
        })
        
        total_time = hover_end_time
        
        # 階段4: 自動任務
        prev_x, prev_y, prev_z = home_x, home_y, self.takeoff_config.takeoff_altitude
        
        for wp in waypoints[1:]:
            x, y = self.coordinate_system.lat_lon_to_meters(wp['lat'], wp['lon'])
            z = wp['alt']
            
            # 計算飛行時間
            distance = math.sqrt((x - prev_x)**2 + (y - prev_y)**2 + (z - prev_z)**2)
            flight_time = distance / speed
            total_time += flight_time
            
            trajectory.append({
                'x': x, 'y': y, 'z': z,
                'time': total_time, 'phase': FlightPhase.AUTO,
                'lat': wp['lat'], 'lon': wp['lon'], 'alt': wp['alt']
            })
            
            prev_x, prev_y, prev_z = x, y, z
        
        return trajectory
    
    def get_drone_position_at_time(self, drone_id: str, time: float) -> Optional[Dict]:
        """取得指定時間的無人機位置"""
        if drone_id not in self.drones:
            return None
            
        trajectory = self.drones[drone_id]['trajectory']
        if not trajectory:
            return None
        
        # 邊界條件
        if time >= trajectory[-1]['time']:
            return trajectory[-1]
        if time <= trajectory[0]['time']:
            return trajectory[0]
        
        # 線性插值
        for i in range(len(trajectory) - 1):
            t1, t2 = trajectory[i]['time'], trajectory[i + 1]['time']
            if t1 <= time <= t2:
                if t2 - t1 == 0:
                    return trajectory[i]
                    
                ratio = (time - t1) / (t2 - t1)
                
                result = {
                    'x': trajectory[i]['x'] + ratio * (trajectory[i + 1]['x'] - trajectory[i]['x']),
                    'y': trajectory[i]['y'] + ratio * (trajectory[i + 1]['y'] - trajectory[i]['y']),
                    'z': trajectory[i]['z'] + ratio * (trajectory[i + 1]['z'] - trajectory[i]['z']),
                    'time': time,
                    'phase': trajectory[i]['phase']
                }
                return result
        
        return None
    
    def update_3d_plot(self):
        """更新3D繪圖 - GPU加速版本"""
        self.ax.clear()
        self.setup_axis_style()
        
        if not self.drones:
            # 顯示提示訊息
            self.ax.text(0.5, 0.5, 0.5, '請載入任務檔案或創建測試任務', 
                        transform=self.ax.transAxes, fontsize=14, color='#ffffff',
                        ha='center', va='center')
            self.canvas.draw_idle()
            return
        
        # 收集當前位置
        current_positions = {}
        all_x, all_y, all_z = [], [], []
        
        for drone_id, drone_data in self.drones.items():
            trajectory = drone_data['trajectory']
            color = drone_data['color']
            
            if not trajectory:
                continue
            
            # 收集座標
            x_coords = [p['x'] for p in trajectory]
            y_coords = [p['y'] for p in trajectory]
            z_coords = [p['z'] for p in trajectory]
            
            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_z.extend(z_coords)
            
            # 繪製完整軌跡（淡色虛線）
            self.ax.plot(x_coords, y_coords, z_coords, 
                        color=color, linewidth=1.5, alpha=0.4, linestyle='--', label=f'{drone_id}')
            
            # 繪製航點
            self.ax.scatter(x_coords, y_coords, z_coords, 
                           color=color, s=20, alpha=0.6, marker='.')
            
            # 取得當前位置
            current_pos = self.get_drone_position_at_time(drone_id, self.current_time)
            if current_pos:
                current_positions[drone_id] = current_pos
                
                # 繪製已飛行路徑（亮色實線）
                flown_path = self.get_flown_path(drone_id, self.current_time)
                if len(flown_path) > 1:
                    flown_x = [p['x'] for p in flown_path]
                    flown_y = [p['y'] for p in flown_path]
                    flown_z = [p['z'] for p in flown_path]
                    self.ax.plot(flown_x, flown_y, flown_z, 
                                color=color, linewidth=3, alpha=0.9)
                
                # 繪製無人機模型
                self.draw_drone_model(current_pos, color, drone_id)
        
        # 碰撞檢測（每0.1秒檢查一次）
        if (self.collision_var.get() and current_positions and 
            self.current_time - self.last_collision_check >= self.safety_config.collision_check_interval):
            
            warnings, new_loiters = self.collision_system.check_collisions_gpu(current_positions, self.current_time)
            
            # 繪製碰撞警告
            for warning in warnings:
                self.draw_collision_warning(warning, current_positions)
            
            self.last_collision_check = self.current_time
            self.update_warning_display(warnings)
        
        # 設置軸範圍
        if all_x and all_y and all_z:
            margin = max(20, (max(all_x) - min(all_x)) * 0.1)
            self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            self.ax.set_zlim(0, max(all_z) + margin)
        
        # 添加資訊文字
        info_text = f"時間: {self.current_time:.1f}s | 無人機: {len(self.drones)} | 安全距離: {self.safety_config.safety_distance:.1f}m"
        self.ax.text2D(0.02, 0.98, info_text, 
                      transform=self.ax.transAxes, fontsize=10, color='#00d4aa', weight='bold')
        
        # 顯示後端資訊
        if compute_manager:
            backend_text = f"計算後端: {compute_manager.backend.name}"
            self.ax.text2D(0.02, 0.02, backend_text, 
                          transform=self.ax.transAxes, fontsize=9, color='#ffd700')
        
        self.canvas.draw_idle()
    
    def draw_drone_model(self, position: Dict, color: str, drone_id: str):
        """繪製無人機模型"""
        x, y, z = position['x'], position['y'], position['z']
        size = 1.5
        
        # 根據飛行階段調整顯示
        phase = position.get('phase', FlightPhase.AUTO)
        
        if phase == FlightPhase.TAXI:
            # 地面滑行：小方塊
            self.ax.scatter([x], [y], [z], s=80, c=[color], marker='s', 
                           alpha=0.8, edgecolors='white', linewidth=1)
        elif phase == FlightPhase.TAKEOFF:
            # 起飛中：三角形
            self.ax.scatter([x], [y], [z], s=120, c=[color], marker='^', 
                           alpha=0.9, edgecolors='white', linewidth=2)
        else:
            # 正常飛行：十字形無人機
            # 機身
            self.ax.scatter([x], [y], [z], s=150, c=[color], marker='s', 
                           alpha=0.9, edgecolors='white', linewidth=2)
            
            # 螺旋槳臂
            arms = [
                (x + size, y, z), (x - size, y, z),
                (x, y + size, z), (x, y - size, z)
            ]
            
            # 螺旋槳
            arm_x, arm_y, arm_z = zip(*arms)
            self.ax.scatter(arm_x, arm_y, arm_z, s=40, c=[color]*4, 
                           marker='o', alpha=0.8, edgecolors='white', linewidth=1)
            
            # 連接線
            for arm_x, arm_y, arm_z in arms:
                self.ax.plot([x, arm_x], [y, arm_y], [z, arm_z], 
                            color=color, linewidth=2, alpha=0.8)
        
        # 標籤
        label = drone_id.split('_')[1]
        self.ax.text(x, y, z + size + 1, label, fontsize=9, color='white', 
                    weight='bold', ha='center', va='bottom')
        
        # 高度指示線
        if z > 0.1:
            self.ax.plot([x, x], [y, y], [0, z], 
                        color=color, linewidth=1, alpha=0.3, linestyle=':')
    
    def draw_collision_warning(self, warning: Dict, positions: Dict):
        """繪製碰撞警告"""
        drone1, drone2 = warning['drone1'], warning['drone2']
        
        if drone1 in positions and drone2 in positions:
            pos1, pos2 = positions[drone1], positions[drone2]
            
            # 紅色警告線
            self.ax.plot([pos1['x'], pos2['x']], 
                        [pos1['y'], pos2['y']], 
                        [pos1['z'], pos2['z']], 
                        color='red', linewidth=4, alpha=0.8)
            
            # 碰撞點標記
            mid_pos = warning['position']
            marker_size = 300 if warning['severity'] == 'critical' else 200
            
            self.ax.scatter([mid_pos[0]], [mid_pos[1]], [mid_pos[2]], 
                           s=marker_size, c='red', marker='X', 
                           alpha=0.9, edgecolors='white', linewidth=2)
            
            # 距離標籤
            distance_text = f"{warning['distance']:.1f}m"
            self.ax.text(mid_pos[0], mid_pos[1], mid_pos[2] + 1, distance_text,
                        fontsize=9, color='red', weight='bold',
                        ha='center', va='bottom')
    
    def get_flown_path(self, drone_id: str, current_time: float) -> List[Dict]:
        """取得已飛行路徑"""
        if drone_id not in self.drones:
            return []
            
        trajectory = self.drones[drone_id]['trajectory']
        if not trajectory:
            return []
        
        flown_path = []
        for point in trajectory:
            if point['time'] <= current_time:
                flown_path.append(point)
            else:
                # 添加當前插值位置
                current_pos = self.get_drone_position_at_time(drone_id, current_time)
                if current_pos:
                    flown_path.append(current_pos)
                break
                
        return flown_path
    
    def calculate_max_time(self):
        """計算最大時間"""
        self.max_time = 0.0
        
        for drone_data in self.drones.values():
            trajectory = drone_data['trajectory']
            if trajectory:
                drone_max_time = trajectory[-1]['time']
                self.max_time = max(self.max_time, drone_max_time)
        
        if self.max_time > 0:
            self.time_slider.config(to=self.max_time)
    
    def update_status_display(self):
        """更新狀態顯示"""
        self.status_text.delete(1.0, tk.END)
        
        if not self.drones:
            self.status_text.insert(tk.END, "尚未載入無人機數據\n\n請載入任務檔案或創建測試任務")
            return
        
        for drone_id, drone_data in self.drones.items():
            trajectory = drone_data['trajectory']
            current_pos = self.get_drone_position_at_time(drone_id, self.current_time)
            
            status = f"{drone_id} ({drone_data.get('name', 'Unknown')}):\n"
            status += f"  起飛位置: {drone_data['takeoff_position']}\n"
            status += f"  航點數量: {len(drone_data['waypoints'])}\n"
            status += f"  任務時長: {trajectory[-1]['time']:.1f}秒\n" if trajectory else "  時長: 0秒\n"
            
            if current_pos:
                phase_text = {
                    FlightPhase.TAXI: "地面滑行",
                    FlightPhase.TAKEOFF: "起飛爬升", 
                    FlightPhase.HOVER: "懸停等待",
                    FlightPhase.AUTO: "自動任務",
                    FlightPhase.LOITER: "等待避讓"
                }.get(current_pos.get('phase', FlightPhase.AUTO), "執行中")
                
                status += f"  當前階段: {phase_text}\n"
                status += f"  座標: ({current_pos['x']:.1f}, {current_pos['y']:.1f}, {current_pos['z']:.1f})\n"
            else:
                status += f"  狀態: 待機\n"
            
            status += "\n"
            self.status_text.insert(tk.END, status)
    
    def update_warning_display(self, warnings: List[Dict]):
        """更新警告顯示"""
        self.warning_text.delete(1.0, tk.END)
        
        if not warnings:
            self.warning_text.insert(tk.END, "✓ 飛行安全，無碰撞風險")
        else:
            self.warning_text.insert(tk.END, f"⚠ 偵測到 {len(warnings)} 個碰撞警告！\n\n")
            
            for i, warning in enumerate(warnings, 1):
                severity_text = "🚨 嚴重" if warning['severity'] == 'critical' else "⚠ 警告"
                text = f"{severity_text} {i}: {warning['drone1']} ↔ {warning['drone2']}\n"
                text += f"距離: {warning['distance']:.2f}m (安全距離: {self.safety_config.safety_distance:.1f}m)\n"
                if i < len(warnings):
                    text += "\n"
                self.warning_text.insert(tk.END, text)
    
    def load_qgc_files(self):
        """載入QGC檔案"""
        messagebox.showinfo("功能開發中", "QGC檔案載入功能正在開發中")
    
    def load_csv_files(self):
        """載入CSV檔案"""
        messagebox.showinfo("功能開發中", "CSV檔案載入功能正在開發中")
    
    def export_modified_missions(self):
        """匯出修改後的任務"""
        messagebox.showinfo("功能開發中", "任務匯出功能正在開發中")
    
    def toggle_play(self):
        """切換播放/暫停"""
        if not self.drones:
            messagebox.showwarning("無數據", "請先載入任務檔案或創建測試任務")
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.config(text="⏸", bg='#ffc107', fg='black')
            self.start_animation()
        else:
            self.play_button.config(text="▶", bg='#28a745', fg='white')
            self.stop_animation()
    
    def start_animation(self):
        """開始動畫"""
        if self.animation:
            self.animation.event_source.stop()
            
        self.last_update_time = time.time()
        
        def update_frame(frame):
            if not self.is_playing or self.max_time == 0:
                return
                
            current_real_time = time.time()
            dt = (current_real_time - self.last_update_time) * self.time_scale
            self.last_update_time = current_real_time
            
            self.current_time += dt
            
            if self.current_time > self.max_time:
                self.current_time = self.max_time
                self.toggle_play()
                
            # 優化：減少UI更新頻率
            if frame % 2 == 0:  # 每隔一幀更新一次UI
                self.time_var.set(self.current_time)
                self.update_time_display()
                self.update_status_display()
            
            self.update_3d_plot()
            
        self.animation = animation.FuncAnimation(
            self.fig, update_frame, interval=self.update_interval, blit=False
        )
    
    def stop_animation(self):
        """停止動畫"""
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
    
    def stop_simulation(self):
        """停止模擬"""
        self.is_playing = False
        self.play_button.config(text="▶", bg='#28a745', fg='white')
        self.stop_animation()
    
    def reset_simulation(self):
        """重置模擬"""
        self.stop_simulation()
        self.current_time = 0.0
        self.time_var.set(0.0)
        self.last_collision_check = 0.0
        
        self.update_time_display()
        self.update_status_display()
        self.update_3d_plot()
        
        logger.info("模擬已重置")
    
    def on_time_change(self, value):
        """時間滑桿改變"""
        if not self.is_playing:  # 只在暫停時響應手動調整
            self.current_time = float(value)
            self.update_time_display()
            self.update_status_display()
            self.update_3d_plot()
    
    def on_speed_change(self, value):
        """速度改變"""
        self.time_scale = float(value)
    
    def on_safety_change(self, value):
        """安全距離改變"""
        self.safety_config.safety_distance = float(value)
        if not self.is_playing:
            self.update_3d_plot()
    
    def update_time_display(self):
        """更新時間顯示"""
        current_min = int(self.current_time // 60)
        current_sec = int(self.current_time % 60)
        max_min = int(self.max_time // 60)
        max_sec = int(self.max_time % 60)
        
        time_text = f"{current_min:02d}:{current_sec:02d} / {max_min:02d}:{max_sec:02d}"
        self.time_label.config(text=time_text)
    
    def on_closing(self):
        """視窗關閉處理"""
        if self.is_playing:
            self.stop_simulation()
            
        plt.close('all')
        self.root.quit()
        self.root.destroy()


def main():
    """主函數（用於獨立測試）"""
    root = tk.Tk()
    app = DroneSimulatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()