"""
進階3D視覺化元件 - GPU加速版本
提供專業級的無人機群飛3D可視化功能
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, List, Tuple, Optional, Any
import time

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = np

from utils.gpu_utils import get_array_module
from config.settings import VisualizationConfig
from core.trajectory_calculator import FlightPhase

logger = logging.getLogger(__name__)

class Advanced3DPlotter:
    """
    進階3D繪圖器
    提供高性能的3D無人機群飛可視化
    """
    
    def __init__(self, parent_frame: tk.Widget, config: VisualizationConfig, use_gpu: bool = True):
        """
        初始化3D繪圖器
        
        Args:
            parent_frame: 父級tkinter框架
            config: 視覺化配置
            use_gpu: 是否使用GPU加速
        """
        self.parent_frame = parent_frame
        self.config = config
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.xp = get_array_module(self.use_gpu)
        
        # 繪圖組件
        self.figure = None
        self.ax = None
        self.canvas = None
        self.toolbar = None
        
        # 數據快取
        self.drone_data_cache = {}
        self.trajectory_cache = {}
        self.collision_cache = []
        
        # 動畫控制
        self.animation = None
        self.last_update_time = time.time()
        
        # 視角控制
        self.current_view = "3d"  # "top", "side", "3d"
        self.view_angles = {
            "top": (90, 0),
            "side": (0, 0), 
            "3d": (30, 45)
        }
        
        self._setup_3d_plot()
        self._setup_custom_toolbar()
        self._enable_mouse_interactions()
        
        logger.info(f"進階3D繪圖器初始化: GPU={'啟用' if self.use_gpu else '禁用'}")
    
    def _setup_3d_plot(self):
        """設置3D繪圖環境"""
        # 使用專業暗色主題
        plt.style.use('dark_background')
        
        # 創建高解析度圖形
        self.figure = plt.figure(
            figsize=self.config.figure_size,
            facecolor=self.config.background_color,
            dpi=self.config.dpi
        )
        
        # 創建3D子圖
        self.ax = self.figure.add_subplot(111, projection='3d')
        
        # 設置背景和樣式
        self.ax.set_facecolor(self.config.background_color)
        self.figure.patch.set_facecolor(self.config.background_color)
        
        # 軸標籤和標題
        self._setup_axis_style()
        
        # 創建畫布
        canvas_frame = tk.Frame(self.parent_frame, bg=self.config.background_color)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.figure, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 設置初始視角
        self.ax.view_init(elev=30, azim=45)
    
    def _setup_axis_style(self):
        """設置軸樣式"""
        # 網格設置
        self.ax.grid(True, alpha=0.3, color=self.config.grid_color, linewidth=0.5)
        
        # 軸標籤
        self.ax.set_xlabel('東向距離 (m)', fontsize=12, color=self.config.text_color, labelpad=10)
        self.ax.set_ylabel('北向距離 (m)', fontsize=12, color=self.config.text_color, labelpad=10)
        self.ax.set_zlabel('飛行高度 (m)', fontsize=12, color=self.config.text_color, labelpad=10)
        
        # 標題
        self.ax.set_title('無人機群飛3D軌跡模擬 - GPU加速版',
                         fontsize=14, color='#ffffff', pad=20)
        
        # 刻度樣式
        self.ax.tick_params(colors='#888888', labelsize=10)
        
        # 軸面板設置
        for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor(self.config.grid_color)
            pane.set_alpha(0.1)
    
    def _setup_custom_toolbar(self):
        """設置自定義工具欄"""
        toolbar_frame = tk.Frame(self.parent_frame, bg='#3a3a3a', height=40)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar_frame.pack_propagate(False)
        
        # 標準matplotlib工具欄
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.configure(bg='#3a3a3a')
        
        # 自定義視角按鈕
        custom_frame = tk.Frame(toolbar_frame, bg='#3a3a3a')
        custom_frame.pack(side=tk.RIGHT, padx=10)
        
        # 視角控制按鈕
        tk.Button(custom_frame, text="俯視", command=lambda: self.set_view("top"),
                 bg='#007bff', fg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(custom_frame, text="側視", command=lambda: self.set_view("side"),
                 bg='#007bff', fg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        
        tk.Button(custom_frame, text="3D", command=lambda: self.set_view("3d"),
                 bg='#007bff', fg='white', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        
        # 渲染品質控制
        quality_label = tk.Label(custom_frame, text="品質:", bg='#3a3a3a', fg='white', font=('Arial', 8))
        quality_label.pack(side=tk.LEFT, padx=(10, 2))
        
        quality_var = tk.StringVar(value=self.config.render_quality)
        quality_combo = ttk.Combobox(custom_frame, textvariable=quality_var,
                                   values=["low", "medium", "high"], width=8)
        quality_combo.pack(side=tk.LEFT, padx=2)
        quality_combo.bind('<<ComboboxSelected>>', self._on_quality_change)
    
    def _enable_mouse_interactions(self):
        """啟用滑鼠交互功能"""
        def on_scroll(event):
            """滑鼠滾輪縮放"""
            if event.inaxes == self.ax:
                # 獲取當前軸限制
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
        
        def on_double_click(event):
            """雙擊重置視圖"""
            if event.inaxes == self.ax and event.dblclick:
                self.reset_view()
        
        self.canvas.mpl_connect('scroll_event', on_scroll)
        self.canvas.mpl_connect('button_press_event', on_double_click)
    
    def update_plot(self, drone_states: Dict[str, Any], collision_warnings: List[Dict],
                   current_time: float) -> None:
        """
        更新3D繪圖
        
        Args:
            drone_states: 無人機狀態字典
            collision_warnings: 碰撞警告列表
            current_time: 當前時間
        """
        start_time = time.time()
        
        # 清除舊的繪圖元素
        self.ax.clear()
        self._setup_axis_style()
        
        if not drone_states:
            self._draw_empty_plot()
            self.canvas.draw_idle()
            return
        
        # 繪製無人機軌跡和位置
        positions = self._draw_drones(drone_states, current_time)
        
        # 繪製碰撞警告
        if collision_warnings:
            self._draw_collision_warnings(collision_warnings, positions)
        
        # 設置軸範圍
        self._update_axis_limits(drone_states)
        
        # 繪製資訊文字
        self._draw_info_text(current_time, drone_states, collision_warnings)
        
        # 更新畫布
        self.canvas.draw_idle()
        
        # 性能統計
        render_time = time.time() - start_time
        if render_time > 0.05:  # 超過50ms警告
            logger.debug(f"渲染時間較長: {render_time*1000:.1f}ms")
    
    def _draw_drones(self, drone_states: Dict[str, Any], current_time: float) -> Dict[str, Dict]:
        """
        繪製無人機和軌跡
        
        Args:
            drone_states: 無人機狀態
            current_time: 當前時間
            
        Returns:
            當前位置字典
        """
        positions = {}
        
        for drone_id, drone_state in drone_states.items():
            trajectory = drone_state.trajectory
            color = drone_state.color
            current_pos = drone_state.current_position
            
            if not trajectory:
                continue
            
            # 提取軌跡坐標
            x_coords = [p.x for p in trajectory]
            y_coords = [p.y for p in trajectory]
            z_coords = [p.z for p in trajectory]
            
            # 繪製完整軌跡（虛線，低透明度）
            if self.config.render_quality in ["medium", "high"]:
                self.ax.plot(x_coords, y_coords, z_coords,
                           color=color, linewidth=1.5, 
                           alpha=self.config.trajectory_alpha,
                           linestyle='--', label=f'{drone_id} 計劃軌跡')
            
            # 繪製航點
            if self.config.render_quality == "high":
                self.ax.scatter(x_coords, y_coords, z_coords,
                              color=color, s=self.config.waypoint_size,
                              alpha=0.6, marker='.')
            
            # 繪製已飛行路徑
            if current_pos:
                positions[drone_id] = current_pos
                
                # 獲取已飛行的軌跡段
                flown_path = self._get_flown_path(trajectory, current_time)
                
                if len(flown_path) > 1:
                    flown_x = [p.x for p in flown_path]
                    flown_y = [p.y for p in flown_path]
                    flown_z = [p.z for p in flown_path]
                    
                    self.ax.plot(flown_x, flown_y, flown_z,
                               color=color, linewidth=4,
                               alpha=self.config.flown_path_alpha,
                               label=f'{drone_id} 已飛行')
                
                # 繪製無人機模型
                self._draw_drone_model(current_pos, color, drone_id)
        
        return positions
    
    def _draw_drone_model(self, position: Dict, color: str, drone_id: str):
        """
        繪製精緻的無人機3D模型
        
        Args:
            position: 位置資訊
            color: 無人機顏色
            drone_id: 無人機ID
        """
        x, y, z = position['x'], position['y'], position['z']
        phase = position.get('phase', FlightPhase.AUTO)
        
        # 根據飛行階段調整模型
        if phase == FlightPhase.TAXI:
            # 地面滑行：小方塊
            self.ax.scatter([x], [y], [z], s=100, c=[color], marker='s',
                          alpha=0.8, edgecolors='white', linewidth=1)
        elif phase == FlightPhase.TAKEOFF:
            # 起飛中：三角形
            self.ax.scatter([x], [y], [z], s=150, c=[color], marker='^',
                          alpha=0.9, edgecolors='white', linewidth=2)
        else:
            # 正常飛行：詳細無人機模型
            size = 2.0
            
            # 機身
            self.ax.scatter([x], [y], [z], s=self.config.drone_model_size,
                          c=[color], marker='s', alpha=0.9,
                          edgecolors='white', linewidth=2)
            
            # 螺旋槳臂（如果是高品質渲染）
            if self.config.render_quality == "high":
                arms = [
                    (x + size, y, z + 0.2),
                    (x - size, y, z + 0.2),
                    (x, y + size, z + 0.2),
                    (x, y - size, z + 0.2)
                ]
                
                # 螺旋槳
                arm_x, arm_y, arm_z = zip(*arms)
                self.ax.scatter(arm_x, arm_y, arm_z, s=60, c=[color]*4,
                              marker='o', alpha=0.8, edgecolors='white', linewidth=1)
                
                # 連接線
                for arm_x, arm_y, arm_z in arms:
                    self.ax.plot([x, arm_x], [y, arm_y], [z, arm_z],
                               color=color, linewidth=2.5, alpha=0.8)
            
            # 高度指示線
            if z > 0.1:
                self.ax.plot([x, x], [y, y], [0, z],
                           color=color, linewidth=1, alpha=0.3, linestyle=':')
        
        # 標籤
        label = drone_id.split('_')[1] if '_' in drone_id else drone_id
        self.ax.text(x, y, z + 3, label, fontsize=11, color='white',
                    weight='bold', ha='center', va='bottom')
    
    def _draw_collision_warnings(self, warnings: List[Dict], positions: Dict[str, Dict]):
        """
        繪製碰撞警告視覺效果
        
        Args:
            warnings: 警告列表
            positions: 位置字典
        """
        for warning in warnings:
            drone1_id = warning.get('drone1_id') or warning.get('drone1')
            drone2_id = warning.get('drone2_id') or warning.get('drone2')
            
            if drone1_id in positions and drone2_id in positions:
                pos1 = positions[drone1_id]
                pos2 = positions[drone2_id]
                
                # 紅色警告線
                self.ax.plot([pos1['x'], pos2['x']],
                           [pos1['y'], pos2['y']],
                           [pos1['z'], pos2['z']],
                           color='red', linewidth=self.config.collision_line_width,
                           alpha=0.8, linestyle='-')
                
                # 碰撞點標記
                mid_x = (pos1['x'] + pos2['x']) / 2
                mid_y = (pos1['y'] + pos2['y']) / 2
                mid_z = (pos1['z'] + pos2['z']) / 2
                
                severity = warning.get('severity', 'warning')
                marker_size = (self.config.critical_marker_size if severity == 'critical'
                             else self.config.warning_marker_size)
                
                self.ax.scatter([mid_x], [mid_y], [mid_z], s=marker_size,
                              c='red', marker='X', alpha=0.9,
                              edgecolors='white', linewidth=3)
                
                # 距離標籤
                distance = warning.get('distance', 0)
                if distance > 0:
                    self.ax.text(mid_x, mid_y, mid_z + 2, f"{distance:.1f}m",
                               fontsize=10, color='red', weight='bold',
                               ha='center', va='bottom')
    
    def _get_flown_path(self, trajectory: List[Any], current_time: float) -> List[Any]:
        """獲取已飛行的軌跡段"""
        flown_path = []
        
        for point in trajectory:
            if point.time <= current_time:
                flown_path.append(point)
            else:
                break
        
        return flown_path
    
    def _update_axis_limits(self, drone_states: Dict[str, Any]):
        """更新軸範圍"""
        all_x, all_y, all_z = [], [], []
        
        for drone_state in drone_states.values():
            trajectory = drone_state.trajectory
            if trajectory:
                all_x.extend([p.x for p in trajectory])
                all_y.extend([p.y for p in trajectory])
                all_z.extend([p.z for p in trajectory])
        
        if all_x and all_y and all_z:
            margin = 50
            self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            self.ax.set_zlim(0, max(all_z) + margin)
    
    def _draw_info_text(self, current_time: float, drone_states: Dict[str, Any],
                       collision_warnings: List[Dict]):
        """繪製資訊文字"""
        info_y = 0.98
        line_height = 0.04
        
        # 時間資訊
        self.ax.text2D(0.02, info_y, f"時間: {current_time:.1f}s",
                      transform=self.ax.transAxes, fontsize=12,
                      color=self.config.text_color, weight='bold')
        
        # 無人機數量
        self.ax.text2D(0.02, info_y - line_height,
                      f"無人機: {len(drone_states)}/8",
                      transform=self.ax.transAxes, fontsize=11, color='#ffffff')
        
        # 碰撞警告
        warning_text = f"碰撞警告: {len(collision_warnings)}"
        warning_color = '#ff5722' if collision_warnings else '#4caf50'
        self.ax.text2D(0.02, info_y - 2*line_height, warning_text,
                      transform=self.ax.transAxes, fontsize=11, color=warning_color)
        
        # GPU狀態
        gpu_text = f"GPU: {'啟用' if self.use_gpu else '禁用'}"
        self.ax.text2D(0.02, info_y - 3*line_height, gpu_text,
                      transform=self.ax.transAxes, fontsize=10, color='#ffd700')
    
    def _draw_empty_plot(self):
        """繪製空白圖表"""
        self.ax.text2D(0.5, 0.5, "📄 無載入的無人機\n\n請載入QGC或CSV檔案",
                      transform=self.ax.transAxes, fontsize=16,
                      color='#888888', ha='center', va='center')
    
    def set_view(self, view_type: str):
        """
        設置視角
        
        Args:
            view_type: 視角類型 ("top", "side", "3d")
        """
        if view_type in self.view_angles:
            elev, azim = self.view_angles[view_type]
            self.ax.view_init(elev=elev, azim=azim)
            self.current_view = view_type
            self.canvas.draw_idle()
            logger.debug(f"切換到 {view_type} 視角")
    
    def reset_view(self):
        """重置視圖"""
        self.set_view("3d")
        # 重置縮放
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
        logger.debug("視圖已重置")
    
    def _on_quality_change(self, event):
        """渲染品質改變事件"""
        quality = event.widget.get()
        self.config.render_quality = quality
        logger.info(f"渲染品質設置為: {quality}")
        
        # 根據品質調整設置
        if quality == "low":
            self.config.trajectory_alpha = 0.2
            self.config.waypoint_size = 15
        elif quality == "medium":
            self.config.trajectory_alpha = 0.4
            self.config.waypoint_size = 25
        else:  # high
            self.config.trajectory_alpha = 0.4
            self.config.waypoint_size = 35
    
    def export_view(self, filename: str, dpi: int = 300) -> bool:
        """
        導出當前視圖
        
        Args:
            filename: 檔案名稱
            dpi: 解析度
            
        Returns:
            是否成功
        """
        try:
            self.figure.savefig(filename, dpi=dpi, bbox_inches='tight',
                              facecolor=self.config.background_color)
            logger.info(f"視圖已導出: {filename}")
            return True
        except Exception as e:
            logger.error(f"導出視圖失敗: {e}")
            return False
    
    def cleanup(self):
        """清理資源"""
        if self.animation:
            self.animation.event_source.stop()
        
        plt.close(self.figure)
        logger.debug("3D繪圖器資源已清理")
