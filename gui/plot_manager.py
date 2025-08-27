"""
GPU加速視覺化管理系統
支持高效能3D無人機群視覺化和即時渲染
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import messagebox
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# GPU工具導入
from utils.gpu_utils import (
    get_array_module, asarray, to_cpu, to_gpu, is_gpu_enabled,
    synchronize, gpu_accelerated, MathOps, performance_monitor
)
from config.settings import settings

logger = logging.getLogger(__name__)

class RenderBackend(Enum):
    """渲染後端選項"""
    MATPLOTLIB = "matplotlib"
    OPENGL = "opengl"  
    WEBGL = "webgl"
    AUTO = "auto"

class ViewMode(Enum):
    """視角模式"""
    TOP = "top"
    SIDE = "side"
    PERSPECTIVE = "3d"
    FRONT = "front"
    FOLLOW = "follow"

@dataclass
class RenderSettings:
    """渲染設定"""
    backend: RenderBackend = RenderBackend.AUTO
    fps_target: int = 30
    trail_length: int = 100
    point_size: float = 100.0
    line_width: float = 2.0
    transparency: float = 0.8
    anti_aliasing: bool = True
    gpu_acceleration: bool = True
    
    # 效能優化設定
    level_of_detail: bool = True  # 距離相關細節層級
    frustum_culling: bool = True  # 視椎體裁剪
    batch_rendering: bool = True  # 批次渲染

@dataclass  
class DroneVisual:
    """無人機視覺元素"""
    position: Any  # 當前位置
    trail_positions: List[Any] = field(default_factory=list)  # 軌跡歷史
    color: str = "blue"
    size: float = 100.0
    visible: bool = True
    label: str = ""

class GPUPlotManager:
    """GPU加速繪圖管理器"""
    
    def __init__(self, parent_frame=None):
        self.parent_frame = parent_frame
        self.xp = get_array_module()
        
        # 渲染設定
        self.render_settings = RenderSettings()
        self.view_mode = ViewMode.PERSPECTIVE
        
        # 視覺化組件
        self.fig = None
        self.ax = None
        self.canvas = None
        self.animation = None
        
        # 資料結構
        self.drone_visuals: Dict[str, DroneVisual] = {}
        self.collision_lines: List[Any] = []
        self.waypoint_markers: List[Any] = []
        
        # GPU渲染緩衝區
        self.position_buffer = None  # GPU位置緩衝
        self.color_buffer = None     # 顏色緩衝
        self.trail_buffer = None     # 軌跡緩衝
        
        # 效能監控
        self.frame_times = []
        self.render_count = 0
        self.last_render_time = 0
        
        # 執行緒池用於並行處理
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 自動選擇最佳渲染後端
        self._select_optimal_backend()
        
        # 初始化渲染系統
        self._initialize_renderer()
        
        logger.info(f"🎨 GPU視覺化管理器初始化 (後端: {self.render_settings.backend.value})")

    def _select_optimal_backend(self):
        """自動選擇最佳渲染後端"""
        if self.render_settings.backend == RenderBackend.AUTO:
            if is_gpu_enabled() and settings.gpu.accelerate_visualization:
                # 檢測OpenGL可用性
                try:
                    import OpenGL.GL as gl
                    self.render_settings.backend = RenderBackend.OPENGL
                    logger.info("🚀 選用OpenGL GPU渲染後端")
                except ImportError:
                    self.render_settings.backend = RenderBackend.MATPLOTLIB
                    logger.info("⚡ 選用優化的Matplotlib後端")
            else:
                self.render_settings.backend = RenderBackend.MATPLOTLIB
                logger.info("🖥️ 選用標準Matplotlib後端")

    def _initialize_renderer(self):
        """初始化渲染器"""
        if self.render_settings.backend == RenderBackend.OPENGL:
            self._setup_opengl_renderer()
        else:
            self._setup_matplotlib_renderer()

    def _setup_matplotlib_renderer(self):
        """設置優化的Matplotlib渲染器"""
        # 使用高效能後端
        if self.render_settings.gpu_acceleration:
            try:
                plt.switch_backend('Qt5Agg')  # 硬體加速後端
            except:
                plt.switch_backend('TkAgg')   # 回退後端
        
        # 創建高DPI圖表
        dpi = settings.visualization.dpi
        self.fig = plt.figure(
            figsize=(12, 8), 
            dpi=dpi,
            facecolor='black',
            edgecolor='white'
        )
        
        # 設置3D軸
        self.ax = self.fig.add_subplot(111, projection='3d')
        self._configure_3d_axis()
        
        # 優化渲染設定
        self._optimize_matplotlib()

    def _configure_3d_axis(self):
        """配置3D軸"""
        self.ax.set_facecolor('black')
        self.ax.grid(True, alpha=0.3, color='white')
        
        # 軸標籤
        self.ax.set_xlabel('東 (公尺)', color='white', fontsize=10)
        self.ax.set_ylabel('北 (公尺)', color='white', fontsize=10)  
        self.ax.set_zlabel('高度 (公尺)', color='white', fontsize=10)
        
        # 標題
        backend_info = "GPU" if is_gpu_enabled() else "CPU"
        self.ax.set_title(f'無人機群模擬 ({backend_info}加速)', color='white', fontsize=12)
        
        # 軸顏色
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.tick_params(colors='white')

    def _optimize_matplotlib(self):
        """優化Matplotlib效能"""
        # 關閉不必要的功能以提升效能
        self.ax.set_rasterization_zorder(0)
        
        # 使用較少的線段數來繪製3D物件
        self.ax.zaxis._axinfo['juggled'] = (1, 2, 0)
        
        # 設置更新間隔
        self.update_interval = max(1000 // self.render_settings.fps_target, 16)  # 至少16ms

    def _setup_opengl_renderer(self):
        """設置OpenGL渲染器 (進階功能)"""
        try:
            import OpenGL.GL as gl
            from OpenGL.arrays import vbo
            
            logger.info("🔧 設置OpenGL GPU渲染器...")
            
            # OpenGL上下文將在實際使用時初始化
            self.gl_context = None
            self.vertex_buffer = None
            self.color_buffer_gl = None
            
            # 暫時仍使用matplotlib作為顯示介面
            self._setup_matplotlib_renderer()
            
            logger.info("✅ OpenGL渲染器準備就緒")
            
        except Exception as e:
            logger.warning(f"⚠️ OpenGL初始化失敗，回退到Matplotlib: {e}")
            self.render_settings.backend = RenderBackend.MATPLOTLIB
            self._setup_matplotlib_renderer()

    def setup_canvas(self, parent_frame):
        """設置畫布"""
        if parent_frame:
            self.parent_frame = parent_frame
        
        # 創建tkinter畫布
        self.canvas = FigureCanvasTkAgg(self.fig, self.parent_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 綁定互動事件
        self._setup_interactions()
        
        # 啟動動畫循環 (避免快取警告)
        self.animation = animation.FuncAnimation(
            self.fig,
            self._render_frame,
            interval=self.update_interval,
            blit=False,
            cache_frame_data=False,  # 避免記憶體無限增長警告
            repeat=True
        )
        
        logger.info("🖼️ 畫布設置完成")

    def _setup_interactions(self):
        """設置使用者互動"""
        if not self.canvas:
            return
        
        # 滑鼠滾輪縮放
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        # 鍵盤快捷鍵
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        # 滑鼠拖曳旋轉 (由matplotlib 3D預設處理)

    def _on_scroll(self, event):
        """處理滑鼠滾輪縮放"""
        if event.inaxes != self.ax:
            return
        
        # 調整視角距離
        scale_factor = 1.1 if event.step > 0 else 0.9
        
        # 獲取當前軸範圍
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim() 
        zlim = self.ax.get_zlim()
        
        # 計算中心點
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        z_center = (zlim[0] + zlim[1]) / 2
        
        # 縮放範圍
        x_range = (xlim[1] - xlim[0]) * scale_factor / 2
        y_range = (ylim[1] - ylim[0]) * scale_factor / 2
        z_range = (zlim[1] - zlim[0]) * scale_factor / 2
        
        # 設置新的軸範圍
        self.ax.set_xlim(x_center - x_range, x_center + x_range)
        self.ax.set_ylim(y_center - y_range, y_center + y_range)
        self.ax.set_zlim(z_center - z_range, z_center + z_range)
        
        self.canvas.draw_idle()

    def _on_key_press(self, event):
        """處理鍵盤快捷鍵"""
        if event.key == '1':
            self.set_view_mode(ViewMode.TOP)
        elif event.key == '2':
            self.set_view_mode(ViewMode.SIDE)
        elif event.key == '3':
            self.set_view_mode(ViewMode.PERSPECTIVE)
        elif event.key == 'f':
            self.fit_view_to_data()
        elif event.key == 'r':
            self.reset_view()

    @gpu_accelerated()
    def update_drone_data(
        self, 
        drone_positions: Dict[str, Any],
        collision_pairs: Optional[List[Tuple[str, str, float]]] = None,
        waypoints: Optional[Dict[str, Any]] = None
    ):
        """更新無人機資料（GPU優化）"""
        # 批次處理位置更新
        if drone_positions:
            self._update_positions_gpu(drone_positions)
        
        # 更新碰撞線
        if collision_pairs:
            self._update_collision_lines(collision_pairs, drone_positions)
        
        # 更新航點標記
        if waypoints:
            self._update_waypoint_markers(waypoints)

    @gpu_accelerated()
    def _update_positions_gpu(self, drone_positions: Dict[str, Any]):
        """GPU批次更新無人機位置"""
        drone_names = list(drone_positions.keys())
        n_drones = len(drone_names)
        
        if n_drones == 0:
            return
        
        # 準備GPU批次處理
        if self.position_buffer is None or len(self.position_buffer) != n_drones:
            self.position_buffer = self.xp.zeros((n_drones, 3), dtype=self.xp.float32)
            self.color_buffer = self.xp.zeros((n_drones, 4), dtype=self.xp.float32)  # RGBA
        
        # 批次更新位置
        for i, (drone_name, position) in enumerate(drone_positions.items()):
            if drone_name not in self.drone_visuals:
                # 創建新的視覺元素
                self.drone_visuals[drone_name] = DroneVisual(
                    position=asarray(position),
                    color=self._get_drone_color(i),
                    label=drone_name
                )
            
            # 更新位置
            drone_visual = self.drone_visuals[drone_name]
            new_pos = asarray(position)
            drone_visual.position = new_pos
            
            # 更新GPU緩衝
            self.position_buffer[i] = new_pos
            
            # 更新軌跡
            self._update_drone_trail(drone_visual, new_pos)
            
            # 設置顏色
            color_rgba = self._color_to_rgba(drone_visual.color)
            self.color_buffer[i] = asarray(color_rgba)
        
        # 同步GPU操作
        if is_gpu_enabled():
            synchronize()

    def _update_drone_trail(self, drone_visual: DroneVisual, new_position: Any):
        """更新無人機軌跡"""
        # 添加新位置到軌跡
        drone_visual.trail_positions.append(new_position.copy())
        
        # 限制軌跡長度
        max_trail = self.render_settings.trail_length
        if len(drone_visual.trail_positions) > max_trail:
            drone_visual.trail_positions = drone_visual.trail_positions[-max_trail:]

    def _update_collision_lines(
        self, 
        collision_pairs: List[Tuple[str, str, float]],
        drone_positions: Dict[str, Any]
    ):
        """更新碰撞警告線"""
        self.collision_lines.clear()
        
        for drone1, drone2, distance in collision_pairs:
            if drone1 in drone_positions and drone2 in drone_positions:
                pos1 = drone_positions[drone1]
                pos2 = drone_positions[drone2]
                
                # 根據距離設置顏色
                if distance < settings.safety.critical_distance:
                    color = 'red'
                elif distance < settings.safety.warning_distance:
                    color = 'orange'
                else:
                    color = 'yellow'
                
                self.collision_lines.append({
                    'start': to_cpu(asarray(pos1)),
                    'end': to_cpu(asarray(pos2)),
                    'color': color,
                    'distance': distance
                })

    def _update_waypoint_markers(self, waypoints: Dict[str, Any]):
        """更新航點標記"""
        # 清除舊標記
        self.waypoint_markers.clear()
        
        # 添加新標記
        for drone_name, waypoint_list in waypoints.items():
            for i, waypoint in enumerate(waypoint_list):
                if isinstance(waypoint, (list, tuple)) and len(waypoint) >= 3:
                    self.waypoint_markers.append({
                        'position': waypoint[:3],
                        'drone': drone_name,
                        'index': i,
                        'color': self.drone_visuals.get(drone_name, DroneVisual()).color
                    })

    def _render_frame(self, frame_num):
        """渲染單一幀"""
        start_time = time.perf_counter()
        
        try:
            # 清除軸
            self.ax.clear()
            self._configure_3d_axis()
            
            # 渲染無人機
            self._render_drones()
            
            # 渲染軌跡
            self._render_trails()
            
            # 渲染碰撞警告
            self._render_collisions()
            
            # 渲染航點
            self._render_waypoints()
            
            # 渲染效能資訊
            self._render_performance_info()
            
            # 更新效能統計
            render_time = time.perf_counter() - start_time
            self.frame_times.append(render_time)
            
            # 限制記錄長度
            if len(self.frame_times) > 100:
                self.frame_times = self.frame_times[-50:]
            
            self.render_count += 1
            self.last_render_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ 渲染失敗: {e}")

    def _render_drones(self):
        """渲染無人機"""
        if not self.drone_visuals:
            return
        
        # 批次收集位置和顏色
        positions = []
        colors = []
        labels = []
        
        for drone_name, visual in self.drone_visuals.items():
            if visual.visible and visual.position is not None:
                pos_cpu = to_cpu(visual.position)
                positions.append(pos_cpu)
                colors.append(visual.color)
                labels.append(visual.label)
        
        if positions:
            positions_array = np.array(positions)
            
            # 批次繪製所有無人機
            scatter = self.ax.scatter(
                positions_array[:, 0],
                positions_array[:, 1], 
                positions_array[:, 2],
                c=colors,
                s=self.render_settings.point_size,
                alpha=self.render_settings.transparency,
                depthshade=True,
                edgecolors='white',
                linewidth=1
            )
            
            # 添加標籤 (可選，效能考量)
            if len(positions) <= 20:  # 只在無人機數量較少時顯示標籤
                for pos, label in zip(positions, labels):
                    self.ax.text(pos[0], pos[1], pos[2] + 2, label,
                               fontsize=8, color='white')

    def _render_trails(self):
        """渲染軌跡"""
        for drone_name, visual in self.drone_visuals.items():
            if not visual.visible or len(visual.trail_positions) < 2:
                continue
            
            # 轉換軌跡位置
            trail_cpu = [to_cpu(pos) for pos in visual.trail_positions]
            trail_array = np.array(trail_cpu)
            
            # 繪製軌跡線
            self.ax.plot(
                trail_array[:, 0],
                trail_array[:, 1],
                trail_array[:, 2],
                color=visual.color,
                alpha=0.6,
                linewidth=self.render_settings.line_width,
                linestyle='-'
            )

    def _render_collisions(self):
        """渲染碰撞警告線"""
        for collision in self.collision_lines:
            start = collision['start']
            end = collision['end']
            color = collision['color']
            distance = collision['distance']
            
            # 繪製警告線
            self.ax.plot(
                [start[0], end[0]],
                [start[1], end[1]], 
                [start[2], end[2]],
                color=color,
                linewidth=4,
                alpha=0.8,
                linestyle='--'
            )
            
            # 顯示距離標籤
            mid_point = (start + end) / 2
            self.ax.text(
                mid_point[0], mid_point[1], mid_point[2],
                f'{distance:.1f}m',
                fontsize=9,
                color=color,
                fontweight='bold'
            )

    def _render_waypoints(self):
        """渲染航點標記"""
        if not self.waypoint_markers:
            return
        
        # 批次收集航點資料
        positions = [marker['position'] for marker in self.waypoint_markers]
        colors = [marker['color'] for marker in self.waypoint_markers]
        
        if positions:
            positions_array = np.array(positions)
            
            # 繪製航點
            self.ax.scatter(
                positions_array[:, 0],
                positions_array[:, 1],
                positions_array[:, 2],
                c=colors,
                s=50,
                alpha=0.7,
                marker='x',
                depthshade=False
            )

    def _render_performance_info(self):
        """渲染效能資訊"""
        if not self.frame_times:
            return
        
        # 計算FPS
        avg_frame_time = np.mean(self.frame_times[-30:])
        current_fps = 1.0 / (avg_frame_time + 1e-6)
        
        # 獲取後端資訊
        backend_info = "GPU" if is_gpu_enabled() else "CPU"
        if self.render_settings.backend == RenderBackend.OPENGL:
            backend_info += "+OpenGL"
        
        # 顯示效能資訊
        info_text = f'{backend_info} | FPS: {current_fps:.1f} | 無人機: {len(self.drone_visuals)}'
        
        self.ax.text2D(
            0.02, 0.98, info_text,
            transform=self.ax.transAxes,
            color='lime',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7)
        )

    def set_view_mode(self, mode: ViewMode):
        """設置視角模式"""
        self.view_mode = mode
        
        if mode == ViewMode.TOP:
            self.ax.view_init(elev=90, azim=0)
        elif mode == ViewMode.SIDE:
            self.ax.view_init(elev=0, azim=0)
        elif mode == ViewMode.FRONT:
            self.ax.view_init(elev=0, azim=90)
        elif mode == ViewMode.PERSPECTIVE:
            self.ax.view_init(elev=20, azim=45)
        
        if self.canvas:
            self.canvas.draw_idle()
        
        logger.info(f"📐 視角切換到: {mode.value}")

    def fit_view_to_data(self):
        """自動調整視角以適應所有資料"""
        if not self.drone_visuals:
            return
        
        # 收集所有位置
        all_positions = []
        for visual in self.drone_visuals.values():
            if visual.position is not None:
                all_positions.append(to_cpu(visual.position))
            # 包含軌跡點
            for trail_pos in visual.trail_positions:
                all_positions.append(to_cpu(trail_pos))
        
        if not all_positions:
            return
        
        positions_array = np.array(all_positions)
        
        # 計算邊界
        min_coords = np.min(positions_array, axis=0)
        max_coords = np.max(positions_array, axis=0)
        
        # 添加邊距
        margin = 0.1
        ranges = max_coords - min_coords
        margin_values = ranges * margin
        
        # 設置軸範圍
        self.ax.set_xlim(min_coords[0] - margin_values[0], max_coords[0] + margin_values[0])
        self.ax.set_ylim(min_coords[1] - margin_values[1], max_coords[1] + margin_values[1])
        self.ax.set_zlim(min_coords[2] - margin_values[2], max_coords[2] + margin_values[2])
        
        if self.canvas:
            self.canvas.draw_idle()
        
        logger.info("🎯 視角已自動調整")

    def reset_view(self):
        """重置視角"""
        self.ax.view_init(elev=20, azim=45)
        self.fit_view_to_data()

    def _get_drone_color(self, index: int) -> str:
        """獲取無人機顏色"""
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'yellow', 'pink']
        return colors[index % len(colors)]

    def _color_to_rgba(self, color: str) -> Tuple[float, float, float, float]:
        """將顏色名稱轉換為RGBA值"""
        color_map = {
            'blue': (0.0, 0.0, 1.0, 1.0),
            'red': (1.0, 0.0, 0.0, 1.0),
            'green': (0.0, 1.0, 0.0, 1.0),
            'orange': (1.0, 0.5, 0.0, 1.0),
            'purple': (0.5, 0.0, 0.5, 1.0),
            'cyan': (0.0, 1.0, 1.0, 1.0),
            'yellow': (1.0, 1.0, 0.0, 1.0),
            'pink': (1.0, 0.75, 0.8, 1.0)
        }
        return color_map.get(color, (1.0, 1.0, 1.0, 1.0))

    def toggle_drone_visibility(self, drone_name: str, visible: bool):
        """切換無人機可見性"""
        if drone_name in self.drone_visuals:
            self.drone_visuals[drone_name].visible = visible

    def clear_trails(self):
        """清除所有軌跡"""
        for visual in self.drone_visuals.values():
            visual.trail_positions.clear()
        logger.info("🧹 軌跡已清除")

    def export_frame(self, filename: str, dpi: int = 300):
        """匯出當前幀為圖片"""
        try:
            self.fig.savefig(
                filename,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='black',
                edgecolor='white'
            )
            logger.info(f"💾 幀已匯出: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ 匯出幀失敗: {e}")
            return False

    def get_performance_stats(self) -> Dict[str, Any]:
        """獲取效能統計"""
        avg_fps = 1.0 / (np.mean(self.frame_times[-30:]) + 1e-6) if self.frame_times else 0
        
        return {
            "backend": self.render_settings.backend.value,
            "gpu_acceleration": is_gpu_enabled(),
            "current_fps": avg_fps,
            "target_fps": self.render_settings.fps_target,
            "total_frames": self.render_count,
            "drone_count": len(self.drone_visuals),
            "collision_lines": len(self.collision_lines),
            "waypoint_markers": len(self.waypoint_markers),
            "render_settings": {
                "trail_length": self.render_settings.trail_length,
                "point_size": self.render_settings.point_size,
                "anti_aliasing": self.render_settings.anti_aliasing,
                "batch_rendering": self.render_settings.batch_rendering
            }
        }

    def cleanup(self):
        """清理資源"""
        logger.info("🧹 清理視覺化資源...")
        
        # 停止動畫
        if self.animation:
            self.animation.event_source.stop()
        
        # 關閉執行緒池
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        # 清理GPU緩衝區
        if is_gpu_enabled():
            from utils.gpu_utils import compute_manager
            if hasattr(compute_manager, '_cupy'):
                compute_manager._cupy.get_default_memory_pool().free_all_blocks()
        
        # 清理matplotlib
        if self.fig:
            plt.close(self.fig)
        
        logger.info("✅ 視覺化資源清理完成")

# 向後相容的別名
PlotManager = GPUPlotManager