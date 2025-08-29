"""
控制面板GUI元件
提供專業的無人機模擬控制界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
import threading

from config.settings import UIConfig, SimulationConfig
from core.trajectory_calculator import FlightPhase

logger = logging.getLogger(__name__)

class StatusIndicator:
    """狀態指示器元件"""
    
    def __init__(self, parent: tk.Widget, label: str, color: str = "#888888"):
        """
        初始化狀態指示器
        
        Args:
            parent: 父級widget
            label: 標籤文字
            color: 初始顏色
        """
        self.frame = tk.Frame(parent, bg='#2d2d2d')
        self.frame.pack(fill=tk.X, pady=1)
        
        # 顏色點
        self.color_dot = tk.Label(self.frame, text="●", font=('Arial', 12),
                                 fg=color, bg='#2d2d2d')
        self.color_dot.pack(side=tk.LEFT)
        
        # 狀態標籤
        self.status_label = tk.Label(self.frame, text=label,
                                    fg='#888', bg='#2d2d2d', font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT, padx=(5, 0))
    
    def update(self, text: str, color: str):
        """更新狀態"""
        self.color_dot.configure(fg=color)
        self.status_label.configure(text=text, fg='#ffffff' if color != '#888' else '#888')


class CompactControlPanel:
    """
    緊湊控制面板
    提供所有必要的模擬控制功能
    """
    
    def __init__(self, parent: tk.Widget, config: UIConfig):
        """
        初始化控制面板
        
        Args:
            parent: 父級widget
            config: UI配置
        """
        self.parent = parent
        self.config = config
        
        # 回調函數
        self.callbacks: Dict[str, Callable] = {}
        
        # 控制變量
        self.time_var = tk.DoubleVar()
        self.speed_var = tk.DoubleVar(value=1.0)
        self.safety_var = tk.DoubleVar(value=5.0)
        
        # UI元件
        self.region_indicators: Dict[str, StatusIndicator] = {}
        self.play_button = None
        self.time_label = None
        self.time_slider = None
        self.status_text = None
        self.warning_text = None
        
        self._create_control_panel()
        
        logger.info("控制面板初始化完成")
    
    def _create_control_panel(self):
        """創建控制面板"""
        # 標題
        self._create_title_section()
        
        # 檔案操作
        self._create_file_section()
        
        # 無人機狀態指示
        self._create_drone_status_section()
        
        # 播放控制
        self._create_playback_section()
        
        # 狀態信息
        self._create_status_section()
        
        # 碰撞警告
        self._create_warning_section()
    
    def _create_title_section(self):
        """創建標題區域"""
        title_frame = tk.Frame(self.parent, bg='#2d2d2d')
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(title_frame, text="🚁 無人機群飛模擬器",
                              font=('Arial', 14, 'bold'),
                              fg='#00d4aa', bg='#2d2d2d')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="GPU加速版 v2.0",
                                 font=('Arial', 8),
                                 fg='#888888', bg='#2d2d2d')
        subtitle_label.pack()
    
    def _create_file_section(self):
        """創建檔案操作區域"""
        file_frame = tk.LabelFrame(self.parent, text="📁 任務",
                                  fg='white', bg='#2d2d2d',
                                  font=('Arial', 10, 'bold'))
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_frame = tk.Frame(file_frame, bg='#2d2d2d')
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # QGC載入按鈕
        qgc_btn = tk.Button(btn_frame, text="QGC",
                           command=self._on_load_qgc,
                           bg='#28a745', fg='white',
                           font=('Arial', 8, 'bold'), width=8)
        qgc_btn.pack(side=tk.LEFT, padx=1)
        
        # CSV載入按鈕  
        csv_btn = tk.Button(btn_frame, text="CSV",
                           command=self._on_load_csv,
                           bg='#007bff', fg='white',
                           font=('Arial', 8, 'bold'), width=8)
        csv_btn.pack(side=tk.LEFT, padx=1)
        
        # 測試任務按鈕
        test_btn = tk.Button(btn_frame, text="測試",
                           command=self._on_create_test,
                           bg='#6f42c1', fg='white',
                           font=('Arial', 8, 'bold'), width=8)
        test_btn.pack(side=tk.LEFT, padx=1)
    
    def _create_drone_status_section(self):
        """創建無人機狀態區域"""
        region_frame = tk.LabelFrame(self.parent, text="🎨 無人機狀態",
                                    fg='white', bg='#2d2d2d',
                                    font=('Arial', 10, 'bold'))
        region_frame.pack(fill=tk.X, padx=10, pady=5)
        
        colors = ['#FF4444', '#44FF44', '#4444FF', '#FFFF44',
                 '#FF44FF', '#44FFFF', '#FFAA44', '#AA44FF']
        
        for i in range(8):  # 支持最多8架無人機
            drone_id = f'drone_{i+1}'
            indicator = StatusIndicator(region_frame, "待機", colors[i])
            self.region_indicators[drone_id] = indicator
    
    def _create_playback_section(self):
        """創建播放控制區域"""
        play_frame = tk.LabelFrame(self.parent, text="▶️ 控制",
                                  fg='white', bg='#2d2d2d',
                                  font=('Arial', 10, 'bold'))
        play_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 播放按鈕行
        btn_row = tk.Frame(play_frame, bg='#2d2d2d')
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        
        self.play_button = tk.Button(btn_row, text="▶",
                                   command=self._on_toggle_play,
                                   bg='#28a745', fg='white',
                                   font=('Arial', 12, 'bold'), width=3)
        self.play_button.pack(side=tk.LEFT, padx=1)
        
        stop_btn = tk.Button(btn_row, text="⏹",
                           command=self._on_stop,
                           bg='#dc3545', fg='white',
                           font=('Arial', 12, 'bold'), width=3)
        stop_btn.pack(side=tk.LEFT, padx=1)
        
        reset_btn = tk.Button(btn_row, text="⏮",
                            command=self._on_reset,
                            bg='#ffc107', fg='black',
                            font=('Arial', 12, 'bold'), width=3)
        reset_btn.pack(side=tk.LEFT, padx=1)
        
        export_btn = tk.Button(btn_row, text="💾",
                             command=self._on_export,
                             bg='#17a2b8', fg='white',
                             font=('Arial', 12, 'bold'), width=3)
        export_btn.pack(side=tk.LEFT, padx=1)
        
        # 時間顯示
        self.time_label = tk.Label(play_frame, text="00:00 / 00:00",
                                  fg='#00d4aa', bg='#2d2d2d',
                                  font=('Arial', 11, 'bold'))
        self.time_label.pack(pady=3)
        
        # 時間滑桿
        self.time_slider = tk.Scale(play_frame, from_=0, to=100,
                                   orient=tk.HORIZONTAL,
                                   variable=self.time_var,
                                   command=self._on_time_change,
                                   bg='#2d2d2d', fg='white',
                                   highlightbackground='#2d2d2d',
                                   troughcolor='#404040',
                                   length=250, resolution=0.1)
        self.time_slider.pack(fill=tk.X, padx=5, pady=3)
        
        # 速度和安全距離控制
        self._create_parameter_controls(play_frame)
    
    def _create_parameter_controls(self, parent):
        """創建參數控制"""
        settings_frame = tk.Frame(parent, bg='#2d2d2d')
        settings_frame.pack(fill=tk.X, padx=5, pady=3)
        
        # 速度控制
        tk.Label(settings_frame, text="速度:",
                fg='white', bg='#2d2d2d', font=('Arial', 8)).pack(side=tk.LEFT)
        
        speed_scale = tk.Scale(settings_frame, from_=0.1, to=5.0,
                              resolution=0.1, orient=tk.HORIZONTAL,
                              variable=self.speed_var,
                              command=self._on_speed_change,
                              bg='#2d2d2d', fg='white', length=100)
        speed_scale.pack(side=tk.LEFT, padx=5)
        
        # 安全距離控制
        tk.Label(settings_frame, text="安全:",
                fg='white', bg='#2d2d2d', font=('Arial', 8)).pack(side=tk.LEFT)
        
        safety_scale = tk.Scale(settings_frame, from_=2.0, to=15.0,
                               resolution=0.5, orient=tk.HORIZONTAL,
                               variable=self.safety_var,
                               command=self._on_safety_change,
                               bg='#2d2d2d', fg='white', length=100)
        safety_scale.pack(side=tk.LEFT, padx=5)
    
    def _create_status_section(self):
        """創建狀態信息區域"""
        status_frame = tk.LabelFrame(self.parent, text="📊 狀態信息",
                                    fg='white', bg='#2d2d2d',
                                    font=('Arial', 11, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 創建文字區域和滾動條
        text_container = tk.Frame(status_frame, bg='#2d2d2d')
        text_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.status_text = tk.Text(text_container, bg='#1a1a1a', fg='#00d4aa',
                                  font=('Consolas', 11),
                                  height=self.config.status_text_height)
        
        status_scroll = ttk.Scrollbar(text_container, orient="vertical",
                                     command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scroll.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_warning_section(self):
        """創建警告區域"""
        warning_frame = tk.LabelFrame(self.parent, text="⚠️ 碰撞警告",
                                     fg='white', bg='#2d2d2d',
                                     font=('Arial', 11, 'bold'))
        warning_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.warning_text = tk.Text(warning_frame,
                                   height=self.config.warning_text_height,
                                   bg='#1a1a1a', fg='#ff5722',
                                   font=('Consolas', 10))
        self.warning_text.pack(fill=tk.X, padx=5, pady=5)
    
    def register_callback(self, event: str, callback: Callable):
        """
        註冊回調函數
        
        Args:
            event: 事件名稱
            callback: 回調函數
        """
        self.callbacks[event] = callback
        logger.debug(f"註冊回調: {event}")
    
    def _trigger_callback(self, event: str, *args, **kwargs):
        """觸發回調函數"""
        if event in self.callbacks:
            try:
                self.callbacks[event](*args, **kwargs)
            except Exception as e:
                logger.error(f"回調函數 {event} 執行失敗: {e}")
        else:
            logger.warning(f"未找到回調函數: {event}")
    
    def _on_load_qgc(self):
        """QGC檔案載入事件"""
        files = filedialog.askopenfilenames(
            title="選擇QGC WP檔案",
            filetypes=[("Waypoint files", "*.waypoints"), ("All files", "*.*")]
        )
        if files:
            self._trigger_callback('load_qgc', files)
    
    def _on_load_csv(self):
        """CSV檔案載入事件"""
        files = filedialog.askopenfilenames(
            title="選擇CSV檔案",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if files:
            self._trigger_callback('load_csv', files)
    
    def _on_create_test(self):
        """創建測試任務事件"""
        self._trigger_callback('create_test')
    
    def _on_toggle_play(self):
        """播放/暫停切換事件"""
        self._trigger_callback('toggle_play')
    
    def _on_stop(self):
        """停止事件"""
        self._trigger_callback('stop')
    
    def _on_reset(self):
        """重置事件"""
        self._trigger_callback('reset')
    
    def _on_export(self):
        """導出事件"""
        self._trigger_callback('export')
    
    def _on_time_change(self, value):
        """時間改變事件"""
        self._trigger_callback('time_change', float(value))
    
    def _on_speed_change(self, value):
        """速度改變事件"""
        self._trigger_callback('speed_change', float(value))
    
    def _on_safety_change(self, value):
        """安全距離改變事件"""
        self._trigger_callback('safety_change', float(value))
    
    def update_play_button(self, is_playing: bool):
        """更新播放按鈕狀態"""
        if is_playing:
            self.play_button.configure(text="⏸", bg='#ffc107', fg='black')
        else:
            self.play_button.configure(text="▶", bg='#28a745', fg='white')
    
    def update_time_display(self, current_time: float, max_time: float):
        """更新時間顯示"""
        current_min = int(current_time // 60)
        current_sec = int(current_time % 60)
        max_min = int(max_time // 60)
        max_sec = int(max_time % 60)
        
        time_text = f"{current_min:02d}:{current_sec:02d} / {max_min:02d}:{max_sec:02d}"
        self.time_label.configure(text=time_text)
        
        # 更新滑桿
        if max_time > 0:
            self.time_slider.configure(to=max_time)
            if not self.time_slider.active:  # 避免拖拽時跳動
                self.time_var.set(current_time)
    
    def update_drone_status(self, drone_id: str, status: str, color: str):
        """
        更新無人機狀態
        
        Args:
            drone_id: 無人機ID  
            status: 狀態文字
            color: 狀態顏色
        """
        if drone_id in self.region_indicators:
            self.region_indicators[drone_id].update(status, color)
    
    def update_status_text(self, drone_states: Dict[str, Any]):
        """更新狀態文字"""
        self.status_text.delete(1.0, tk.END)
        
        if not drone_states:
            self.status_text.insert(tk.END, "📄 無載入的無人機\n\n")
            self.status_text.insert(tk.END, "請載入QGC或CSV檔案來開始模擬\n")
            return
        
        for drone_id, drone_state in drone_states.items():
            trajectory = drone_state.trajectory
            current_pos = drone_state.current_position
            
            status = f"🚁 {drone_id}:\n"
            status += f"   📍 起飛位置: {drone_state.takeoff_position}\n"
            status += f"   📊 航點數: {len(drone_state.waypoints)}\n"
            
            if trajectory:
                base_time = trajectory[-1].time
                loiter_time = sum(delay.get('duration', 0) 
                                for delay in drone_state.loiter_delays)
                total_time = base_time + loiter_time
                status += f"   ⏱️  總時長: {total_time:.1f}s"
                if loiter_time > 0:
                    status += f" (含等待 {loiter_time:.1f}s)"
                status += "\n"
            
            if current_pos:
                phase_text = {
                    FlightPhase.TAXI: "地面滑行",
                    FlightPhase.TAKEOFF: "起飛爬升",
                    FlightPhase.HOVER: "懸停等待",
                    FlightPhase.AUTO: "自動任務",
                    FlightPhase.LOITER: "等待避讓"
                }.get(current_pos.get('phase', FlightPhase.AUTO), "執行中")
                
                status += f"   🎯 當前階段: {phase_text}\n"
                status += f"   📍 位置: ({current_pos['x']:.1f}, {current_pos['y']:.1f}, {current_pos['z']:.1f})\n"
            else:
                status += f"   🎯 狀態: 待機\n"
            
            status += "\n"
            self.status_text.insert(tk.END, status)
        
        # 自動滾動到底部
        self.status_text.see(tk.END)
    
    def update_warning_text(self, warnings: List[Dict]):
        """更新警告文字"""
        self.warning_text.delete(1.0, tk.END)
        
        if not warnings:
            self.warning_text.insert(tk.END, "✅ 飛行安全，無碰撞風險\n")
            self.warning_text.tag_add("safe", "1.0", tk.END)
            self.warning_text.tag_config("safe", foreground="#4caf50")
        else:
            self.warning_text.insert(tk.END, f"⚠️ 檢測到 {len(warnings)} 個碰撞警告!\n\n")
            
            for i, warning in enumerate(warnings, 1):
                drone1 = warning.get('drone1_id') or warning.get('drone1', 'Unknown')
                drone2 = warning.get('drone2_id') or warning.get('drone2', 'Unknown')
                distance = warning.get('distance', 0)
                severity = warning.get('severity', 'warning')
                
                severity_text = "🚨 嚴重" if severity == 'critical' else "⚠️ 警告"
                text = f"{severity_text} {i}:\n"
                text += f"  🔄 {drone1} ↔ {drone2}\n"
                text += f"  📏 距離: {distance:.2f}m\n"
                text += f"  ⏰ 時間: {warning.get('time', 0):.1f}s\n\n"
                
                self.warning_text.insert(tk.END, text)
            
            self.warning_text.tag_add("danger", "1.0", tk.END)
            self.warning_text.tag_config("danger", foreground="#ff5722",
                                        font=('Consolas', 10, 'bold'))
        
        # 自動滾動到底部
        self.warning_text.see(tk.END)
    
    def show_message(self, title: str, message: str, msg_type: str = "info"):
        """
        顯示消息對話框
        
        Args:
            title: 標題
            message: 消息內容
            msg_type: 消息類型 ("info", "warning", "error")
        """
        if msg_type == "info":
            messagebox.showinfo(title, message)
        elif msg_type == "warning":
            messagebox.showwarning(title, message)
        elif msg_type == "error":
            messagebox.showerror(title, message)
    
    def get_export_directory(self) -> Optional[str]:
        """獲取導出目錄"""
        return filedialog.askdirectory(title="選擇導出目錄")
    
    def disable_controls(self, disabled: bool = True):
        """
        禁用/啟用控制項
        
        Args:
            disabled: 是否禁用
        """
        state = 'disabled' if disabled else 'normal'
        
        # 禁用主要控制按鈕（但保留停止按鈕）
        widgets_to_disable = [
            self.play_button,
            self.time_slider
        ]
        
        for widget in widgets_to_disable:
            if widget:
                widget.configure(state=state)
    
    def set_parameter_values(self, speed: float, safety_distance: float):
        """設置參數值"""
        self.speed_var.set(speed)
        self.safety_var.set(safety_distance)