"""
無人機群飛模擬器主應用程式 - GPU加速版本
整合所有模組，提供完整的GUI應用程式
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import logging
import threading
import time
import argparse
from typing import Dict, List, Optional, Any

# 確保模組路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU支援可用")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU支援不可用，將使用CPU模式")

# 導入自定義模組
from config.settings import get_simulation_config, get_config_manager, BackendType
from simulator.advanced_simulator_main import AdvancedDroneSimulator
from gui.gui_advanced_plotter import Advanced3DPlotter
from gui.control_panel import CompactControlPanel
from utils.gpu_utils import GPUSystemChecker, setup_gpu_environment

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/simulator.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DroneSimulatorApp:
    """
    無人機模擬器主應用程式
    """
    
    def __init__(self):
        """初始化應用程式"""
        # 載入配置
        self.config = get_simulation_config()
        
        # 檢查並設置GPU環境
        self._setup_gpu_environment()
        
        # 創建主視窗
        self.root = tk.Tk()
        self._setup_main_window()
        
        # 初始化核心組件
        self.simulator = None
        self.plotter = None
        self.control_panel = None
        
        # 模擬狀態
        self.is_running = False
        self.animation_thread = None
        self.last_update_time = time.time()
        
        # 初始化GUI
        self._initialize_components()
        self._setup_callbacks()
        self._setup_menu()
        self._bind_keyboard_shortcuts()
        
        logger.info("無人機模擬器應用程式初始化完成")
    
    def _setup_gpu_environment(self):
        """設置GPU環境"""
        if GPU_AVAILABLE and self.config.backend.use_gpu:
            try:
                # 檢查GPU系統
                gpu_checker = GPUSystemChecker()
                gpu_info = gpu_checker.get_gpu_info()
                
                if gpu_info['available']:
                    setup_gpu_environment(
                        device_id=self.config.backend.gpu_device_id,
                        memory_pool=self.config.backend.memory_pool
                    )
                    logger.info(f"GPU環境設置完成: {gpu_info['name']}")
                else:
                    logger.warning("GPU不可用，切換到CPU模式")
                    self.config.backend.use_gpu = False
                    
            except Exception as e:
                logger.error(f"GPU環境設置失敗: {e}")
                self.config.backend.use_gpu = False
    
    def _setup_main_window(self):
        """設置主視窗"""
        self.root.title(self.config.ui.window_title)
        self.root.geometry(self.config.ui.window_geometry)
        self.root.configure(bg='#1e1e1e')
        
        # 最大化視窗
        if self.config.ui.maximize_on_start:
            try:
                self.root.state('zoomed')  # Windows
            except:
                try:
                    self.root.attributes('-zoomed', True)  # Linux
                except:
                    pass  # macOS或其他系統
        
        # 設置圖標和關閉事件
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # 設置主容器
        self.main_container = tk.Frame(self.root, bg='#1e1e1e')
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _initialize_components(self):
        """初始化主要組件"""
        # 創建左側控制面板
        control_frame = tk.Frame(self.main_container, bg='#2d2d2d', 
                                width=self.config.ui.control_panel_width)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # 創建右側3D視圖容器
        plot_container = tk.Frame(self.main_container, bg='#1e1e1e')
        plot_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 初始化控制面板
        self.control_panel = CompactControlPanel(control_frame, self.config.ui)
        
        # 初始化3D繪圖器
        self.plotter = Advanced3DPlotter(plot_container, self.config.visualization, 
                                        self.config.backend.use_gpu)
        
        # 初始化模擬器
        self.simulator = AdvancedDroneSimulator(self.config)
        
        logger.info("主要組件初始化完成")
    
    def _setup_callbacks(self):
        """設置回調函數"""
        callbacks = {
            'load_qgc': self._load_qgc_files,
            'load_csv': self._load_csv_files,
            'create_test': self._create_test_mission,
            'toggle_play': self._toggle_playback,
            'stop': self._stop_simulation,
            'reset': self._reset_simulation,
            'export': self._export_missions,
            'time_change': self._on_time_change,
            'speed_change': self._on_speed_change,
            'safety_change': self._on_safety_change
        }
        
        for event, callback in callbacks.items():
            self.control_panel.register_callback(event, callback)
    
    def _setup_menu(self):
        """設置選單欄"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 檔案選單
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="檔案", menu=file_menu)
        file_menu.add_command(label="載入QGC檔案", command=self._load_qgc_files)
        file_menu.add_command(label="載入CSV檔案", command=self._load_csv_files)
        file_menu.add_separator()
        file_menu.add_command(label="創建測試任務", command=self._create_test_mission)
        file_menu.add_separator()
        file_menu.add_command(label="導出修正任務", command=self._export_missions)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_closing)
        
        # 視圖選單
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="視圖", menu=view_menu)
        view_menu.add_command(label="俯視角", command=lambda: self.plotter.set_view("top"))
        view_menu.add_command(label="側視角", command=lambda: self.plotter.set_view("side"))
        view_menu.add_command(label="3D視角", command=lambda: self.plotter.set_view("3d"))
        view_menu.add_separator()
        view_menu.add_command(label="重置視圖", command=self.plotter.reset_view)
        
        # 模擬選單
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="模擬", menu=sim_menu)
        sim_menu.add_command(label="播放/暫停", command=self._toggle_playback)
        sim_menu.add_command(label="停止", command=self._stop_simulation)
        sim_menu.add_command(label="重置", command=self._reset_simulation)
        
        # 工具選單
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="系統信息", command=self._show_system_info)
        tools_menu.add_command(label="性能統計", command=self._show_performance_stats)
        tools_menu.add_command(label="GPU設定", command=self._show_gpu_settings)
        
        # 幫助選單
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="幫助", menu=help_menu)
        help_menu.add_command(label="快捷鍵", command=self._show_shortcuts)
        help_menu.add_command(label="關於", command=self._show_about)
    
    def _bind_keyboard_shortcuts(self):
        """綁定鍵盤快捷鍵"""
        def on_key_press(event):
            key = event.keysym.lower()
            
            if key == 'space':
                self._toggle_playback()
            elif key == 'r':
                self._reset_simulation()
            elif key == 's':
                self._stop_simulation()
            elif key in ['1', '2', '3']:
                views = {'1': 'top', '2': 'side', '3': '3d'}
                self.plotter.set_view(views[key])
            elif key == 'escape':
                self._on_closing()
        
        self.root.bind('<KeyPress>', on_key_press)
        self.root.focus_set()
    
    def _load_qgc_files(self):
        """載入QGC檔案"""
        logger.info("使用者請求載入QGC檔案")
        # 由控制面板處理檔案選擇，這裡只需要處理載入邏輯
        # 實際的檔案對話框在控制面板中處理
        pass
    
    def _load_csv_files(self):
        """載入CSV檔案"""  
        logger.info("使用者請求載入CSV檔案")
        pass
    
    def _create_test_mission(self):
        """創建測試任務"""
        logger.info("創建測試任務")
        try:
            success = self.simulator.create_test_mission()
            if success:
                self._update_display()
                self.control_panel.show_message("任務創建", "測試任務創建成功", "info")
            else:
                self.control_panel.show_message("任務創建", "測試任務創建失敗", "error")
        except Exception as e:
            logger.error(f"創建測試任務失敗: {e}")
            self.control_panel.show_message("錯誤", f"創建測試任務失敗: {str(e)}", "error")
    
    def _toggle_playback(self):
        """切換播放狀態"""
        if not self.simulator.drones:
            self.control_panel.show_message("播放控制", "請先載入無人機任務", "warning")
            return
        
        if self.simulator.is_playing:
            self._pause_simulation()
        else:
            self._start_simulation()
    
    def _start_simulation(self):
        """開始模擬"""
        logger.info("開始模擬")
        self.simulator.play_simulation()
        self.control_panel.update_play_button(True)
        
        # 啟動動畫線程
        if not self.is_running:
            self.is_running = True
            self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
            self.animation_thread.start()
    
    def _pause_simulation(self):
        """暫停模擬"""
        logger.info("暫停模擬")
        self.simulator.pause_simulation()
        self.control_panel.update_play_button(False)
    
    def _stop_simulation(self):
        """停止模擬"""
        logger.info("停止模擬")
        self.simulator.stop_simulation()
        self.control_panel.update_play_button(False)
        self.is_running = False
    
    def _reset_simulation(self):
        """重置模擬"""
        logger.info("重置模擬")
        self.simulator.reset_simulation()
        self.control_panel.update_play_button(False)
        self.is_running = False
        self._update_display()
    
    def _export_missions(self):
        """導出修正後的任務"""
        if not self.simulator.modified_missions:
            self.control_panel.show_message("導出", "沒有修正後的任務需要導出", "info")
            return
        
        export_dir = self.control_panel.get_export_directory()
        if export_dir:
            try:
                results = self.simulator.export_modified_missions(export_dir)
                if results:
                    message = f"成功導出 {len(results)} 個任務檔案至:\n{export_dir}"
                    self.control_panel.show_message("導出成功", message, "info")
                else:
                    self.control_panel.show_message("導出失敗", "沒有檔案被導出", "error")
            except Exception as e:
                logger.error(f"導出任務失敗: {e}")
                self.control_panel.show_message("導出錯誤", f"導出失敗: {str(e)}", "error")
    
    def _on_time_change(self, time_value: float):
        """時間改變處理"""
        if not self.simulator.is_playing:
            self.simulator.seek_to_time(time_value)
            self._update_display()
    
    def _on_speed_change(self, speed: float):
        """速度改變處理"""
        self.simulator.set_time_scale(speed)
        logger.debug(f"模擬速度設置為: {speed}x")
    
    def _on_safety_change(self, distance: float):
        """安全距離改變處理"""
        self.simulator.collision_system.config.safety_distance = distance
        logger.debug(f"安全距離設置為: {distance}m")
    
    def _animation_loop(self):
        """動畫循環（在獨立線程中運行）"""
        while self.is_running:
            try:
                current_time = time.time()
                dt = current_time - self.last_update_time
                self.last_update_time = current_time
                
                # 更新模擬狀態
                update_result = self.simulator.update_simulation(dt)
                
                # 在主線程中更新GUI
                if update_result['updated']:
                    self.root.after_idle(self._update_gui_from_simulation, update_result)
                
                # 控制更新頻率
                time.sleep(1.0 / 30.0)  # 30 FPS
                
            except Exception as e:
                logger.error(f"動畫循環錯誤: {e}")
                self.is_running = False
                break
    
    def _update_gui_from_simulation(self, update_result: Dict[str, Any]):
        """從模擬結果更新GUI（在主線程中執行）"""
        try:
            # 更新時間顯示
            self.control_panel.update_time_display(
                update_result['current_time'], 
                self.simulator.max_time
            )
            
            # 更新無人機狀態指示器
            for drone_id, drone_state in self.simulator.drones.items():
                indicator_id = drone_id.lower()
                current_pos = drone_state.current_position
                
                if current_pos:
                    phase = current_pos.get('phase', 'auto')
                    status_text = f"✓ {phase}"
                    color = '#4caf50' if phase in ['auto', 'cruise'] else '#ff9800'
                else:
                    status_text = "待機"
                    color = '#888888'
                
                self.control_panel.update_drone_status(indicator_id, status_text, color)
            
            # 更新狀態文字
            self.control_panel.update_status_text(self.simulator.drones)
            
            # 更新警告文字
            self.control_panel.update_warning_text(update_result.get('collision_warnings', []))
            
            # 更新3D繪圖
            self.plotter.update_plot(
                self.simulator.drones,
                update_result.get('collision_warnings', []),
                update_result['current_time']
            )
            
            # 檢查模擬是否結束
            if not self.simulator.is_playing and self.is_running:
                self.is_running = False
                self.control_panel.update_play_button(False)
                logger.info("模擬已完成")
            
        except Exception as e:
            logger.error(f"更新GUI失敗: {e}")
    
    def _update_display(self):
        """更新顯示（靜態更新）"""
        try:
            # 更新狀態文字
            self.control_panel.update_status_text(self.simulator.drones)
            
            # 更新時間顯示
            self.control_panel.update_time_display(
                self.simulator.current_time,
                self.simulator.max_time
            )
            
            # 更新3D繪圖
            if self.simulator.drones:
                # 獲取當前位置
                positions = {}
                for drone_id in self.simulator.drones:
                    pos = self.simulator.get_drone_position_at_time(drone_id, self.simulator.current_time)
                    if pos:
                        self.simulator.drones[drone_id].current_position = pos
                
                self.plotter.update_plot(
                    self.simulator.drones,
                    [],  # 靜態更新時不檢查碰撞
                    self.simulator.current_time
                )
                
        except Exception as e:
            logger.error(f"更新顯示失敗: {e}")
    
    def _show_system_info(self):
        """顯示系統信息"""
        try:
            from utils.gpu_utils import GPUSystemChecker
            
            gpu_checker = GPUSystemChecker()
            system_info = gpu_checker.get_system_info()
            coord_info = self.simulator.coordinate_system.get_system_info()
            collision_info = self.simulator.collision_system.get_system_status()
            
            info_text = f"""系統信息:

Python版本: {system_info['python_version']}
平台: {system_info['platform']}

GPU信息:
- 可用: {'是' if system_info['gpu_available'] else '否'}
- 驅動版本: {system_info.get('gpu_driver', 'N/A')}
- 記憶體: {system_info.get('gpu_memory', 'N/A')}

坐標系統:
- 原點設置: {'是' if coord_info['origin_set'] else '否'}
- GPU加速: {'啟用' if coord_info['gpu_enabled'] else '禁用'}

碰撞系統:
- 活動警告: {collision_info['active_warnings']}
- 安全距離: {collision_info['safety_distance']}m
- GPU加速: {'啟用' if collision_info['gpu_enabled'] else '禁用'}

模擬狀態:
- 載入無人機: {len(self.simulator.drones)}
- 修正任務: {len(self.simulator.modified_missions)}
"""
            
            messagebox.showinfo("系統信息", info_text)
            
        except Exception as e:
            logger.error(f"獲取系統信息失敗: {e}")
            messagebox.showerror("錯誤", f"無法獲取系統信息: {str(e)}")
    
    def _show_performance_stats(self):
        """顯示性能統計"""
        status = self.simulator.get_simulation_status()
        
        stats_text = f"""性能統計:

模擬狀態:
- FPS: {status['fps']:.1f}
- 時間縮放: {status['time_scale']}x
- GPU記憶體: {status['gpu_memory_mb']:.1f} MB

載入數據:
- 無人機數量: {status['num_drones']}
- 碰撞警告: {status['collision_warnings']}
- 修正任務: {status['modified_missions']}

後端信息:
- GPU可用: {'是' if status['gpu_available'] else '否'}
- GPU使用: {'是' if status['use_gpu'] else '否'}
"""
        
        messagebox.showinfo("性能統計", stats_text)
    
    def _show_gpu_settings(self):
        """顯示GPU設定對話框"""
        # 創建簡單的GPU設定對話框
        settings_window = tk.Toplevel(self.root)
        settings_window.title("GPU設定")
        settings_window.geometry("400x300")
        settings_window.configure(bg='#2d2d2d')
        
        # GPU狀態
        status_frame = tk.LabelFrame(settings_window, text="GPU狀態", 
                                    fg='white', bg='#2d2d2d')
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        gpu_available = "是" if GPU_AVAILABLE else "否"
        gpu_using = "是" if self.config.backend.use_gpu else "否"
        
        tk.Label(status_frame, text=f"GPU可用: {gpu_available}", 
                fg='white', bg='#2d2d2d').pack(anchor=tk.W, padx=5, pady=2)
        tk.Label(status_frame, text=f"GPU使用: {gpu_using}", 
                fg='white', bg='#2d2d2d').pack(anchor=tk.W, padx=5, pady=2)
        
        # 設定選項
        options_frame = tk.LabelFrame(settings_window, text="設定選項", 
                                     fg='white', bg='#2d2d2d')
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(options_frame, text="性能優化", 
                 command=self.simulator.optimize_performance,
                 bg='#007bff', fg='white').pack(pady=5)
        
        tk.Button(options_frame, text="記憶體清理",
                 command=lambda: self.simulator.coordinate_system.optimize_memory_usage(),
                 bg='#28a745', fg='white').pack(pady=5)
    
    def _show_shortcuts(self):
        """顯示快捷鍵幫助"""
        shortcuts_text = """快捷鍵說明:

播放控制:
  空白鍵        播放/暫停
  R            重置模擬
  S            停止模擬

視角控制:
  1            俯視角
  2            側視角  
  3            3D視角

縮放控制:
  滑鼠滾輪     縮放視圖
  滑鼠拖拽     旋轉視圖
  雙擊         重置視圖

其他:
  Ctrl+O       載入檔案
  Ctrl+S       導出任務
  F1           顯示幫助
  ESC          退出程式
"""
        
        messagebox.showinfo("快捷鍵說明", shortcuts_text)
    
    def _show_about(self):
        """顯示關於信息"""
        about_text = f"""進階無人機群飛模擬器 - GPU版本 v{self.config.version}

專業級特色:
• GPU/CPU混合加速運算
• 精確碰撞檢測與軌跡分析
• 智能LOITER避讓系統  
• QGC任務檔案自動修正
• 專業級3D視覺化
• 支援多種檔案格式導入

安全系統:
• 實時碰撞檢測 (每0.1秒)
• 數字小優先權規則
• 自動LOITER延遲插入
• 修正後任務檔案導出

技術規格:
• 真實地理坐標轉換
• 高性能3D渲染引擎
• 模組化系統架構
• 專業GUI設計

開發: 無人機路徑規劃實驗室
支援: GPU加速並行計算
"""
        
        messagebox.showinfo("關於 - 進階無人機群飛模擬器", about_text)
    
    def _on_closing(self):
        """程式關閉處理"""
        logger.info("使用者請求關閉應用程式")
        
        # 停止模擬
        self.is_running = False
        if self.simulator:
            self.simulator.cleanup()
        
        # 清理GUI資源
        if self.plotter:
            self.plotter.cleanup()
        
        # 關閉視窗
        self.root.quit()
        self.root.destroy()
        
        logger.info("應用程式已關閉")
    
    def run(self):
        """運行應用程式"""
        logger.info("啟動無人機模擬器應用程式")
        
        # 顯示啟動信息
        self.control_panel.show_message(
            "歡迎使用",
            "進階無人機群飛模擬器 GPU版本\n\n請載入QGC或CSV檔案開始模擬",
            "info"
        )
        
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("使用者中斷程式")
            self._on_closing()
        except Exception as e:
            logger.error(f"應用程式運行錯誤: {e}")
            messagebox.showerror("嚴重錯誤", f"應用程式運行時發生錯誤:\n{str(e)}")
            self._on_closing()


def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='無人機群飛模擬器 GPU版本')
    
    parser.add_argument('--backend', choices=['auto', 'gpu', 'cpu'], 
                       default='auto', help='計算後端選擇')
    parser.add_argument('--device', type=int, default=0, 
                       help='GPU設備ID')
    parser.add_argument('--test', action='store_true', 
                       help='運行性能測試')
    parser.add_argument('--debug', action='store_true', 
                       help='啟用調試模式')
    parser.add_argument('--config', type=str, 
                       help='自定義配置檔案路徑')
    
    return parser.parse_args()


def main():
    """主函數"""
    print("無人機群飛模擬器 GPU版本啟動中...")
    
    try:
        # 解析命令行參數
        args = parse_arguments()
        
        # 設置調試模式
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("調試模式已啟用")
        
        # 更新後端設定
        config_manager = get_config_manager()
        if args.backend != 'auto':
            backend_type = BackendType.GPU if args.backend == 'gpu' else BackendType.CPU
            config_manager.update_backend_config(
                backend_type, 
                use_gpu=(args.backend == 'gpu'),
                gpu_device=args.device
            )
        
        # 載入自定義配置
        if args.config and os.path.exists(args.config):
            config_manager.load_config(args.config)
            logger.info(f"載入自定義配置: {args.config}")
        
        # 運行性能測試
        if args.test:
            from utils.gpu_utils import run_performance_benchmark
            logger.info("運行性能測試...")
            run_performance_benchmark()
            return
        
        # 確保日誌目錄存在
        os.makedirs('logs', exist_ok=True)
        os.makedirs('exports', exist_ok=True)
        
        # 設置matplotlib後端
        import matplotlib
        matplotlib.use('TkAgg')
        
        # 中文字體支援
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 高性能設置
        plt.rcParams['path.simplify'] = True
        plt.rcParams['path.simplify_threshold'] = 0.1
        plt.rcParams['agg.path.chunksize'] = 10000
        
        # 創建並運行應用程式
        app = DroneSimulatorApp()
        app.run()
        
    except ImportError as e:
        print(f"缺少依賴庫: {e}")
        print("基本依賴: pip install matplotlib pandas numpy")
        print("GPU依賴: pip install cupy-cuda12x")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"程式啟動失敗: {e}")
        print(f"程式啟動失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()