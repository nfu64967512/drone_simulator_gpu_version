#!/usr/bin/env python3
"""
無人機群模擬器主程式 - 完整功能版
支持GPU/CPU後端選擇、3D軌跡模擬、碰撞檢測、檔案導入等完整功能
"""
import sys
import os
import argparse
import tkinter as tk
from tkinter import messagebox, ttk
import logging
import json
from pathlib import Path

# 確保專案根目錄在Python路徑中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 導入配置和工具
from config.settings import settings, ComputeBackend, set_compute_backend, get_compute_backend_info
from utils.logging_config import setup_logging

# 設置日誌
logger = setup_logging()

class BackendSelector:
    """後端選擇對話框 - 增強版"""
    
    def __init__(self):
        self.selected_backend = None
        self.selected_device = 0
        self.root = None
        self.result = None

    def show_selection_dialog(self):
        """顯示後端選擇對話框"""
        self.root = tk.Tk()
        self.root.title("無人機群模擬器 - 計算後端選擇")
        self.root.geometry("650x750")
        self.root.resizable(True, True)
        
        # 設置UI樣式
        self._setup_ui_style()
        
        # 創建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        title_label = ttk.Label(
            main_frame, 
            text="無人機群模擬器 - 專業版",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 功能介紹
        self._create_feature_intro(main_frame)
        
        # 後端選擇區域
        self._create_backend_selection(main_frame)
        
        # GPU資訊顯示
        self._create_gpu_info_section(main_frame)
        
        # 進階設定
        self._create_advanced_settings(main_frame)
        
        # 按鈕區域
        self._create_buttons(main_frame)
        
        # 檢測可用後端
        self._detect_available_backends()
        
        # 運行對話框
        self.root.mainloop()
        
        return self.result

    def _setup_ui_style(self):
        """設置UI樣式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 自訂樣式
        style.configure('Title.TLabel', font=('Arial', 12, 'bold'))
        style.configure('Info.TLabel', font=('Arial', 9))
        style.configure('Feature.TLabel', font=('Arial', 10), foreground='blue')

    def _create_feature_intro(self, parent):
        """創建功能介紹區域"""
        intro_frame = ttk.LabelFrame(parent, text="核心功能", padding="10")
        intro_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        features = [
            "✈️ 3D即時軌跡模擬與視覺化",
            "🛡️ GPU加速碰撞檢測與避讓",
            "📁 支援QGC/CSV任務檔案導入",
            "🎯 智能任務修改與導出",
            "⚡ 高性能GPU/CPU混合計算",
            "📊 即時性能監控與統計"
        ]
        
        for i, feature in enumerate(features):
            ttk.Label(intro_frame, text=feature, style='Feature.TLabel').grid(
                row=i//2, column=i%2, sticky=tk.W, padx=10, pady=2
            )

    def _create_backend_selection(self, parent):
        """創建後端選擇區域"""
        backend_frame = ttk.LabelFrame(parent, text="計算後端選擇", padding="10")
        backend_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # 後端選項
        self.backend_var = tk.StringVar(value="auto")
        
        # 自動選擇
        ttk.Radiobutton(
            backend_frame,
            text="🔄 自動選擇 (推薦)",
            variable=self.backend_var,
            value="auto"
        ).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Label(
            backend_frame,
            text="    自動檢測並選擇最佳計算後端",
            style='Info.TLabel'
        ).grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # GPU選項
        self.gpu_radio = ttk.Radiobutton(
            backend_frame,
            text="🚀 GPU加速模式",
            variable=self.backend_var,
            value="gpu"
        )
        self.gpu_radio.grid(row=1, column=0, sticky=tk.W, pady=2)
        
        self.gpu_info_label = ttk.Label(
            backend_frame,
            text="    檢測中...",
            style='Info.TLabel'
        )
        self.gpu_info_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))
        
        # CPU選項
        ttk.Radiobutton(
            backend_frame,
            text="🖥️ CPU運算模式",
            variable=self.backend_var,
            value="cpu"
        ).grid(row=2, column=0, sticky=tk.W, pady=2)
        
        ttk.Label(
            backend_frame,
            text="    使用CPU進行計算 (相容性最佳)",
            style='Info.TLabel'
        ).grid(row=2, column=1, sticky=tk.W, padx=(10, 0))

    def _create_gpu_info_section(self, parent):
        """創建GPU資訊區域"""
        info_frame = ttk.LabelFrame(parent, text="系統資訊", padding="10")
        info_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # GPU資訊文本框
        self.gpu_info_text = tk.Text(
            info_frame, 
            height=8, 
            width=65, 
            font=('Courier', 9),
            bg='#f0f0f0',
            state='disabled'
        )
        self.gpu_info_text.grid(row=0, column=0, columnspan=2)
        
        # 滾動條
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.gpu_info_text.yview)
        scrollbar.grid(row=0, column=2, sticky=(tk.N, tk.S))
        self.gpu_info_text.configure(yscrollcommand=scrollbar.set)

    def _create_advanced_settings(self, parent):
        """創建進階設定區域"""
        advanced_frame = ttk.LabelFrame(parent, text="進階設定", padding="10")
        advanced_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # GPU設備選擇
        ttk.Label(advanced_frame, text="GPU設備:").grid(row=0, column=0, sticky=tk.W)
        
        self.device_var = tk.StringVar(value="0")
        self.device_combo = ttk.Combobox(
            advanced_frame, 
            textvariable=self.device_var,
            values=["0"], 
            width=10,
            state="readonly"
        )
        self.device_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 回退模式
        self.fallback_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            advanced_frame,
            text="啟用回退模式 (GPU失敗時自動使用CPU)",
            variable=self.fallback_var
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # 性能設定
        ttk.Label(advanced_frame, text="性能模式:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        
        self.performance_var = tk.StringVar(value="balanced")
        perf_frame = tk.Frame(advanced_frame)
        perf_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        ttk.Radiobutton(perf_frame, text="節能", variable=self.performance_var, 
                       value="power_save").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(perf_frame, text="平衡", variable=self.performance_var, 
                       value="balanced").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(perf_frame, text="性能", variable=self.performance_var, 
                       value="performance").pack(side=tk.LEFT)

    def _create_buttons(self, parent):
        """創建按鈕區域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=5, column=0, columnspan=2, pady=(20, 0))
        
        # 啟動按鈕
        start_button = ttk.Button(
            button_frame,
            text="🚀 啟動模擬器",
            command=self._on_start_clicked,
            style='Title.TLabel'
        )
        start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 測試按鈕
        test_button = ttk.Button(
            button_frame,
            text="⚡ 性能測試",
            command=self._on_test_clicked
        )
        test_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 幫助按鈕
        help_button = ttk.Button(
            button_frame,
            text="❓ 使用說明",
            command=self._on_help_clicked
        )
        help_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 退出按鈕
        exit_button = ttk.Button(
            button_frame,
            text="❌ 退出",
            command=self._on_exit_clicked
        )
        exit_button.pack(side=tk.LEFT)

    def _detect_available_backends(self):
        """檢測可用的計算後端"""
        info_lines = ["系統計算能力檢測結果:\n"]
        
        # 檢測CPU
        try:
            import psutil
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            
            info_lines.append(f"[OK] CPU: {cpu_count} 核心")
            if cpu_freq:
                info_lines.append(f"   頻率: {cpu_freq.current:.0f} MHz")
            info_lines.append(f"   記憶體: {memory.total / (1024**3):.1f} GB")
            info_lines.append("")
            
        except Exception as e:
            info_lines.append(f"[WARN] CPU資訊獲取失敗: {e}")
        
        # 檢測GPU
        gpu_available = False
        try:
            import cupy as cp
            
            # 測試GPU基本功能
            test_array = cp.array([1, 2, 3])
            _ = cp.sum(test_array)
            cp.cuda.Device().synchronize()
            
            # 獲取GPU資訊
            device_count = cp.cuda.runtime.getDeviceCount()
            info_lines.append(f"[OK] GPU (CUDA): {device_count} 設備可用")
            
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode()
                memory = props['totalGlobalMem'] / (1024**3)
                compute_capability = f"{props['major']}.{props['minor']}"
                
                info_lines.append(f"   設備 {i}: {name}")
                info_lines.append(f"   記憶體: {memory:.1f} GB")
                info_lines.append(f"   計算能力: {compute_capability}")
                info_lines.append(f"   多處理器數量: {props['multiProcessorCount']}")
                
                if i < device_count - 1:
                    info_lines.append("")
            
            # 更新設備選擇下拉選單
            device_values = [str(i) for i in range(device_count)]
            self.device_combo['values'] = device_values
            
            gpu_available = True
            self.gpu_info_label.configure(text="    GPU可用，支援CUDA加速")
            
        except ImportError:
            info_lines.append("[ERROR] GPU (CUDA): CuPy未安裝")
            info_lines.append("   安裝指令: pip install cupy-cuda11x 或 cupy-cuda12x")
            self.gpu_info_label.configure(text="    需要安裝CuPy以啟用GPU加速")
            self.gpu_radio.configure(state='disabled')
            
        except Exception as e:
            info_lines.append(f"[ERROR] GPU檢測失敗: {e}")
            info_lines.append("   請檢查CUDA驅動程式和工具包安裝")
            self.gpu_info_label.configure(text="    GPU不可用或CUDA未正確安裝")
            self.gpu_radio.configure(state='disabled')
        
        # 添加模擬器功能說明
        info_lines.append("\n" + "="*50)
        info_lines.append("模擬器功能:")
        info_lines.append("• 支援QGC waypoint檔案和CSV軌跡檔案")
        info_lines.append("• 即時3D軌跡可視化與碰撞檢測")
        info_lines.append("• GPU加速大規模無人機群模擬")
        info_lines.append("• 自動任務修改與碰撞避讓")
        info_lines.append("• 支援最多同時模擬1000架無人機")
        
        # 更新資訊顯示
        self.gpu_info_text.configure(state='normal')
        self.gpu_info_text.delete(1.0, tk.END)
        self.gpu_info_text.insert(tk.END, '\n'.join(info_lines))
        self.gpu_info_text.configure(state='disabled')

    def _on_start_clicked(self):
        """啟動按鈕點擊處理"""
        backend_choice = self.backend_var.get()
        device_id = int(self.device_var.get())
        enable_fallback = self.fallback_var.get()
        performance_mode = self.performance_var.get()
        
        # 轉換後端選擇
        if backend_choice == "auto":
            backend = ComputeBackend.AUTO
        elif backend_choice == "gpu":
            backend = ComputeBackend.GPU
        else:
            backend = ComputeBackend.CPU
        
        # 設置配置
        settings.gpu.backend = backend
        settings.gpu.device_id = device_id
        settings.gpu.enable_fallback = enable_fallback
        
        self.result = {
            'action': 'start',
            'backend': backend,
            'device_id': device_id,
            'enable_fallback': enable_fallback,
            'performance_mode': performance_mode
        }
        
        self.root.destroy()

    def _on_test_clicked(self):
        """性能測試按鈕點擊處理"""
        self.result = {'action': 'test'}
        self.root.destroy()

    def _on_help_clicked(self):
        """幫助按鈕點擊處理"""
        help_text = """無人機群模擬器 - 使用說明

🚀 核心功能:
• 3D即時軌跡模擬與可視化
• GPU加速碰撞檢測與避讓
• QGC/CSV任務檔案導入支援
• 智能任務修改與導出

🎯 快速開始:
1. 選擇計算後端 (推薦自動選擇)
2. 點擊「啟動模擬器」
3. 載入任務檔案或創建測試任務
4. 使用播放控制觀看模擬

⚡ 性能建議:
• GPU模式: 適合大規模模擬 (50+ 無人機)
• CPU模式: 適合小規模模擬 (< 50 無人機)
• 自動模式: 系統自動選擇最佳後端

📁 支援格式:
• QGC Waypoint (.waypoints)
• CSV軌跡檔案 (.csv)
• 支援GPS座標和本地座標系統

🛡️ 安全功能:
• 即時碰撞檢測
• 自動避讓路徑生成
• 安全距離可調整
• 修改後任務檔案導出

需要更多幫助請查閱使用者手冊。"""
        
        messagebox.showinfo("使用說明", help_text)

    def _on_exit_clicked(self):
        """退出按鈕點擊處理"""
        self.result = {'action': 'exit'}
        self.root.destroy()

def safe_get_backend_name(backend_obj):
    """安全地獲取後端名稱"""
    try:
        if isinstance(backend_obj, str):
            return backend_obj.upper()
        
        if hasattr(backend_obj, 'value'):
            backend_value = backend_obj.value
            if hasattr(backend_value, 'value'):
                return backend_value.value.upper()
            else:
                return str(backend_value).upper()
        
        if hasattr(backend_obj, 'name'):
            return backend_obj.name.upper()
        
        return str(backend_obj).upper()
        
    except Exception as e:
        print(f"[WARN] 獲取後端名稱失敗: {e}")
        return "UNKNOWN"

def run_performance_test():
    """運行性能測試"""
    print("[TEST] 啟動性能測試...")
    
    # 導入測試工具
    try:
        from utils.gpu_utils import compute_manager, performance_monitor, MathOps
        import numpy as np
        import time
        
        # 修復：安全地獲取後端名稱
        backend_name = safe_get_backend_name(compute_manager.backend)
        print(f"計算後端: {backend_name}")
        print("=" * 50)
        
        # 測試1: 基本陣列運算
        print("測試1: 基本陣列運算")
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            # 創建測試資料
            a = np.random.random((size, 3)).astype(np.float32)
            b = np.random.random((size, 3)).astype(np.float32)
            
            # 測試運算時間
            start_time = time.perf_counter()
            
            # 轉換為當前後端格式
            from utils.gpu_utils import asarray, to_cpu, synchronize
            a_backend = asarray(a)
            b_backend = asarray(b)
            
            # 執行運算
            result = a_backend + b_backend
            result = result * 2.0
            result_sum = compute_manager.xp.sum(result)
            
            # 同步操作
            synchronize()
            
            elapsed = time.perf_counter() - start_time
            print(f"  大小 {size}: {elapsed*1000:.2f} ms")
        
        # 測試2: 距離計算 (模擬碰撞檢測)
        print("\n測試2: 距離矩陣計算 (碰撞檢測模擬)")
        n_points = [50, 100, 200]
        
        for n in n_points:
            positions = np.random.random((n, 3)).astype(np.float32) * 100
            
            start_time = time.perf_counter()
            if MathOps:
                distances = MathOps.distance_matrix(positions, positions)
            else:
                # CPU回退版本
                distances = np.zeros((n, n))
                for i in range(n):
                    for j in range(i+1, n):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        distances[i, j] = distances[j, i] = dist
            
            synchronize()
            elapsed = time.perf_counter() - start_time
            
            print(f"  {n}架無人機碰撞檢測: {elapsed*1000:.2f} ms")
        
        # 測試3: 記憶體使用
        print("\n測試3: 記憶體使用情況")
        try:
            memory_info = compute_manager.get_memory_info()
            print(f"  後端: {memory_info['backend']}")
            print(f"  使用記憶體: {memory_info['used_bytes']/1024**2:.1f} MB")
            print(f"  總記憶體: {memory_info['total_bytes']/1024**2:.1f} MB")
        except Exception as mem_e:
            print(f"  記憶體資訊獲取失敗: {mem_e}")
        
        # 測試4: 模擬器核心功能
        print("\n測試4: 模擬器核心功能")
        try:
            # 測試座標轉換
            from core.drone_physics import EarthCoordinateSystem
            coord_system = EarthCoordinateSystem()
            coord_system.set_origin(24.0, 121.0)
            
            # 批次座標轉換測試
            gps_coords = np.random.uniform([24.0, 121.0, 0], [24.01, 121.01, 100], (1000, 3))
            
            start_time = time.perf_counter()
            local_coords = coord_system.batch_convert_to_meters(gps_coords)
            elapsed = time.perf_counter() - start_time
            
            print(f"  1000個GPS座標轉換: {elapsed*1000:.2f} ms")
            
            # 測試碰撞系統
            from core.collision_system import create_collision_system
            collision_system = create_collision_system()
            
            positions = {}
            for i in range(10):
                positions[f"Drone_{i+1}"] = np.random.uniform(0, 100, 3)
            
            start_time = time.perf_counter()
            warnings = collision_system.detector.check_immediate_collisions(positions)
            elapsed = time.perf_counter() - start_time
            
            print(f"  10架無人機碰撞檢測: {elapsed*1000:.2f} ms")
            print(f"  檢測到 {len(warnings)} 個碰撞警告")
            
        except Exception as e:
            print(f"  模擬器功能測試失敗: {e}")
        
        print("\n[OK] 性能測試完成")
        
    except Exception as e:
        print(f"[ERROR] 性能測試失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='無人機群模擬器 - 專業版')
    parser.add_argument('--backend', choices=['cpu', 'gpu', 'auto'], 
                       default='auto', help='計算後端選擇')
    parser.add_argument('--device', type=int, default=0, 
                       help='GPU設備ID')
    parser.add_argument('--no-gui-select', action='store_true',
                       help='跳過GUI後端選擇對話框')
    parser.add_argument('--test', action='store_true',
                       help='運行性能測試後退出')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日誌級別')
    parser.add_argument('--max-drones', type=int, default=100,
                       help='最大同時模擬無人機數量')
    
    args = parser.parse_args()
    
    # 設置日誌級別
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("🚁 無人機群模擬器 - 專業版")
    print("=" * 50)
    print("功能特色:")
    print("• 3D即時軌跡模擬與可視化")
    print("• GPU加速碰撞檢測與避讓")
    print("• QGC/CSV任務檔案導入")
    print("• 智能任務修改與導出")
    print("• 支援大規模無人機群模擬")
    print("=" * 50)
    
    # 直接運行測試
    if args.test:
        run_performance_test()
        return
    
    # 後端配置
    if args.no_gui_select:
        # 命令列模式
        backend_map = {
            'auto': ComputeBackend.AUTO,
            'gpu': ComputeBackend.GPU,
            'cpu': ComputeBackend.CPU
        }
        set_compute_backend(backend_map[args.backend], args.device)
        action = 'start'
        result = {'performance_mode': 'balanced'}
    else:
        # GUI選擇模式
        try:
            selector = BackendSelector()
            result = selector.show_selection_dialog()
            
            if not result or result.get('action') == 'exit':
                print("[EXIT] 使用者取消，程式退出")
                return
            
            action = result.get('action')
            
            if action == 'test':
                run_performance_test()
                return
            elif action == 'start':
                set_compute_backend(result['backend'], result['device_id'])
        
        except Exception as e:
            logger.error(f"GUI選擇器錯誤: {e}")
            print("回退到命令列模式...")
            set_compute_backend(ComputeBackend.AUTO, 0)
            action = 'start'
            result = {'performance_mode': 'balanced'}
    
    if action == 'start':
        # 顯示後端資訊
        try:
            backend_info = get_compute_backend_info()
            backend_name = safe_get_backend_name(backend_info['backend'])
            print(f"[OK] 計算後端: {backend_name}")
            if backend_info['device_id'] is not None:
                print(f"[GPU] GPU設備ID: {backend_info['device_id']}")
            
            performance_mode = result.get('performance_mode', 'balanced')
            print(f"[設定] 性能模式: {performance_mode}")
            print(f"[設定] 最大無人機數: {args.max_drones}")
            
        except Exception as e:
            print(f"[WARN] 無法獲取後端資訊: {e}")
            backend_info = {'backend': 'CPU', 'device_id': None}
        
        # 啟動主程序
        try:
            print("[START] 啟動主程序...")
            
            # 導入並啟動完整模擬器GUI
            from gui.main_window import DroneSimulatorApp
            
            # 創建主應用程式
            root = tk.Tk()
            app = DroneSimulatorApp(root, backend_info)
            
            # 設置性能參數
            if hasattr(app, 'max_drones'):
                app.max_drones = args.max_drones
            
            # 運行主循環
            print("[INFO] GUI已啟動，請使用圖形介面操作")
            root.mainloop()
            
        except ImportError as e:
            logger.error(f"導入主程序失敗: {e}")
            print("[ERROR] 請確保所有依賴項目都已安裝")
            print("基本依賴: pip install matplotlib pandas numpy")
            print("GPU支援: pip install cupy-cuda11x 或 cupy-cuda12x")
            print("完整安裝: pip install -r requirements.txt")
            
        except Exception as e:
            logger.error(f"主程序運行錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()