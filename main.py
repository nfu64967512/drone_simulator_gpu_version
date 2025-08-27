#!/usr/bin/env python3
"""
無人機群模擬器主程式
支持GPU/CPU後端選擇和命令列參數
"""
import sys
import os
import argparse
import tkinter as tk
from tkinter import messagebox, ttk
import logging
import json
from pathlib import Path

# 確保項目根目錄在Python路徑中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 導入配置和工具
from config.settings import settings, ComputeBackend, set_compute_backend, get_compute_backend_info
from utils.logging_config import setup_logging

# 設置日誌
logger = setup_logging()

class BackendSelector:
    """後端選擇對話框"""
    
    def __init__(self):
        self.selected_backend = None
        self.selected_device = 0
        self.root = None
        self.result = None

    def show_selection_dialog(self):
        """顯示後端選擇對話框"""
        self.root = tk.Tk()
        self.root.title("無人機模擬器 - 計算後端選擇")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # 設置圖標和樣式
        self._setup_ui_style()
        
        # 創建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 標題
        title_label = ttk.Label(
            main_frame, 
            text="🚁 無人機群模擬器",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
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

    def _create_backend_selection(self, parent):
        """創建後端選擇區域"""
        # 分組框架
        backend_frame = ttk.LabelFrame(parent, text="計算後端選擇", padding="10")
        backend_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
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
            text="🚀 GPU加速",
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
            text="🖥️ CPU運算",
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
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # GPU資訊文本框
        self.gpu_info_text = tk.Text(
            info_frame, 
            height=60, 
            width=60, 
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
        advanced_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # GPU設備選擇
        ttk.Label(advanced_frame, text="GPU設備:").grid(row=0, column=0, sticky=tk.W)
        
        self.device_var = tk.StringVar(value="0")
        device_combo = ttk.Combobox(
            advanced_frame, 
            textvariable=self.device_var,
            values=["0"], 
            width=10,
            state="readonly"
        )
        device_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))
        
        # 回退模式
        self.fallback_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            advanced_frame,
            text="啟用回退模式 (GPU失敗時自動使用CPU)",
            variable=self.fallback_var
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))

    def _create_buttons(self, parent):
        """創建按鈕區域"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=4, column=0, columnspan=2, pady=(20, 0))
        
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
            text="🧪 性能測試",
            command=self._on_test_clicked
        )
        test_button.pack(side=tk.LEFT, padx=(0, 10))
        
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
            
            info_lines.append(f"✅ CPU: {cpu_count} 核心")
            if cpu_freq:
                info_lines.append(f"   頻率: {cpu_freq.current:.0f} MHz")
            info_lines.append(f"   記憶體: {memory.total / (1024**3):.1f} GB")
            
        except Exception as e:
            info_lines.append(f"⚠️ CPU資訊獲取失敗: {e}")
        
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
            info_lines.append(f"\n✅ GPU (CUDA): {device_count} 設備可用")
            
            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode()
                memory = props['totalGlobalMem'] / (1024**3)
                info_lines.append(f"   設備 {i}: {name}")
                info_lines.append(f"   記憶體: {memory:.1f} GB")
            
            # 更新設備選擇下拉選單
            device_values = [str(i) for i in range(device_count)]
            if hasattr(self, 'device_combo'):
                self.device_combo['values'] = device_values
            
            gpu_available = True
            self.gpu_info_label.configure(text="    GPU可用，支援CUDA加速")
            
        except ImportError:
            info_lines.append("\n❌ GPU (CUDA): CuPy未安裝")
            self.gpu_info_label.configure(text="    需要安裝CuPy以啟用GPU加速")
            self.gpu_radio.configure(state='disabled')
            
        except Exception as e:
            info_lines.append(f"\n❌ GPU檢測失敗: {e}")
            self.gpu_info_label.configure(text="    GPU不可用或CUDA未正確安裝")
            self.gpu_radio.configure(state='disabled')
        
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
            'enable_fallback': enable_fallback
        }
        
        self.root.destroy()

    def _on_test_clicked(self):
        """性能測試按鈕點擊處理"""
        self.result = {'action': 'test'}
        self.root.destroy()

    def _on_exit_clicked(self):
        """退出按鈕點擊處理"""
        self.result = {'action': 'exit'}
        self.root.destroy()

def run_performance_test():
    """運行性能測試"""
    print("🧪 啟動性能測試...")
    
    # 導入測試工具
    try:
        from utils.gpu_utils import compute_manager, performance_monitor, MathOps
        import numpy as np
        import time
        
        print(f"計算後端: {compute_manager.backend.value.upper()}")
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
        
        # 測試2: 距離計算
        print("\n測試2: 距離矩陣計算")
        n_points = [100, 500, 1000]
        
        for n in n_points:
            points = np.random.random((n, 3)).astype(np.float32) * 100
            
            start_time = time.perf_counter()
            distances = MathOps.distance_matrix(points, points)
            synchronize()
            elapsed = time.perf_counter() - start_time
            
            print(f"  {n}x{n} 矩陣: {elapsed*1000:.2f} ms")
        
        # 測試3: 記憶體使用
        print("\n測試3: 記憶體使用情況")
        memory_info = compute_manager.get_memory_info()
        print(f"  後端: {memory_info['backend']}")
        print(f"  使用記憶體: {memory_info['used_bytes']/1024**2:.1f} MB")
        print(f"  總記憶體: {memory_info['total_bytes']/1024**2:.1f} MB")
        
        print("\n✅ 性能測試完成")
        
    except Exception as e:
        print(f"❌ 性能測試失敗: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函數"""
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='無人機群模擬器')
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
    
    args = parser.parse_args()
    
    # 設置日誌級別
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    print("🚁 無人機群模擬器")
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
    else:
        # GUI選擇模式
        try:
            selector = BackendSelector()
            result = selector.show_selection_dialog()
            
            if not result or result.get('action') == 'exit':
                print("👋 使用者取消，程式退出")
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
    
    if action == 'start':
        # 顯示後端資訊
        backend_info = get_compute_backend_info()
        print(f"✅ 計算後端: {backend_info['backend'].upper()}")
        if backend_info['device_id'] is not None:
            print(f"📱 GPU設備ID: {backend_info['device_id']}")
        
        # 啟動主程序
        try:
            print("🚀 啟動主程序...")
            
            # 導入並啟動主GUI
            from gui.main_window import DroneSimulatorApp
            
            # 創建主應用程式
            root = tk.Tk()
            app = DroneSimulatorApp(root)
            
            # 運行主循環
            root.mainloop()
            
        except ImportError as e:
            logger.error(f"導入主程序失敗: {e}")
            print("❌ 請確保所有依賴項目都已安裝")
            print("安裝指令: pip install -r requirements.txt")
            
        except Exception as e:
            logger.error(f"主程序運行錯誤: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()