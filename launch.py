"""
無人機群飛模擬器快速啟動腳本
提供圖形化設定介面和系統診斷功能
"""

import sys
import os
import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import threading
import logging
from typing import Dict, Any, Optional
import platform

# 確保模組路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 基本日誌設定
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemChecker:
    """系統檢查器"""
    
    def __init__(self):
        self.python_version = platform.python_version()
        self.system_info = self._get_system_info()
    
    def _get_system_info(self) -> Dict[str, Any]:
        """獲取系統信息"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'python_version': self.python_version,
            'python_executable': sys.executable
        }
    
    def check_python_version(self) -> tuple:
        """檢查Python版本"""
        version_info = sys.version_info
        if version_info >= (3, 8):
            return True, f"Python {self.python_version} ✓"
        else:
            return False, f"Python {self.python_version} ✗ (需要 3.8+)"
    
    def check_dependencies(self) -> Dict[str, tuple]:
        """檢查依賴套件"""
        dependencies = {
            'numpy': 'numpy',
            'matplotlib': 'matplotlib', 
            'pandas': 'pandas',
            'tkinter': 'tkinter'
        }
        
        results = {}
        
        for name, module_name in dependencies.items():
            try:
                __import__(module_name)
                results[name] = (True, f"{name} ✓")
            except ImportError:
                results[name] = (False, f"{name} ✗")
        
        return results
    
    def check_gpu_support(self) -> tuple:
        """檢查GPU支援"""
        try:
            import cupy as cp
            
            # 嘗試獲取GPU信息
            try:
                device_count = cp.cuda.runtime.getDeviceCount()
                if device_count > 0:
                    device = cp.cuda.Device(0)
                    with device:
                        name = device.attributes['Name'].decode('utf-8')
                        memory = cp.cuda.runtime.memGetInfo()[1] // (1024**2)  # MB
                    return True, f"GPU: {name} ({memory}MB) ✓"
                else:
                    return False, "GPU: 無可用設備 ✗"
                    
            except Exception as e:
                return False, f"GPU: 檢測失敗 - {str(e)} ✗"
                
        except ImportError:
            return False, "CuPy 未安裝 ✗"
    
    def get_installation_commands(self) -> Dict[str, str]:
        """獲取安裝命令"""
        return {
            'basic': 'pip install numpy matplotlib pandas',
            'gpu_cuda11': 'pip install cupy-cuda11x',
            'gpu_cuda12': 'pip install cupy-cuda12x',
            'all_dependencies': 'pip install -r requirements.txt'
        }


class LauncherGUI:
    """啟動器圖形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.checker = SystemChecker()
        
        self._setup_window()
        self._create_widgets()
        self._check_system()
        
    def _setup_window(self):
        """設定主視窗"""
        self.root.title("無人機群飛模擬器 - 啟動器")
        self.root.geometry("700x800")
        self.root.configure(bg='#2d2d2d')
        self.root.resizable(False, False)
        
        # 置中顯示
        self.root.eval('tk::PlaceWindow . center')
        
    def _create_widgets(self):
        """創建界面元件"""
        # 標題區域
        self._create_header()
        
        # 系統狀態檢查區域
        self._create_system_check_section()
        
        # 選項區域
        self._create_options_section()
        
        # 按鈕區域
        self._create_buttons_section()
        
        # 日誌輸出區域
        self._create_log_section()
        
    def _create_header(self):
        """創建標題"""
        header_frame = tk.Frame(self.root, bg='#2d2d2d')
        header_frame.pack(fill=tk.X, padx=20, pady=20)
        
        title_label = tk.Label(header_frame, 
                              text="🚁 無人機群飛模擬器", 
                              font=('Arial', 18, 'bold'),
                              fg='#00d4aa', bg='#2d2d2d')
        title_label.pack()
        
        subtitle_label = tk.Label(header_frame,
                                 text="GPU加速版本 v2.0 - 快速啟動器",
                                 font=('Arial', 12),
                                 fg='#ffffff', bg='#2d2d2d')
        subtitle_label.pack(pady=(5, 0))
        
        description_label = tk.Label(header_frame,
                                    text="專業級無人機群飛模擬系統，支援GPU/CPU計算後端靈活切換",
                                    font=('Arial', 10),
                                    fg='#888888', bg='#2d2d2d',
                                    wraplength=600)
        description_label.pack(pady=(10, 0))
    
    def _create_system_check_section(self):
        """創建系統檢查區域"""
        check_frame = tk.LabelFrame(self.root, text="🔍 系統檢查", 
                                   font=('Arial', 12, 'bold'),
                                   fg='white', bg='#2d2d2d')
        check_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 檢查結果顯示區域
        self.check_text = tk.Text(check_frame, height=10, bg='#1a1a1a', 
                                 fg='#ffffff', font=('Consolas', 10))
        self.check_text.pack(fill=tk.X, padx=10, pady=10)
        
        # 重新檢查按鈕
        refresh_btn = tk.Button(check_frame, text="🔄 重新檢查",
                               command=self._check_system,
                               bg='#007bff', fg='white', 
                               font=('Arial', 10))
        refresh_btn.pack(pady=5)
    
    def _create_options_section(self):
        """創建選項區域"""
        options_frame = tk.LabelFrame(self.root, text="⚙️ 啟動選項",
                                     font=('Arial', 12, 'bold'), 
                                     fg='white', bg='#2d2d2d')
        options_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # 後端選擇
        backend_frame = tk.Frame(options_frame, bg='#2d2d2d')
        backend_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(backend_frame, text="計算後端:", 
                fg='white', bg='#2d2d2d', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.backend_var = tk.StringVar(value="auto")
        backend_combo = ttk.Combobox(backend_frame, textvariable=self.backend_var,
                                    values=["auto", "gpu", "cpu"],
                                    state="readonly", width=15)
        backend_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # GPU設備選擇
        device_frame = tk.Frame(options_frame, bg='#2d2d2d')
        device_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(device_frame, text="GPU設備:", 
                fg='white', bg='#2d2d2d', font=('Arial', 10)).pack(side=tk.LEFT)
        
        self.device_var = tk.StringVar(value="0")
        device_spin = tk.Spinbox(device_frame, textvariable=self.device_var,
                               from_=0, to=7, width=10, bg='#404040', fg='white')
        device_spin.pack(side=tk.LEFT, padx=(10, 0))
        
        # 其他選項
        options_check_frame = tk.Frame(options_frame, bg='#2d2d2d')
        options_check_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.debug_var = tk.BooleanVar()
        debug_check = tk.Checkbutton(options_check_frame, text="調試模式",
                                    variable=self.debug_var,
                                    fg='white', bg='#2d2d2d', 
                                    selectcolor='#404040',
                                    font=('Arial', 10))
        debug_check.pack(side=tk.LEFT, padx=(0, 20))
        
        self.test_var = tk.BooleanVar()
        test_check = tk.Checkbutton(options_check_frame, text="性能測試",
                                   variable=self.test_var,
                                   fg='white', bg='#2d2d2d',
                                   selectcolor='#404040', 
                                   font=('Arial', 10))
        test_check.pack(side=tk.LEFT)
    
    def _create_buttons_section(self):
        """創建按鈕區域"""
        buttons_frame = tk.Frame(self.root, bg='#2d2d2d')
        buttons_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # 主要操作按鈕
        main_buttons = tk.Frame(buttons_frame, bg='#2d2d2d')
        main_buttons.pack(fill=tk.X, pady=(0, 10))
        
        # 啟動模擬器
        self.launch_btn = tk.Button(main_buttons, text="🚀 啟動模擬器",
                                   command=self._launch_simulator,
                                   bg='#28a745', fg='white',
                                   font=('Arial', 12, 'bold'),
                                   height=2, width=20)
        self.launch_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 安裝依賴
        install_btn = tk.Button(main_buttons, text="📦 安裝依賴",
                               command=self._show_install_dialog,
                               bg='#17a2b8', fg='white',
                               font=('Arial', 12, 'bold'),
                               height=2, width=15)
        install_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 系統診斷
        diagnosis_btn = tk.Button(main_buttons, text="🔧 系統診斷",
                                 command=self._run_diagnosis,
                                 bg='#6f42c1', fg='white',
                                 font=('Arial', 12, 'bold'),
                                 height=2, width=15)
        diagnosis_btn.pack(side=tk.LEFT)
        
        # 其他功能按鈕
        other_buttons = tk.Frame(buttons_frame, bg='#2d2d2d')
        other_buttons.pack(fill=tk.X)
        
        # 性能測試
        test_btn = tk.Button(other_buttons, text="🧪 性能測試",
                           command=self._run_performance_test,
                           bg='#fd7e14', fg='white',
                           font=('Arial', 10), width=12)
        test_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 查看日誌
        log_btn = tk.Button(other_buttons, text="📋 查看日誌",
                          command=self._view_logs,
                          bg='#6c757d', fg='white',
                          font=('Arial', 10), width=12)
        log_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 設定檔編輯
        config_btn = tk.Button(other_buttons, text="⚙️ 編輯設定",
                              command=self._edit_config,
                              bg='#20c997', fg='white',
                              font=('Arial', 10), width=12)
        config_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # 說明文件
        help_btn = tk.Button(other_buttons, text="❓ 說明文件",
                           command=self._show_help,
                           bg='#e83e8c', fg='white',
                           font=('Arial', 10), width=12)
        help_btn.pack(side=tk.LEFT)
    
    def _create_log_section(self):
        """創建日誌區域"""
        log_frame = tk.LabelFrame(self.root, text="📝 日誌輸出",
                                 font=('Arial', 10, 'bold'),
                                 fg='white', bg='#2d2d2d')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # 日誌文字區域
        log_container = tk.Frame(log_frame, bg='#2d2d2d')
        log_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_container, height=6, bg='#1a1a1a',
                               fg='#00d4aa', font=('Consolas', 9))
        log_scrollbar = ttk.Scrollbar(log_container, orient="vertical",
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 清除日誌按鈕
        clear_btn = tk.Button(log_frame, text="清除日誌",
                             command=self._clear_log,
                             bg='#dc3545', fg='white',
                             font=('Arial', 8))
        clear_btn.pack(pady=5)
    
    def _check_system(self):
        """檢查系統狀態"""
        self.check_text.delete(1.0, tk.END)
        self._add_check_result("🔍 開始系統檢查...\n")
        
        # 檢查Python版本
        py_ok, py_msg = self.checker.check_python_version()
        self._add_check_result(f"Python版本: {py_msg}")
        
        # 檢查依賴套件
        deps = self.checker.check_dependencies()
        self._add_check_result("\n📦 依賴套件檢查:")
        for name, (ok, msg) in deps.items():
            self._add_check_result(f"  {msg}")
        
        # 檢查GPU支援
        gpu_ok, gpu_msg = self.checker.check_gpu_support()
        self._add_check_result(f"\n🎮 GPU支援: {gpu_msg}")
        
        # 檢查系統信息
        self._add_check_result(f"\n💻 系統信息:")
        info = self.checker.system_info
        self._add_check_result(f"  平台: {info['platform']} {info['architecture']}")
        self._add_check_result(f"  Python: {info['python_version']}")
        
        # 總結
        all_deps_ok = all(ok for ok, _ in deps.values())
        if py_ok and all_deps_ok:
            if gpu_ok:
                self._add_check_result("\n✅ 系統檢查通過，GPU加速可用")
                self.launch_btn.configure(state='normal')
            else:
                self._add_check_result("\n⚠️ 系統檢查通過，但GPU不可用（將使用CPU模式）")
                self.launch_btn.configure(state='normal')
        else:
            self._add_check_result("\n❌ 系統檢查失敗，需要安裝缺少的依賴")
            self.launch_btn.configure(state='disabled')
    
    def _add_check_result(self, text: str):
        """添加檢查結果"""
        self.check_text.insert(tk.END, text + "\n")
        self.check_text.see(tk.END)
        self.root.update_idletasks()
    
    def _add_log(self, message: str, color: str = "#00d4aa"):
        """添加日誌"""
        self.log_text.insert(tk.END, f"[{self._get_time()}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def _get_time(self):
        """獲取當前時間字符串"""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def _clear_log(self):
        """清除日誌"""
        self.log_text.delete(1.0, tk.END)
    
    def _launch_simulator(self):
        """啟動模擬器"""
        self._add_log("準備啟動模擬器...")
        
        # 構建啟動命令
        cmd = [sys.executable, "main.py"]
        
        if self.backend_var.get() != "auto":
            cmd.extend(["--backend", self.backend_var.get()])
        
        if self.device_var.get() != "0":
            cmd.extend(["--device", self.device_var.get()])
        
        if self.debug_var.get():
            cmd.append("--debug")
        
        if self.test_var.get():
            cmd.append("--test")
        
        self._add_log(f"啟動命令: {' '.join(cmd)}")
        
        # 在新線程中執行
        def run_simulator():
            try:
                process = subprocess.Popen(cmd, 
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE,
                                         text=True, encoding='utf-8')
                
                self._add_log("模擬器已啟動")
                
                # 監控輸出
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        self.root.after(0, lambda: self._add_log(output.strip()))
                
                return_code = process.poll()
                if return_code == 0:
                    self.root.after(0, lambda: self._add_log("模擬器正常結束"))
                else:
                    error = process.stderr.read()
                    self.root.after(0, lambda: self._add_log(f"模擬器異常結束: {error}"))
                
            except Exception as e:
                self.root.after(0, lambda: self._add_log(f"啟動失敗: {str(e)}"))
        
        thread = threading.Thread(target=run_simulator, daemon=True)
        thread.start()
    
    def _show_install_dialog(self):
        """顯示安裝對話框"""
        install_window = tk.Toplevel(self.root)
        install_window.title("依賴套件安裝")
        install_window.geometry("500x400")
        install_window.configure(bg='#2d2d2d')
        
        tk.Label(install_window, text="選擇要安裝的套件:",
                font=('Arial', 12, 'bold'), fg='white', bg='#2d2d2d').pack(pady=10)
        
        commands = self.checker.get_installation_commands()
        
        for name, cmd in commands.items():
            frame = tk.Frame(install_window, bg='#2d2d2d')
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            btn = tk.Button(frame, text=f"安裝 {name}",
                           command=lambda c=cmd: self._run_install_command(c),
                           bg='#007bff', fg='white', width=15)
            btn.pack(side=tk.LEFT)
            
            tk.Label(frame, text=cmd, fg='#888888', bg='#2d2d2d',
                    font=('Consolas', 9)).pack(side=tk.LEFT, padx=(10, 0))
    
    def _run_install_command(self, command: str):
        """執行安裝命令"""
        self._add_log(f"執行安裝命令: {command}")
        
        def install():
            try:
                process = subprocess.run(command, shell=True, 
                                       capture_output=True, text=True)
                if process.returncode == 0:
                    self.root.after(0, lambda: self._add_log("安裝完成"))
                    self.root.after(0, lambda: self._check_system())
                else:
                    self.root.after(0, lambda: self._add_log(f"安裝失敗: {process.stderr}"))
            except Exception as e:
                self.root.after(0, lambda: self._add_log(f"安裝錯誤: {str(e)}"))
        
        thread = threading.Thread(target=install, daemon=True)
        thread.start()
    
    def _run_diagnosis(self):
        """運行系統診斷"""
        self._add_log("開始系統診斷...")
        
        diag_info = []
        diag_info.append("=== 系統診斷報告 ===")
        diag_info.append(f"時間: {self._get_time()}")
        diag_info.extend([f"{k}: {v}" for k, v in self.checker.system_info.items()])
        
        # GPU詳細信息
        try:
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            diag_info.append(f"GPU設備數量: {device_count}")
            
            for i in range(device_count):
                device = cp.cuda.Device(i)
                with device:
                    props = device.attributes
                    diag_info.append(f"GPU {i}: {props.get('Name', 'Unknown').decode('utf-8')}")
                    
        except Exception as e:
            diag_info.append(f"GPU診斷失敗: {str(e)}")
        
        for info in diag_info:
            self._add_log(info)
    
    def _run_performance_test(self):
        """運行性能測試"""
        self._add_log("啟動性能測試...")
        
        def test():
            try:
                cmd = [sys.executable, "main.py", "--test"]
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode == 0:
                    self.root.after(0, lambda: self._add_log("性能測試完成"))
                    self.root.after(0, lambda: self._add_log(process.stdout))
                else:
                    self.root.after(0, lambda: self._add_log(f"測試失敗: {process.stderr}"))
                    
            except Exception as e:
                self.root.after(0, lambda: self._add_log(f"測試錯誤: {str(e)}"))
        
        thread = threading.Thread(target=test, daemon=True)
        thread.start()
    
    def _view_logs(self):
        """查看日誌檔案"""
        log_dir = "logs"
        if os.path.exists(log_dir):
            if platform.system() == "Windows":
                os.startfile(log_dir)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", log_dir])
            else:  # Linux
                subprocess.run(["xdg-open", log_dir])
        else:
            self._add_log("日誌目錄不存在")
    
    def _edit_config(self):
        """編輯配置檔案"""
        config_file = "config/settings.yaml"
        if os.path.exists(config_file):
            if platform.system() == "Windows":
                os.startfile(config_file)
            else:
                subprocess.run(["open", config_file] if platform.system() == "Darwin" 
                             else ["xdg-open", config_file])
        else:
            self._add_log("配置檔案不存在，將創建默認配置")
            try:
                os.makedirs("config", exist_ok=True)
                from config.settings import get_config_manager
                manager = get_config_manager()
                manager.create_default_config_file()
                self._add_log("默認配置檔案已創建")
            except Exception as e:
                self._add_log(f"創建配置檔案失敗: {str(e)}")
    
    def _show_help(self):
        """顯示幫助信息"""
        help_text = """無人機群飛模擬器 - 使用說明

🚀 啟動選項:
• 計算後端: 選擇GPU、CPU或自動檢測
• GPU設備: 選擇GPU設備ID (多GPU系統)
• 調試模式: 啟用詳細日誌輸出
• 性能測試: 運行基準測試

📦 依賴安裝:
• 基本依賴: numpy, matplotlib, pandas
• GPU支援: cupy-cuda11x 或 cupy-cuda12x
• 完整安裝: pip install -r requirements.txt

🔧 系統要求:
• Python 3.8+
• Windows 10+, Ubuntu 18.04+, macOS 10.14+
• GPU: NVIDIA (可選，支援CUDA)

💡 使用技巧:
1. 首次使用建議先運行系統檢查
2. 如果GPU不可用，系統會自動回退到CPU模式
3. 性能測試可以幫助評估系統性能
4. 日誌檔案保存在 logs/ 目錄中

❓ 常見問題:
• 如果啟動失敗，請檢查系統要求和依賴
• GPU問題可嘗試更新NVIDIA驅動
• macOS用戶無法使用GPU加速

📧 技術支援:
查看GitHub項目頁面獲取更多幫助
"""
        
        messagebox.showinfo("使用說明", help_text)
    
    def run(self):
        """運行啟動器"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("用戶中斷啟動器")
        except Exception as e:
            logger.error(f"啟動器運行錯誤: {e}")
            messagebox.showerror("錯誤", f"啟動器運行時發生錯誤:\n{str(e)}")


def main():
    """主函數"""
    print("無人機群飛模擬器 - 快速啟動器")
    
    try:
        # 確保必要目錄存在
        os.makedirs('logs', exist_ok=True)
        os.makedirs('config', exist_ok=True)
        os.makedirs('exports', exist_ok=True)
        
        # 創建並運行啟動器
        launcher = LauncherGUI()
        launcher.run()
        
    except Exception as e:
        print(f"啟動器初始化失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()