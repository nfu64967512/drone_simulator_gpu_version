#!/usr/bin/env python3
"""
快速創建gui/__init__.py文件
"""
from pathlib import Path

def create_gui_init():
    """創建gui/__init__.py文件"""
    
    gui_init_content = '''"""
GUI module for drone simulator
圖形界面模組：包含主視窗、控制面板和視覺化
"""

__version__ = "5.2.0"

# 嘗試導入GUI組件
try:
    from .main_window import DroneSimulatorApp, MainWindow
    from .control_panel import ControlPanel
    from .plot_manager import PlotManager, GPUPlotManager
    
    # 成功導入所有GUI組件
    GUI_AVAILABLE = True
    
    __all__ = [
        'DroneSimulatorApp', 
        'MainWindow',
        'ControlPanel', 
        'PlotManager', 
        'GPUPlotManager'
    ]
    
except ImportError as e:
    # GUI組件不可用時的佔位實現
    GUI_AVAILABLE = False
    
    import tkinter as tk
    from tkinter import messagebox
    
    class DroneSimulatorApp:
        """基本GUI應用程式佔位類"""
        def __init__(self, root):
            self.root = root
            self.root.title("無人機模擬器 - 基本模式")
            self.root.geometry("400x300")
            
            # 顯示基本資訊
            info_label = tk.Label(
                root, 
                text="無人機模擬器\\n\\n完整GUI組件不可用\\n運行於基本模式",
                font=("Arial", 12),
                justify=tk.CENTER
            )
            info_label.pack(expand=True)
            
            # 基本按鈕
            btn_frame = tk.Frame(root)
            btn_frame.pack(pady=20)
            
            tk.Button(
                btn_frame, 
                text="性能測試", 
                command=self.run_test,
                width=12
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                btn_frame, 
                text="關於", 
                command=self.show_about,
                width=12
            ).pack(side=tk.LEFT, padx=5)
            
            tk.Button(
                btn_frame, 
                text="退出", 
                command=root.quit,
                width=12
            ).pack(side=tk.LEFT, padx=5)
        
        def run_test(self):
            """運行基本測試"""
            try:
                from utils.gpu_utils import is_gpu_enabled
                backend = "GPU" if is_gpu_enabled() else "CPU"
                messagebox.showinfo(
                    "系統資訊", 
                    f"計算後端: {backend}\\n\\n完整性能測試請使用:\\npython main.py --test"
                )
            except Exception as e:
                messagebox.showerror("錯誤", f"測試失敗: {e}")
        
        def show_about(self):
            """顯示關於資訊"""
            messagebox.showinfo(
                "關於", 
                "無人機群模擬器 v5.2.0\\n\\nGPU/CPU混合運算系統\\n基本模式運行中"
            )
    
    class MainWindow:
        """主視窗佔位類"""
        def __init__(self):
            print("MainWindow: GUI組件不可用，使用基本模式")
        
        def show(self):
            pass
        
        def hide(self):
            pass
    
    class ControlPanel:
        """控制面板佔位類"""
        def __init__(self):
            print("ControlPanel: GUI組件不可用")
        
        def update_status(self, status):
            print(f"狀態更新: {status}")
        
        def set_enabled(self, enabled):
            pass
    
    class PlotManager:
        """繪圖管理器佔位類"""
        def __init__(self):
            print("PlotManager: 視覺化組件不可用")
        
        def update_plot(self, data):
            print("繪圖更新請求（視覺化不可用）")
        
        def clear_plot(self):
            pass
        
        def save_plot(self, filename):
            print(f"無法保存繪圖到 {filename}：視覺化組件不可用")
    
    # GPU版本的別名
    GPUPlotManager = PlotManager
    
    __all__ = [
        'DroneSimulatorApp',
        'MainWindow', 
        'ControlPanel',
        'PlotManager',
        'GPUPlotManager'
    ]

def get_gui_info():
    """獲取GUI模組資訊"""
    return {
        'version': __version__,
        'available': GUI_AVAILABLE,
        'components': __all__,
        'description': '無人機模擬器圖形界面模組'
    }

def create_basic_app():
    """創建基本GUI應用程式"""
    try:
        import tkinter as tk
        root = tk.Tk()
        app = DroneSimulatorApp(root)
        return root, app
    except Exception as e:
        print(f"無法創建GUI應用程式: {e}")
        return None, None

# 模組初始化完成提示
if GUI_AVAILABLE:
    print("[GUI] 完整GUI組件已載入")
else:
    print("[GUI] 基本GUI模式已載入")
'''
    
    # 確保gui目錄存在
    gui_dir = Path("gui")
    gui_dir.mkdir(exist_ok=True)
    
    # 創建__init__.py文件
    init_file = gui_dir / "__init__.py"
    
    if init_file.exists():
        print(f"gui/__init__.py 已存在")
    else:
        init_file.write_text(gui_init_content, encoding='utf-8')
        print(f"已創建 gui/__init__.py")
    
    return init_file.exists()

def test_gui_import():
    """測試gui模組導入"""
    try:
        import sys
        # 清除快取
        if 'gui' in sys.modules:
            del sys.modules['gui']
        
        import gui
        print("gui模組導入成功")
        
        # 測試基本GUI創建
        try:
            from gui import DroneSimulatorApp
            print("DroneSimulatorApp 可用")
            return True
        except ImportError as e:
            print(f"DroneSimulatorApp 導入失敗: {e}")
            return False
            
    except Exception as e:
        print(f"gui模組測試失敗: {e}")
        return False

def main():
    """主函數"""
    print("創建gui/__init__.py文件")
    print("=" * 30)
    
    # 創建文件
    success = create_gui_init()
    
    if success:
        print("測試gui模組...")
        test_success = test_gui_import()
        
        if test_success:
            print("\n[OK] gui模組創建成功！")
            print("現在可以運行:")
            print("  python -c \"import gui; print('GUI模組可用')\"")
            print("  python main.py")
        else:
            print("\n[WARN] gui模組創建了，但測試未完全通過")
    else:
        print("[ERROR] gui/__init__.py創建失敗")

if __name__ == "__main__":
    main()