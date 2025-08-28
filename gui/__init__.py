"""
GUI module for drone simulator
"""

__version__ = "5.2.0"

try:
    from .main_window import *
    from .control_panel import *
    from .plot_manager import *
    
    # 確保這些類可用
    __all__ = [
        'DroneSimulatorApp',
        'MainWindow', 
        'ControlPanel',
        'PlotManager'
    ]
    
except ImportError as e:
    # 基本回退
    import tkinter as tk
    
    class DroneSimulatorApp:
        def __init__(self, root):
            self.root = root
            label = tk.Label(root, text="無人機模擬器基本模式")
            label.pack()
    
    class MainWindow:
        pass
    
    class ControlPanel:
        pass
    
    class PlotManager:
        pass
    
    __all__ = ['DroneSimulatorApp', 'MainWindow', 'ControlPanel', 'PlotManager']