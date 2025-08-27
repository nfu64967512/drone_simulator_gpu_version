#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Window Management Module
Handles main window setup, menus, and keyboard shortcuts for v5.1
"""

import tkinter as tk
from tkinter import messagebox
import logging
from typing import Optional, Callable
from config.settings import SimulatorConfig, UILabels

logger = logging.getLogger(__name__)

class MainWindow:
    """
    Main window manager for the drone simulator
    Handles window setup, menus, and global shortcuts
    """
    
    def __init__(self):
        self.window: Optional[tk.Tk] = None
        self.callbacks: dict = {}
        
    def create_window(self) -> tk.Tk:
        """
        Create and configure the main window
        
        Returns:
            Configured tkinter window
        """
        logger.info("Creating main window")
        
        self.window = tk.Tk()
        self.window.title(SimulatorConfig.WINDOW_TITLE)
        self.window.geometry(SimulatorConfig.WINDOW_SIZE)
        self.window.configure(bg=SimulatorConfig.UI_COLORS['background'])
        
        # Set window icon if available
        try:
            # You can add an icon file here
            # self.window.iconbitmap('icon.ico')
            pass
        except Exception as e:
            logger.debug(f"Could not set window icon: {e}")
        
        # Configure window behavior
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Maximize window
        self._maximize_window()
        
        # Setup focus
        self.window.focus_set()
        
        logger.info("Main window created successfully")
        return self.window
    
    def _maximize_window(self) -> None:
        """Maximize window using platform-specific methods"""
        try:
            self.window.state('zoomed')  # Windows
            logger.debug("Window maximized (Windows)")
        except tk.TclError:
            try:
                self.window.attributes('-zoomed', True)  # Linux
                logger.debug("Window maximized (Linux)")
            except tk.TclError:
                # macOS or fallback
                self.window.geometry(f"{self.window.winfo_screenwidth()}x{self.window.winfo_screenheight()}+0+0")
                logger.debug("Window maximized (fallback)")
    
    def register_callback(self, event_name: str, callback: Callable) -> None:
        """
        Register callback for window events
        
        Args:
            event_name: Name of the event
            callback: Callback function
        """
        self.callbacks[event_name] = callback
        logger.debug(f"Registered callback for event: {event_name}")
    
    def create_menu(self) -> None:
        """Create application menu bar with English labels"""
        if not self.window:
            logger.error("Window not created yet")
            return
        
        logger.info("Creating menu bar")
        
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        
        # File menu
        self._create_file_menu(menubar)
        
        # View menu
        self._create_view_menu(menubar)
        
        # Simulation menu
        self._create_simulation_menu(menubar)
        
        # Help menu
        self._create_help_menu(menubar)
        
        logger.info("Menu bar created successfully")
    
    def _create_file_menu(self, menubar: tk.Menu) -> None:
        """Create file menu"""
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=UILabels.MENU_FILE, menu=file_menu)
        
        file_menu.add_command(
            label="Load QGC Files",
            command=lambda: self._execute_callback('load_qgc_files'),
            accelerator="Ctrl+O"
        )
        
        file_menu.add_command(
            label="Load CSV Files", 
            command=lambda: self._execute_callback('load_csv_files')
        )
        
        file_menu.add_separator()
        
        file_menu.add_command(
            label="Create Test Mission",
            command=lambda: self._execute_callback('create_test_mission'),
            accelerator="Ctrl+T"
        )
        
        file_menu.add_separator()
        
        file_menu.add_command(
            label="Export Modified Missions",
            command=lambda: self._execute_callback('export_modified_missions'),
            accelerator="Ctrl+S"
        )
        
        file_menu.add_command(
            label="Export Collision Log",
            command=lambda: self._execute_callback('export_collision_log'),
            accelerator="Ctrl+L"
        )
        
        file_menu.add_separator()
        
        file_menu.add_command(
            label="Exit",
            command=self._on_window_close,
            accelerator="Alt+F4"
        )
    
    def _create_view_menu(self, menubar: tk.Menu) -> None:
        """Create view menu"""
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=UILabels.MENU_VIEW, menu=view_menu)
        
        view_menu.add_command(
            label=UILabels.TOP_VIEW,
            command=lambda: self._execute_callback('set_top_view'),
            accelerator="1"
        )
        
        view_menu.add_command(
            label=UILabels.SIDE_VIEW,
            command=lambda: self._execute_callback('set_side_view'),
            accelerator="2"
        )
        
        view_menu.add_command(
            label=UILabels.VIEW_3D,
            command=lambda: self._execute_callback('set_3d_view'),
            accelerator="3"
        )
        
        view_menu.add_separator()
        
        view_menu.add_command(
            label="Reset View",
            command=lambda: self._execute_callback('reset_view'),
            accelerator="R"
        )
        
        view_menu.add_command(
            label="Fit All",
            command=lambda: self._execute_callback('fit_all_view'),
            accelerator="F"
        )
    
    def _create_simulation_menu(self, menubar: tk.Menu) -> None:
        """Create simulation menu"""
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=UILabels.MENU_SIMULATION, menu=sim_menu)
        
        sim_menu.add_command(
            label="Play/Pause",
            command=lambda: self._execute_callback('toggle_play'),
            accelerator="Space"
        )
        
        sim_menu.add_command(
            label="Stop",
            command=lambda: self._execute_callback('stop_simulation'),
            accelerator="S"
        )
        
        sim_menu.add_command(
            label="Reset",
            command=lambda: self._execute_callback('reset_simulation'),
            accelerator="Ctrl+R"
        )
        
        sim_menu.add_separator()
        
        sim_menu.add_command(
            label="Analyze Collisions",
            command=lambda: self._execute_callback('analyze_collisions')
        )
        
        sim_menu.add_command(
            label="Clear Warnings",
            command=lambda: self._execute_callback('clear_warnings')
        )
    
    def _create_help_menu(self, menubar: tk.Menu) -> None:
        """Create help menu"""
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=UILabels.MENU_HELP, menu=help_menu)
        
        help_menu.add_command(
            label="Keyboard Shortcuts",
            command=self._show_shortcuts,
            accelerator="F1"
        )
        
        help_menu.add_command(
            label="User Manual",
            command=self._show_user_manual
        )
        
        help_menu.add_separator()
        
        help_menu.add_command(
            label="System Info",
            command=self._show_system_info
        )
        
        help_menu.add_command(
            label="About",
            command=self._show_about
        )
    
    def bind_shortcuts(self) -> None:
        """Bind keyboard shortcuts"""
        if not self.window:
            logger.error("Window not created yet")
            return
        
        logger.info("Binding keyboard shortcuts")
        
        # Global shortcuts
        self.window.bind('<KeyPress>', self._on_key_press)
        self.window.bind('<Control-o>', lambda e: self._execute_callback('load_qgc_files'))
        self.window.bind('<Control-t>', lambda e: self._execute_callback('create_test_mission'))
        self.window.bind('<Control-s>', lambda e: self._execute_callback('export_modified_missions'))
        self.window.bind('<Control-l>', lambda e: self._execute_callback('export_collision_log'))
        self.window.bind('<Control-r>', lambda e: self._execute_callback('reset_simulation'))
        self.window.bind('<F1>', lambda e: self._show_shortcuts())
        self.window.bind('<Escape>', lambda e: self._on_window_close())
        
        # Function key shortcuts
        for i in range(1, 13):  # F1-F12
            self.window.bind(f'<F{i}>', lambda e, key=i: self._on_function_key(key))
        
        logger.debug("Keyboard shortcuts bound successfully")
    
    def _on_key_press(self, event) -> None:
        """Handle key press events"""
        key = event.keysym.lower()
        
        # Map common keys to callbacks
        key_mappings = {
            'space': 'toggle_play',
            'r': 'reset_simulation',
            's': 'stop_simulation',
            '1': 'set_top_view',
            '2': 'set_side_view',
            '3': 'set_3d_view',
            'f': 'fit_all_view',
            'h': 'show_shortcuts'
        }
        
        if key in key_mappings:
            self._execute_callback(key_mappings[key])
        
        logger.debug(f"Key pressed: {key}")
    
    def _on_function_key(self, key_number: int) -> None:
        """Handle function key presses"""
        function_mappings = {
            1: 'show_shortcuts',
            2: 'show_system_info',
            3: 'analyze_collisions',
            4: 'export_collision_log',
            5: 'create_test_mission',
            11: 'toggle_fullscreen',
            12: 'show_about'
        }
        
        if key_number in function_mappings:
            self._execute_callback(function_mappings[key_number])
    
    def _execute_callback(self, event_name: str) -> None:
        """Execute registered callback"""
        if event_name in self.callbacks:
            try:
                self.callbacks[event_name]()
                logger.debug(f"Executed callback: {event_name}")
            except Exception as e:
                logger.error(f"Error executing callback {event_name}: {e}")
                messagebox.showerror("Error", f"Failed to execute {event_name}: {e}")
        else:
            logger.warning(f"No callback registered for event: {event_name}")
    
    def _on_window_close(self) -> None:
        """Handle window close event"""
        logger.info("Window close requested")
        
        # Ask for confirmation if callback is registered
        if 'on_closing' in self.callbacks:
            self._execute_callback('on_closing')
        else:
            # Default close behavior
            if messagebox.askokcancel("Quit", "Do you want to quit the simulator?"):
                self.window.destroy()
    
    def _show_shortcuts(self) -> None:
        """Show keyboard shortcuts dialog"""
        shortcuts_text = """Keyboard Shortcuts:

🎮 Playback Control:
  Space        Play/Pause simulation
  S            Stop simulation
  Ctrl+R       Reset simulation

🎯 View Control:
  1            Top view
  2            Side view  
  3            3D view
  F            Fit all in view
  R            Reset view

📁 File Operations:
  Ctrl+O       Load QGC files
  Ctrl+T       Create test mission
  Ctrl+S       Export modified missions
  Ctrl+L       Export collision log

🔍 Analysis:
  F3           Analyze collisions
  F4           Export collision log
  F5           Create test mission

🖼️ Display:
  F11          Toggle fullscreen
  Mouse Wheel  Zoom in/out
  Mouse Drag   Rotate 3D view

ℹ️ Help & Info:
  F1           Show this help
  F2           System information
  F12          About dialog
  H            Quick help
  ESC          Exit application
"""
        
        messagebox.showinfo("Keyboard Shortcuts", shortcuts_text)
    
    def _show_user_manual(self) -> None:
        """Show user manual dialog"""
        manual_text = """User Manual - Advanced Drone Swarm Simulator v5.1

🚀 Getting Started:
1. Load mission files (QGC .waypoints or CSV format)
2. Click 'Test' to create a sample 2x2 formation mission
3. Use Play/Pause controls to run simulation
4. Monitor collision warnings in real-time

📁 File Formats:
• QGC Waypoints: Standard QGroundControl .waypoints format
• CSV: Latitude, Longitude, Altitude columns (various naming supported)

🛡️ Safety Features:
• Real-time collision detection (configurable distance)
• Automatic LOITER insertion for conflict avoidance
• Priority-based conflict resolution (lower drone numbers have priority)
• Comprehensive collision logging and export

🎯 2x2 Formation:
• 6-meter spacing between drones (configurable)
• Unified east takeoff area (50m offset from base point)
• Sequence: Ground taxi → Takeoff → Hover → Mission → RTL

⚙️ Configuration:
• Safety distance: 2-15 meters (default: 5m)
• Formation spacing: Configurable (default: 6m)
• Simulation speed: 0.1x to 5x real-time

📊 Analysis Tools:
• 3D trajectory visualization with collision markers
• Collision event logging with JSON export
• Modified mission file generation with LOITER commands
• Real-time status monitoring

🎨 Visualization:
• Professional 3D rendering with multiple view modes
• Blender-style collision warning lines
• Color-coded drone trajectories
• Interactive zoom and rotation controls

For technical support or advanced features, refer to the documentation
or contact the development team.
"""
        
        # Create a scrollable text window for the manual
        manual_window = tk.Toplevel(self.window)
        manual_window.title("User Manual")
        manual_window.geometry("800x600")
        manual_window.configure(bg=SimulatorConfig.UI_COLORS['background'])
        
        # Create text widget with scrollbar
        text_frame = tk.Frame(manual_window, bg=SimulatorConfig.UI_COLORS['background'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, 
                             bg=SimulatorConfig.UI_COLORS['panel'],
                             fg=SimulatorConfig.UI_COLORS['text'],
                             font=('Consolas', 10))
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget.configure(yscrollcommand=scrollbar.set)
        text_widget.insert(tk.END, manual_text)
        text_widget.configure(state=tk.DISABLED)
    
    def _show_system_info(self) -> None:
        """Show system information dialog"""
        try:
            import platform
            import psutil
            import matplotlib
            import pandas
            import numpy
            
            system_info = f"""System Information:

🖥️ Platform:
  OS: {platform.system()} {platform.release()}
  Architecture: {platform.architecture()[0]}
  Python: {platform.python_version()}
  Processor: {platform.processor()}

💾 Memory:
  Total: {psutil.virtual_memory().total / 1024**3:.1f} GB
  Available: {psutil.virtual_memory().available / 1024**3:.1f} GB
  Usage: {psutil.virtual_memory().percent}%

🔧 CPU:
  Cores: {psutil.cpu_count()}
  Usage: {psutil.cpu_percent(interval=1)}%

📚 Libraries:
  Matplotlib: {matplotlib.__version__}
  Pandas: {pandas.__version__}
  NumPy: {numpy.__version__}
  Tkinter: {tk.TkVersion}

🚁 Simulator:
  Version: {SimulatorConfig.VERSION}
  Edition: {SimulatorConfig.EDITION}
  Max Drones: {SimulatorConfig.MAX_DRONES}
  Update Rate: {1000/SimulatorConfig.UPDATE_INTERVAL:.1f} FPS
"""
            
        except ImportError as e:
            system_info = f"System information partially unavailable: {e}"
        
        messagebox.showinfo("System Information", system_info)
    
    def _show_about(self) -> None:
        """Show about dialog"""
        about_text = f"""{SimulatorConfig.WINDOW_TITLE}

Version: {SimulatorConfig.VERSION}
Edition: {SimulatorConfig.EDITION}

🚀 Professional Features:
• Advanced collision detection with logging
• 2x2 formation takeoff with 6m spacing
• Real-time 3D trajectory visualization
• QGC waypoint file generation and modification
• Comprehensive collision avoidance system
• Professional GUI with English interface

🛡️ Safety Systems:
• Real-time collision monitoring (every 0.1s)
• Priority-based conflict resolution
• Automatic LOITER delay insertion
• JSON collision event logging
• Modified mission file export

🎯 Technical Specifications:
• Earth coordinate system with curvature correction
• High-performance 3D rendering engine
• Modular system architecture
• Professional logging and diagnostics
• Mouse wheel zoom and interactive controls

Development Team: Drone Path Planning Laboratory
© 2024 Advanced Drone Swarm Simulator Project

This software is designed for professional drone mission planning
and collision avoidance research applications.
"""
        
        messagebox.showinfo("About", about_text)
    
    def set_status(self, message: str) -> None:
        """Set window status (if status bar exists)"""
        # This could be extended to show status in a status bar
        logger.info(f"Status: {message}")
    
    def show_error(self, title: str, message: str) -> None:
        """Show error dialog"""
        messagebox.showerror(title, message)
        logger.error(f"Error dialog: {title} - {message}")
    
    def show_warning(self, title: str, message: str) -> None:
        """Show warning dialog"""
        messagebox.showwarning(title, message)
        logger.warning(f"Warning dialog: {title} - {message}")
    
    def show_info(self, title: str, message: str) -> None:
        """Show info dialog"""
        messagebox.showinfo(title, message)
        logger.info(f"Info dialog: {title} - {message}")
    
    def ask_yes_no(self, title: str, message: str) -> bool:
        """Show yes/no dialog"""
        result = messagebox.askyesno(title, message)
        logger.info(f"Yes/No dialog: {title} - Result: {result}")
        return result
    
    def get_window(self) -> Optional[tk.Tk]:
        """Get the main window instance"""
        return self.window
    
    def destroy(self) -> None:
        """Destroy the window"""
        if self.window:
            logger.info("Destroying main window")
            self.window.destroy()
            self.window = None


# Example usage
if __name__ == "__main__":
    # Test main window creation
    logging.basicConfig(level=logging.DEBUG)
    
    def test_callback():
        print("Test callback executed")
    
    # Create main window
    main_window = MainWindow()
    window = main_window.create_window()
    
    # Register test callbacks
    main_window.register_callback('test_action', test_callback)
    
    # Create menu and bind shortcuts
    main_window.create_menu()
    main_window.bind_shortcuts()
    
    print("Main window test - Press Escape to exit")
    
    # Run test
    window.mainloop()