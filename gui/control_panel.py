#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control Panel UI Module
Compact and professional control interface for v5.1
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Callable, Optional
from config.settings import SimulatorConfig, UILabels, TakeoffConfig, SafetyConfig

logger = logging.getLogger(__name__)

class ControlPanel:
    """
    Professional control panel with compact design
    Features English UI and enhanced functionality for v5.1
    """
    
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.callbacks: Dict[str, Callable] = {}
        self.widgets: Dict[str, tk.Widget] = {}
        self.variables: Dict[str, tk.Variable] = {}
        
        # Initialize variables
        self._init_variables()
        
        logger.info("Control panel initialized")
    
    def _init_variables(self) -> None:
        """Initialize tkinter variables"""
        self.variables = {
            'time_var': tk.DoubleVar(value=0.0),
            'speed_var': tk.DoubleVar(value=1.0),
            'safety_var': tk.DoubleVar(value=5.0),
            'max_time': tk.DoubleVar(value=100.0)
        }
        
        logger.debug("Control panel variables initialized")
    
    def register_callback(self, event_name: str, callback: Callable) -> None:
        """
        Register callback for control events
        
        Args:
            event_name: Name of the event
            callback: Callback function
        """
        self.callbacks[event_name] = callback
        logger.debug(f"Registered callback for: {event_name}")
    
    def create_panel(self) -> tk.Frame:
        """
        Create the complete control panel
        
        Returns:
            Main control panel frame
        """
        logger.info("Creating control panel")
        
        # Main control frame
        control_frame = tk.Frame(self.parent, bg=SimulatorConfig.UI_COLORS['panel'], width=280)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        control_frame.pack_propagate(False)
        
        # Create sections
        self._create_title_section(control_frame)
        self._create_file_section(control_frame)
        self._create_drone_status_section(control_frame)
        self._create_playback_section(control_frame)
        self._create_settings_section(control_frame)
        self._create_status_section(control_frame)
        self._create_warnings_section(control_frame)
        
        logger.info("Control panel created successfully")
        return control_frame
    
    def _create_title_section(self, parent: tk.Widget) -> None:
        """Create title section"""
        title_frame = tk.Frame(parent, bg=SimulatorConfig.UI_COLORS['panel'])
        title_frame.pack(fill=tk.X, padx=10, pady=10)
        
        title_label = tk.Label(
            title_frame, 
            text=UILabels.DRONE_SIMULATOR,
            font=('Arial', 14, 'bold'),
            fg=SimulatorConfig.UI_COLORS['accent'],
            bg=SimulatorConfig.UI_COLORS['panel']
        )
        title_label.pack()
        
        # Version info
        version_label = tk.Label(
            title_frame,
            text=f"v{SimulatorConfig.VERSION} - {SimulatorConfig.EDITION}",
            font=('Arial', 8),
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel']
        )
        version_label.pack()
        
        self.widgets['title_frame'] = title_frame
    
    def _create_file_section(self, parent: tk.Widget) -> None:
        """Create file operations section"""
        file_frame = tk.LabelFrame(
            parent, 
            text=UILabels.MISSION_FILES,
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 10, 'bold')
        )
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_frame = tk.Frame(file_frame, bg=SimulatorConfig.UI_COLORS['panel'])
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # File operation buttons
        buttons = [
            (UILabels.LOAD_QGC, 'load_qgc_files', SimulatorConfig.BUTTON_CONFIGS['play']),
            (UILabels.LOAD_CSV, 'load_csv_files', {'bg': '#007bff', 'fg': 'white'}),
            (UILabels.CREATE_TEST, 'create_test_mission', {'bg': '#6f42c1', 'fg': 'white'})
        ]
        
        for text, callback_name, style in buttons:
            btn = tk.Button(
                btn_frame,
                text=text,
                command=lambda cb=callback_name: self._execute_callback(cb),
                font=('Arial', 8, 'bold'),
                width=8,
                **style
            )
            btn.pack(side=tk.LEFT, padx=1)
            self.widgets[f'{callback_name}_button'] = btn
        
        self.widgets['file_frame'] = file_frame
    
    def _create_drone_status_section(self, parent: tk.Widget) -> None:
        """Create drone status indicators section"""
        drone_frame = tk.LabelFrame(
            parent,
            text=UILabels.DRONE_STATUS,
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 10, 'bold')
        )
        drone_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.widgets['drone_indicators'] = {}
        
        for i in range(SimulatorConfig.MAX_DRONES):
            row = tk.Frame(drone_frame, bg=SimulatorConfig.UI_COLORS['panel'])
            row.pack(fill=tk.X, pady=1)
            
            # Color indicator
            color_dot = tk.Label(
                row,
                text="●",
                font=('Arial', 12),
                fg=SimulatorConfig.DRONE_COLORS[i],
                bg=SimulatorConfig.UI_COLORS['panel']
            )
            color_dot.pack(side=tk.LEFT)
            
            # Status label
            status_label = tk.Label(
                row,
                text=UILabels.STANDBY,
                fg='#888',
                bg=SimulatorConfig.UI_COLORS['panel'],
                font=('Arial', 9)
            )
            status_label.pack(side=tk.LEFT, padx=(5, 0))
            
            self.widgets['drone_indicators'][f'drone_{i+1}'] = status_label
        
        self.widgets['drone_frame'] = drone_frame
    
    def _create_playback_section(self, parent: tk.Widget) -> None:
        """Create playback controls section"""
        play_frame = tk.LabelFrame(
            parent,
            text=UILabels.CONTROLS,
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 10, 'bold')
        )
        play_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Control buttons
        btn_row = tk.Frame(play_frame, bg=SimulatorConfig.UI_COLORS['panel'])
        btn_row.pack(fill=tk.X, padx=5, pady=5)
        
        # Play/Pause button
        self.widgets['play_button'] = tk.Button(
            btn_row,
            text=UILabels.PLAY,
            command=lambda: self._execute_callback('toggle_play'),
            font=('Arial', 12, 'bold'),
            width=3,
            **SimulatorConfig.BUTTON_CONFIGS['play']
        )
        self.widgets['play_button'].pack(side=tk.LEFT, padx=1)
        
        # Control buttons
        control_buttons = [
            (UILabels.STOP, 'stop_simulation', SimulatorConfig.BUTTON_CONFIGS['stop']),
            (UILabels.RESET, 'reset_simulation', SimulatorConfig.BUTTON_CONFIGS['reset']),
            (UILabels.EXPORT_MISSIONS, 'export_modified_missions', SimulatorConfig.BUTTON_CONFIGS['export']),
            (UILabels.EXPORT_LOG, 'export_collision_log', SimulatorConfig.BUTTON_CONFIGS['log'])
        ]
        
        for text, callback_name, style in control_buttons:
            btn = tk.Button(
                btn_row,
                text=text,
                command=lambda cb=callback_name: self._execute_callback(cb),
                font=('Arial', 12, 'bold'),
                width=3,
                **style
            )
            btn.pack(side=tk.LEFT, padx=1)
            self.widgets[f'{callback_name}_button'] = btn
        
        # Time display
        self.widgets['time_label'] = tk.Label(
            play_frame,
            text="00:00 / 00:00",
            fg=SimulatorConfig.UI_COLORS['accent'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 11, 'bold')
        )
        self.widgets['time_label'].pack(pady=3)
        
        # Time slider
        self.widgets['time_slider'] = tk.Scale(
            play_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.variables['time_var'],
            command=self._on_time_change,
            bg=SimulatorConfig.UI_COLORS['panel'],
            fg=SimulatorConfig.UI_COLORS['text'],
            highlightbackground=SimulatorConfig.UI_COLORS['panel'],
            troughcolor='#404040',
            length=250,
            resolution=0.1
        )
        self.widgets['time_slider'].pack(fill=tk.X, padx=5, pady=3)
        
        self.widgets['play_frame'] = play_frame
    
    def _create_settings_section(self, parent: tk.Widget) -> None:
        """Create settings section"""
        settings_frame = tk.Frame(parent, bg=SimulatorConfig.UI_COLORS['panel'])
        settings_frame.pack(fill=tk.X, padx=15, pady=3)
        
        # Speed control
        speed_frame = tk.Frame(settings_frame, bg=SimulatorConfig.UI_COLORS['panel'])
        speed_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(
            speed_frame,
            text="Speed:",
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 8)
        ).pack(side=tk.LEFT)
        
        self.widgets['speed_scale'] = tk.Scale(
            speed_frame,
            from_=SimulatorConfig.TIME_SCALE_RANGE[0],
            to=SimulatorConfig.TIME_SCALE_RANGE[1],
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.variables['speed_var'],
            command=self._on_speed_change,
            bg=SimulatorConfig.UI_COLORS['panel'],
            fg=SimulatorConfig.UI_COLORS['text'],
            length=120
        )
        self.widgets['speed_scale'].pack(side=tk.RIGHT)
        
        # Safety distance control
        safety_frame = tk.Frame(settings_frame, bg=SimulatorConfig.UI_COLORS['panel'])
        safety_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(
            safety_frame,
            text="Safety:",
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 8)
        ).pack(side=tk.LEFT)
        
        self.widgets['safety_scale'] = tk.Scale(
            safety_frame,
            from_=SimulatorConfig.SAFETY_DISTANCE_RANGE[0],
            to=SimulatorConfig.SAFETY_DISTANCE_RANGE[1],
            resolution=0.5,
            orient=tk.HORIZONTAL,
            variable=self.variables['safety_var'],
            command=self._on_safety_change,
            bg=SimulatorConfig.UI_COLORS['panel'],
            fg=SimulatorConfig.UI_COLORS['text'],
            length=120
        )
        self.widgets['safety_scale'].pack(side=tk.RIGHT)
        
        self.widgets['settings_frame'] = settings_frame
    
    def _create_status_section(self, parent: tk.Widget) -> None:
        """Create status information section"""
        status_frame = tk.LabelFrame(
            parent,
            text=UILabels.STATUS_INFO,
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 11, 'bold')
        )
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create text widget with scrollbar
        text_container = tk.Frame(status_frame, bg=SimulatorConfig.UI_COLORS['panel'])
        text_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.widgets['status_text'] = tk.Text(
            text_container,
            bg='#1a1a1a',
            fg=SimulatorConfig.UI_COLORS['accent'],
            font=('Consolas', 11),
            height=12,
            wrap=tk.WORD
        )
        
        status_scroll = ttk.Scrollbar(
            text_container,
            orient="vertical",
            command=self.widgets['status_text'].yview
        )
        
        self.widgets['status_text'].configure(yscrollcommand=status_scroll.set)
        self.widgets['status_text'].pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        status_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.widgets['status_frame'] = status_frame
    
    def _create_warnings_section(self, parent: tk.Widget) -> None:
        """Create collision warnings section"""
        warning_frame = tk.LabelFrame(
            parent,
            text=UILabels.COLLISION_ALERTS,
            fg=SimulatorConfig.UI_COLORS['text'],
            bg=SimulatorConfig.UI_COLORS['panel'],
            font=('Arial', 11, 'bold')
        )
        warning_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.widgets['warning_text'] = tk.Text(
            warning_frame,
            height=6,
            bg='#1a1a1a',
            fg='#ff5722',
            font=('Consolas', 10),
            wrap=tk.WORD
        )
        self.widgets['warning_text'].pack(fill=tk.X, padx=5, pady=5)
        
        # Configure text tags for different warning levels
        self.widgets['warning_text'].tag_config(
            'danger',
            foreground='#ff5722',
            font=('Consolas', 10, 'bold')
        )
        self.widgets['warning_text'].tag_config(
            'safe',
            foreground='#4caf50',
            font=('Consolas', 10)
        )
        
        self.widgets['warning_frame'] = warning_frame
    
    def _execute_callback(self, callback_name: str) -> None:
        """Execute registered callback"""
        if callback_name in self.callbacks:
            try:
                self.callbacks[callback_name]()
                logger.debug(f"Executed callback: {callback_name}")
            except Exception as e:
                logger.error(f"Error executing callback {callback_name}: {e}")
        else:
            logger.warning(f"No callback registered for: {callback_name}")
    
    def _on_time_change(self, value: str) -> None:
        """Handle time slider change"""
        self._execute_callback('on_time_change')
    
    def _on_speed_change(self, value: str) -> None:
        """Handle speed change"""
        self._execute_callback('on_speed_change')
    
    def _on_safety_change(self, value: str) -> None:
        """Handle safety distance change"""
        self._execute_callback('on_safety_change')
    
    def update_play_button(self, is_playing: bool) -> None:
        """Update play button appearance"""
        if 'play_button' in self.widgets:
            if is_playing:
                self.widgets['play_button'].config(
                    text=UILabels.PAUSE,
                    **SimulatorConfig.BUTTON_CONFIGS['pause']
                )
            else:
                self.widgets['play_button'].config(
                    text=UILabels.PLAY,
                    **SimulatorConfig.BUTTON_CONFIGS['play']
                )
    
    def update_time_display(self, current_time: float, max_time: float) -> None:
        """Update time display"""
        current_min = int(current_time // 60)
        current_sec = int(current_time % 60)
        max_min = int(max_time // 60)
        max_sec = int(max_time % 60)
        
        time_text = f"{current_min:02d}:{current_sec:02d} / {max_min:02d}:{max_sec:02d}"
        
        if 'time_label' in self.widgets:
            self.widgets['time_label'].config(text=time_text)
        
        # Update slider range
        if 'time_slider' in self.widgets:
            self.widgets['time_slider'].config(to=max_time)
    
    def update_drone_status(self, drone_id: str, status: str, color: str = None) -> None:
        """Update drone status indicator"""
        if drone_id in self.widgets.get('drone_indicators', {}):
            indicator = self.widgets['drone_indicators'][drone_id]
            indicator.config(text=status, fg=color or '#4caf50')
            logger.debug(f"Updated {drone_id} status: {status}")
    
    def update_status_text(self, text: str) -> None:
        """Update status text area"""
        if 'status_text' in self.widgets:
            status_text = self.widgets['status_text']
            status_text.delete(1.0, tk.END)
            status_text.insert(tk.END, text)
            status_text.see(tk.END)
    
    def update_warning_text(self, text: str, tag: str = None) -> None:
        """Update warning text area"""
        if 'warning_text' in self.widgets:
            warning_text = self.widgets['warning_text']
            warning_text.delete(1.0, tk.END)
            warning_text.insert(tk.END, text, tag)
            warning_text.see(tk.END)
    
    def get_variable_value(self, var_name: str) -> any:
        """Get value of a control variable"""
        if var_name in self.variables:
            return self.variables[var_name].get()
        else:
            logger.warning(f"Variable not found: {var_name}")
            return None
    
    def set_variable_value(self, var_name: str, value: any) -> None:
        """Set value of a control variable"""
        if var_name in self.variables:
            self.variables[var_name].set(value)
            logger.debug(f"Set {var_name} = {value}")
        else:
            logger.warning(f"Variable not found: {var_name}")
    
    def enable_controls(self, enabled: bool = True) -> None:
        """Enable or disable all controls"""
        state = tk.NORMAL if enabled else tk.DISABLED
        
        # Update button states
        for widget_name, widget in self.widgets.items():
            if 'button' in widget_name and isinstance(widget, tk.Button):
                widget.config(state=state)
        
        # Update scale states
        for widget_name, widget in self.widgets.items():
            if 'scale' in widget_name and isinstance(widget, tk.Scale):
                widget.config(state=state)
        
        logger.info(f"Controls {'enabled' if enabled else 'disabled'}")
    
    def get_widget(self, widget_name: str) -> Optional[tk.Widget]:
        """Get widget by name"""
        return self.widgets.get(widget_name)
    
    def highlight_button(self, button_name: str, highlight: bool = True) -> None:
        """Highlight a button temporarily"""
        if button_name in self.widgets:
            button = self.widgets[button_name]
            if highlight:
                button.config(relief=tk.RAISED, borderwidth=3)
            else:
                button.config(relief=tk.FLAT, borderwidth=1)


# Example usage and testing
if __name__ == "__main__":
    # Test control panel
    logging.basicConfig(level=logging.DEBUG)
    
    def test_callback():
        print("Test callback executed")
    
    # Create test window
    root = tk.Tk()
    root.title("Control Panel Test")
    root.geometry("300x800")
    root.configure(bg=SimulatorConfig.UI_COLORS['background'])
    
    # Create control panel
    control_panel = ControlPanel(root)
    
    # Register test callbacks
    control_panel.register_callback('toggle_play', test_callback)
    control_panel.register_callback('create_test_mission', test_callback)
    
    # Create panel
    panel_frame = control_panel.create_panel()
    
    # Test updates
    control_panel.update_drone_status('drone_1', '✓ Ready', '#4caf50')
    control_panel.update_status_text("Control panel test initialized\n\nReady for testing...")
    control_panel.update_warning_text("✅ No warnings", 'safe')
    
    print("Control panel test - Close window to exit")
    
    # Run test
    root.mainloop()