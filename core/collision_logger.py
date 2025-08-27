#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collision Logger Module - New feature for v5.1
Professional collision event documentation and analysis
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from config.settings import CollisionLogConfig, SimulatorConfig

logger = logging.getLogger(__name__)

class CollisionLogger:
    """
    Professional collision event logger with JSON export capability
    New feature for v5.1 - Enhanced collision documentation
    """
    
    def __init__(self):
        self.collision_events: List[Dict] = []
        self.log_file: Optional[str] = None
        self.config = CollisionLogConfig()
        
    def initialize_log_file(self, base_name: str = None) -> str:
        """
        Initialize collision log file with timestamp
        
        Args:
            base_name: Base name for log file (default: collision_log)
            
        Returns:
            Generated log file name
        """
        if base_name is None:
            base_name = self.config.FILE_EXTENSION.replace('.', '')
            base_name = f"{self.config.__class__.__name__.lower()}_log"
            
        timestamp = datetime.now().strftime(self.config.TIMESTAMP_FORMAT)
        self.log_file = f"{base_name}_{timestamp}{self.config.FILE_EXTENSION}"
        
        logger.info(f"Collision logger initialized: {self.log_file}")
        return self.log_file
        
    def log_collision(self, collision_data: Dict) -> None:
        """
        Log collision event with comprehensive data
        
        Args:
            collision_data: Dictionary containing collision information
        """
        event = {
            'timestamp': datetime.now().isoformat(),
            'simulation_time': collision_data.get('time', 0),
            'drone1': collision_data.get('drone1'),
            'drone2': collision_data.get('drone2'),
            'distance': collision_data.get('distance'),
            'severity': collision_data.get('severity'),
            'position1': collision_data.get('position1'),
            'position2': collision_data.get('position2'),
            'waypoint1_index': collision_data.get('waypoint1_index'),
            'waypoint2_index': collision_data.get('waypoint2_index')
        }
        
        self.collision_events.append(event)
        
        # Log to console with appropriate level
        if event['severity'] == 'critical':
            logger.error(f"CRITICAL COLLISION: {event['drone1']} vs {event['drone2']} "
                        f"at {event['distance']:.2f}m (sim time: {event['simulation_time']:.1f}s)")
        else:
            logger.warning(f"Collision warning: {event['drone1']} vs {event['drone2']} "
                          f"at {event['distance']:.2f}m (sim time: {event['simulation_time']:.1f}s)")
        
    def get_collision_statistics(self) -> Dict:
        """
        Generate collision statistics for analysis
        
        Returns:
            Dictionary with collision statistics
        """
        if not self.collision_events:
            return {
                'total_events': 0,
                'critical_events': 0,
                'warning_events': 0,
                'affected_drones': [],
                'min_distance': None,
                'avg_distance': None,
                'duration_range': None
            }
        
        critical_count = sum(1 for event in self.collision_events 
                           if event['severity'] == 'critical')
        warning_count = len(self.collision_events) - critical_count
        
        # Collect affected drones
        affected_drones = set()
        distances = []
        times = []
        
        for event in self.collision_events:
            affected_drones.add(event['drone1'])
            affected_drones.add(event['drone2'])
            distances.append(event['distance'])
            times.append(event['simulation_time'])
        
        return {
            'total_events': len(self.collision_events),
            'critical_events': critical_count,
            'warning_events': warning_count,
            'affected_drones': sorted(list(affected_drones)),
            'min_distance': min(distances),
            'avg_distance': sum(distances) / len(distances),
            'max_distance': max(distances),
            'duration_range': (min(times), max(times)) if times else None,
            'first_collision_time': min(times) if times else None,
            'last_collision_time': max(times) if times else None
        }
    
    def clear_events(self) -> int:
        """
        Clear all collision events
        
        Returns:
            Number of events cleared
        """
        count = len(self.collision_events)
        self.collision_events.clear()
        logger.info(f"Cleared {count} collision events")
        return count
    
    def export_collision_log(self, export_path: str = None) -> Optional[str]:
        """
        Export collision log to JSON file with statistics
        
        Args:
            export_path: Custom export path (optional)
            
        Returns:
            Path to exported file, or None if failed
        """
        if not self.collision_events:
            logger.warning("No collision events to export")
            return None
            
        # Determine export path
        if not export_path:
            if self.log_file:
                export_path = self.log_file
            else:
                timestamp = datetime.now().strftime(self.config.TIMESTAMP_FORMAT)
                export_path = f"collision_log_{timestamp}{self.config.FILE_EXTENSION}"
        
        try:
            # Generate statistics
            statistics = self.get_collision_statistics()
            
            # Prepare export data
            export_data = {
                'metadata': {
                    'total_events': len(self.collision_events),
                    'export_time': datetime.now().isoformat(),
                    'simulation_version': SimulatorConfig.VERSION,
                    'simulation_edition': SimulatorConfig.EDITION,
                    'statistics': statistics
                },
                'collision_events': self.collision_events
            }
            
            # Write to file
            with open(export_path, 'w', encoding=self.config.ENSURE_ASCII) as f:
                json.dump(export_data, f, 
                         indent=self.config.INDENT,
                         ensure_ascii=self.config.ENSURE_ASCII)
            
            logger.info(f"Collision log exported successfully: {export_path}")
            logger.info(f"Export summary: {statistics['total_events']} events, "
                       f"{statistics['critical_events']} critical, "
                       f"{statistics['warning_events']} warnings")
            
            return export_path
            
        except Exception as e:
            logger.error(f"Failed to export collision log: {e}")
            return None
    
    def import_collision_log(self, import_path: str) -> bool:
        """
        Import collision log from JSON file
        
        Args:
            import_path: Path to import file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(import_path, 'r', encoding=self.config.ENSURE_ASCII) as f:
                data = json.load(f)
            
            # Validate data structure
            if 'collision_events' not in data:
                raise ValueError("Invalid collision log format: missing 'collision_events'")
            
            # Import events
            imported_events = data['collision_events']
            self.collision_events.extend(imported_events)
            
            logger.info(f"Imported {len(imported_events)} collision events from {import_path}")
            
            # Log metadata if available
            if 'metadata' in data:
                metadata = data['metadata']
                logger.info(f"Imported log metadata: version {metadata.get('simulation_version', 'unknown')}, "
                           f"exported {metadata.get('export_time', 'unknown')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to import collision log from {import_path}: {e}")
            return False
    
    def get_events_by_drone(self, drone_id: str) -> List[Dict]:
        """
        Get all collision events involving a specific drone
        
        Args:
            drone_id: ID of the drone to search for
            
        Returns:
            List of collision events involving the drone
        """
        return [event for event in self.collision_events 
                if event['drone1'] == drone_id or event['drone2'] == drone_id]
    
    def get_events_by_severity(self, severity: str) -> List[Dict]:
        """
        Get all collision events of a specific severity
        
        Args:
            severity: 'critical' or 'warning'
            
        Returns:
            List of collision events with specified severity
        """
        return [event for event in self.collision_events 
                if event['severity'] == severity]
    
    def get_events_in_time_range(self, start_time: float, end_time: float) -> List[Dict]:
        """
        Get collision events within a specific time range
        
        Args:
            start_time: Start time in simulation seconds
            end_time: End time in simulation seconds
            
        Returns:
            List of collision events in the time range
        """
        return [event for event in self.collision_events 
                if start_time <= event['simulation_time'] <= end_time]
    
    def __len__(self) -> int:
        """Return number of logged collision events"""
        return len(self.collision_events)
    
    def __bool__(self) -> bool:
        """Return True if there are collision events"""
        return bool(self.collision_events)
    
    def __str__(self) -> str:
        """String representation of collision logger"""
        stats = self.get_collision_statistics()
        return (f"CollisionLogger: {stats['total_events']} events "
                f"({stats['critical_events']} critical, {stats['warning_events']} warnings)")