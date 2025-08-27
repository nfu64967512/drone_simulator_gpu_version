#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flight Management Module
Includes takeoff manager and QGC waypoint generator for v5.1
"""

import logging
from typing import List, Dict, Tuple, Optional
from config.settings import TakeoffConfig
from core.coordinate_system import EarthCoordinateSystem

logger = logging.getLogger(__name__)

class TakeoffManager:
    """
    Takeoff manager with improved 2x2 formation
    Enhanced with 6m spacing for v5.1
    """
    
    def __init__(self, config: TakeoffConfig, coordinate_system: EarthCoordinateSystem):
        self.config = config
        self.coordinate_system = coordinate_system
        
        logger.info(f"TakeoffManager initialized with {config.formation_spacing}m spacing")
        
    def generate_takeoff_formation(self, base_lat: float, base_lon: float) -> List[Tuple[float, float]]:
        """
        Generate 2x2 takeoff formation with configurable spacing
        
        Args:
            base_lat: Base latitude for formation center
            base_lon: Base longitude for formation center
            
        Returns:
            List of (lat, lon) tuples for each drone takeoff position
        """
        if not self.coordinate_system.is_origin_set():
            logger.warning("Coordinate system origin not set, using provided coordinates as origin")
            self.coordinate_system.set_origin(base_lat, base_lon)
        
        # Base point east offset as takeoff area
        base_x, base_y = self.coordinate_system.lat_lon_to_meters(base_lat, base_lon)
        base_x += self.config.east_offset
        
        # 2x2 formation coordinates with configurable spacing
        spacing = self.config.formation_spacing
        positions = [
            (base_x - spacing/2, base_y - spacing/2),  # Bottom-left - Drone_1
            (base_x + spacing/2, base_y - spacing/2),  # Bottom-right - Drone_2
            (base_x - spacing/2, base_y + spacing/2),  # Top-left - Drone_3
            (base_x + spacing/2, base_y + spacing/2)   # Top-right - Drone_4
        ]
        
        # Convert back to lat/lon
        formation_points = []
        for i, (x, y) in enumerate(positions, 1):
            lat, lon = self.coordinate_system.meters_to_lat_lon(x, y)
            formation_points.append((lat, lon))
            logger.debug(f"Drone_{i} takeoff position: {lat:.8f}, {lon:.8f}")
            
        logger.info(f"Generated 2x2 takeoff formation with {spacing}m spacing at "
                   f"{self.config.east_offset}m east of base point")
        
        return formation_points
    
    def validate_formation_spacing(self, min_spacing: float = 2.0, max_spacing: float = 20.0) -> bool:
        """
        Validate formation spacing is within safe limits
        
        Args:
            min_spacing: Minimum safe spacing in meters
            max_spacing: Maximum practical spacing in meters
            
        Returns:
            True if spacing is valid
        """
        spacing = self.config.formation_spacing
        
        if spacing < min_spacing:
            logger.error(f"Formation spacing {spacing}m is too small (minimum: {min_spacing}m)")
            return False
            
        if spacing > max_spacing:
            logger.warning(f"Formation spacing {spacing}m is very large (maximum recommended: {max_spacing}m)")
            
        return True
    
    def get_formation_center(self, base_lat: float, base_lon: float) -> Tuple[float, float]:
        """
        Get the center point of the takeoff formation
        
        Args:
            base_lat: Base latitude
            base_lon: Base longitude
            
        Returns:
            Tuple of (center_lat, center_lon)
        """
        base_x, base_y = self.coordinate_system.lat_lon_to_meters(base_lat, base_lon)
        center_x = base_x + self.config.east_offset
        center_y = base_y
        
        center_lat, center_lon = self.coordinate_system.meters_to_lat_lon(center_x, center_y)
        return center_lat, center_lon
    
    def update_config(self, new_config: TakeoffConfig) -> None:
        """
        Update takeoff configuration
        
        Args:
            new_config: New takeoff configuration
        """
        old_spacing = self.config.formation_spacing
        self.config = new_config
        
        logger.info(f"Takeoff configuration updated: spacing {old_spacing}m → {new_config.formation_spacing}m")


class QGCWaypointGenerator:
    """
    QGC waypoint file generator with precise LOITER insertion
    Enhanced for professional mission planning
    """
    
    def __init__(self):
        self.sequence_counter = 0
        
    def generate_complete_mission(self, drone_id: str, waypoints: List[Dict], 
                                 conflict_info: Dict = None) -> List[str]:
        """
        Generate complete mission file with LOITER insertion at specified waypoint
        
        Args:
            drone_id: Identifier for the drone
            waypoints: List of waypoint dictionaries
            conflict_info: Optional conflict information for LOITER insertion
            
        Returns:
            List of mission file lines in QGC format
        """
        lines = ["QGC WPL 110"]
        self.sequence_counter = 0
        
        if not waypoints:
            logger.error(f"No waypoints provided for {drone_id}")
            return lines
        
        # HOME point
        home_wp = waypoints[0]
        lines.append(f"{self.sequence_counter}\t1\t0\t179\t0\t0\t0\t0\t"
                    f"{home_wp['lat']:.8f}\t{home_wp['lon']:.8f}\t{home_wp['alt']:.2f}\t1")
        self.sequence_counter += 1
        
        # Speed setting (8 m/s)
        lines.append(f"{self.sequence_counter}\t0\t3\t178\t0\t8.0\t0\t0\t0\t0\t0\t1")
        self.sequence_counter += 1
        
        # Takeoff
        lines.append(f"{self.sequence_counter}\t0\t3\t22\t0\t0\t0\t0\t"
                    f"{home_wp['lat']:.8f}\t{home_wp['lon']:.8f}\t10.0\t1")
        self.sequence_counter += 1
        
        # Hover wait
        lines.append(f"{self.sequence_counter}\t0\t3\t19\t2.0\t0\t0\t0\t0\t0\t0\t1")
        self.sequence_counter += 1
        
        # Process mission waypoints and insert LOITER if needed
        loiter_waypoint_index = None
        loiter_time = 0.0
        
        if conflict_info:
            loiter_waypoint_index = conflict_info.get('insert_after_waypoint', None)
            loiter_time = conflict_info.get('wait_time', 5.0)
            
            logger.info(f"{drone_id}: Will insert {loiter_time:.1f}s LOITER after waypoint {loiter_waypoint_index}")
        
        # Mission waypoints (starting from waypoints[1:] since waypoints[0] is HOME)
        for wp_idx, wp in enumerate(waypoints[1:], start=2):  # Waypoint numbering starts from 2
            # Add waypoint
            lines.append(f"{self.sequence_counter}\t0\t3\t16\t0\t0\t0\t0\t"
                        f"{wp['lat']:.8f}\t{wp['lon']:.8f}\t{wp['alt']:.2f}\t1")
            self.sequence_counter += 1
            
            # Check if LOITER should be inserted after this waypoint
            if loiter_waypoint_index == wp_idx and loiter_time > 0:
                lines.append(f"{self.sequence_counter}\t0\t3\t19\t{loiter_time:.1f}\t0\t0\t0\t0\t0\t0\t1")
                self.sequence_counter += 1
                logger.info(f"{drone_id}: Inserted {loiter_time:.1f}s LOITER after waypoint {wp_idx} "
                           f"(sequence {self.sequence_counter-1})")
        
        # RTL (Return to Launch)
        lines.append(f"{self.sequence_counter}\t0\t3\t20\t0\t0\t0\t0\t0\t0\t0\t1")
        
        logger.info(f"Generated complete mission for {drone_id}: {len(lines)} commands")
        
        return lines
    
    def generate_mission_with_conflicts(self, drone_id: str, waypoints: List[Dict], 
                                      conflicts: List[Dict]) -> List[str]:
        """
        Generate mission file based on conflict analysis
        
        Args:
            drone_id: Identifier for the drone
            waypoints: List of waypoint dictionaries
            conflicts: List of conflict information
            
        Returns:
            List of mission file lines with conflict avoidance
        """
        # Find conflicts affecting this drone
        relevant_conflicts = [c for c in conflicts if c['waiting_drone'] == drone_id]
        
        if not relevant_conflicts:
            # No conflicts, generate standard mission
            logger.info(f"No conflicts found for {drone_id}, generating standard mission")
            return self.generate_complete_mission(drone_id, waypoints)
        
        # Select earliest conflict for LOITER insertion
        earliest_conflict = min(relevant_conflicts, key=lambda c: c['time'])
        
        conflict_info = {
            'insert_after_waypoint': earliest_conflict['waypoint2_index'],
            'wait_time': earliest_conflict['wait_time'],
            'conflict_with': earliest_conflict['priority_drone']
        }
        
        logger.info(f"Generating avoidance mission for {drone_id}: wait {conflict_info['wait_time']:.1f}s "
                   f"after waypoint {conflict_info['insert_after_waypoint']} (avoid {conflict_info['conflict_with']})")
        
        return self.generate_complete_mission(drone_id, waypoints, conflict_info)
    
    def generate_basic_mission(self, drone_id: str, waypoints: List[Dict]) -> List[str]:
        """
        Generate basic mission without any special commands
        
        Args:
            drone_id: Identifier for the drone
            waypoints: List of waypoint dictionaries
            
        Returns:
            List of basic mission file lines
        """
        lines = ["QGC WPL 110"]
        sequence = 0
        
        for i, wp in enumerate(waypoints):
            if i == 0:
                # HOME point
                lines.append(f"{sequence}\t1\t0\t179\t0\t0\t0\t0\t"
                            f"{wp['lat']:.8f}\t{wp['lon']:.8f}\t{wp['alt']:.2f}\t1")
            else:
                # Mission waypoint
                lines.append(f"{sequence}\t0\t3\t16\t0\t0\t0\t0\t"
                            f"{wp['lat']:.8f}\t{wp['lon']:.8f}\t{wp['alt']:.2f}\t1")
            sequence += 1
        
        # RTL
        lines.append(f"{sequence}\t0\t3\t20\t0\t0\t0\t0\t0\t0\t0\t1")
        
        logger.info(f"Generated basic mission for {drone_id}: {len(lines)} commands")
        
        return lines
    
    def validate_waypoints(self, waypoints: List[Dict]) -> bool:
        """
        Validate waypoint data structure
        
        Args:
            waypoints: List of waypoint dictionaries to validate
            
        Returns:
            True if waypoints are valid
        """
        if not waypoints:
            logger.error("No waypoints provided")
            return False
        
        required_fields = ['lat', 'lon', 'alt']
        
        for i, wp in enumerate(waypoints):
            if not isinstance(wp, dict):
                logger.error(f"Waypoint {i} is not a dictionary")
                return False
                
            for field in required_fields:
                if field not in wp:
                    logger.error(f"Waypoint {i} missing required field: {field}")
                    return False
                    
                try:
                    float(wp[field])
                except (ValueError, TypeError):
                    logger.error(f"Waypoint {i} field {field} is not a valid number: {wp[field]}")
                    return False
        
        logger.debug(f"Validated {len(waypoints)} waypoints successfully")
        return True
    
    def add_custom_command(self, lines: List[str], command_type: int, 
                          params: List[float], position: Tuple[float, float, float] = None) -> List[str]:
        """
        Add custom command to mission file
        
        Args:
            lines: Existing mission lines
            command_type: MAVLink command type
            params: Command parameters
            position: Optional (lat, lon, alt) position
            
        Returns:
            Updated mission lines
        """
        sequence = len(lines) - 1  # Insert before RTL
        
        # Ensure params list has 7 elements
        params_padded = (params + [0.0] * 7)[:7]
        
        if position:
            lat, lon, alt = position
        else:
            lat = lon = alt = 0.0
        
        command_line = f"{sequence}\t0\t3\t{command_type}\t"
        command_line += "\t".join([f"{p:.1f}" for p in params_padded])
        command_line += f"\t{lat:.8f}\t{lon:.8f}\t{alt:.2f}\t1"
        
        # Insert before RTL
        lines.insert(-1, command_line)
        
        # Update RTL sequence number
        rtl_parts = lines[-1].split('\t')
        rtl_parts[0] = str(sequence + 1)
        lines[-1] = '\t'.join(rtl_parts)
        
        logger.debug(f"Added custom command {command_type} at sequence {sequence}")
        
        return lines
    
    def estimate_mission_time(self, waypoints: List[Dict], cruise_speed: float = 8.0) -> float:
        """
        Estimate total mission time based on waypoints and speed
        
        Args:
            waypoints: List of waypoint dictionaries
            cruise_speed: Cruise speed in m/s
            
        Returns:
            Estimated mission time in seconds
        """
        if len(waypoints) < 2:
            return 0.0
        
        total_time = 0.0
        
        # Takeoff and hover time
        total_time += 7.0  # 2s taxi + 5s takeoff + 2s hover (from config)
        
        # Mission waypoint time
        for i in range(1, len(waypoints)):
            prev_wp = waypoints[i-1]
            curr_wp = waypoints[i]
            
            # Simple 2D distance calculation for estimation
            lat_diff = curr_wp['lat'] - prev_wp['lat']
            lon_diff = curr_wp['lon'] - prev_wp['lon']
            alt_diff = curr_wp['alt'] - prev_wp['alt']
            
            # Approximate distance in meters
            distance = ((lat_diff * 111111)**2 + (lon_diff * 111111)**2 + alt_diff**2)**0.5
            
            # Flight time
            total_time += distance / cruise_speed
        
        logger.debug(f"Estimated mission time: {total_time:.1f}s for {len(waypoints)} waypoints")
        
        return total_time
    
    def __str__(self) -> str:
        """String representation of QGC waypoint generator"""
        return f"QGCWaypointGenerator(sequence_counter: {self.sequence_counter})"