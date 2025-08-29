#!/usr/bin/env python3
"""
任務檔案生成器
支援生成QGC waypoint檔案、修改任務以避免碰撞、批次處理等功能
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
import json
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class QGCCommand:
    """QGC命令數據結構"""
    sequence: int
    current: int = 0
    frame: int = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT
    command: int = 16  # MAV_CMD_NAV_WAYPOINT
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    x: float = 0.0  # latitude or x
    y: float = 0.0  # longitude or y
    z: float = 0.0  # altitude or z
    autocontinue: int = 1

    def to_qgc_line(self) -> str:
        """轉換為QGC檔案行格式"""
        return f"{self.sequence}\t{self.current}\t{self.frame}\t{self.command}\t{self.param1}\t{self.param2}\t{self.param3}\t{self.param4}\t{self.x:.8f}\t{self.y:.8f}\t{self.z:.2f}\t{self.autocontinue}"

class QGCMissionGenerator:
    """QGroundControl任務檔案生成器"""
    
    # MAVLink命令常數
    CMD_NAV_WAYPOINT = 16
    CMD_NAV_LOITER_TIME = 19
    CMD_NAV_RETURN_TO_LAUNCH = 20
    CMD_NAV_TAKEOFF = 22
    CMD_SET_HOME = 179
    CMD_DO_CHANGE_SPEED = 178
    
    def __init__(self):
        self.sequence_counter = 0
        self.default_speed = 8.0  # m/s
        self.default_takeoff_altitude = 10.0  # m
        
    def reset_sequence(self):
        """重置序列計數器"""
        self.sequence_counter = 0
    
    def create_standard_mission(self, drone_id: str, waypoints: List[Dict], 
                               include_rtl: bool = True) -> List[QGCCommand]:
        """創建標準任務"""
        commands = []
        self.reset_sequence()
        
        if not waypoints:
            raise ValueError("航點列表不能為空")
        
        # 1. HOME點設置
        home_wp = waypoints[0]
        home_cmd = QGCCommand(
            sequence=self.sequence_counter,
            current=1,
            frame=0,
            command=self.CMD_SET_HOME,
            x=home_wp['lat'],
            y=home_wp['lon'],
            z=home_wp.get('alt', 0)
        )
        commands.append(home_cmd)
        self._increment_sequence()
        
        # 2. 速度設置
        speed_cmd = QGCCommand(
            sequence=self.sequence_counter,
            frame=3,
            command=self.CMD_DO_CHANGE_SPEED,
            param2=self.default_speed,
            x=0, y=0, z=0
        )
        commands.append(speed_cmd)
        self._increment_sequence()
        
        # 3. 起飛命令
        takeoff_cmd = QGCCommand(
            sequence=self.sequence_counter,
            frame=3,
            command=self.CMD_NAV_TAKEOFF,
            x=home_wp['lat'],
            y=home_wp['lon'],
            z=self.default_takeoff_altitude
        )
        commands.append(takeoff_cmd)
        self._increment_sequence()
        
        # 4. 懸停穩定
        hover_cmd = QGCCommand(
            sequence=self.sequence_counter,
            frame=3,
            command=self.CMD_NAV_LOITER_TIME,
            param1=2.0,  # 懸停2秒
            x=0, y=0, z=0
        )
        commands.append(hover_cmd)
        self._increment_sequence()
        
        # 5. 任務航點（跳過第一個HOME點）
        for wp in waypoints[1:]:
            nav_cmd = QGCCommand(
                sequence=self.sequence_counter,
                frame=3,
                command=self.CMD_NAV_WAYPOINT,
                x=wp['lat'],
                y=wp['lon'],
                z=wp.get('alt', 15.0)
            )
            commands.append(nav_cmd)
            self._increment_sequence()
        
        # 6. 返回起飛點（可選）
        if include_rtl:
            rtl_cmd = QGCCommand(
                sequence=self.sequence_counter,
                frame=3,
                command=self.CMD_NAV_RETURN_TO_LAUNCH,
                x=0, y=0, z=0
            )
            commands.append(rtl_cmd)
        
        logger.info(f"生成標準任務 {drone_id}: {len(commands)} 個命令")
        return commands
    
    def create_collision_avoidance_mission(self, drone_id: str, waypoints: List[Dict],
                                         avoidance_info: Dict) -> List[QGCCommand]:
        """創建包含碰撞避讓的任務"""
        commands = self.create_standard_mission(drone_id, waypoints, include_rtl=False)
        
        # 在指定航點後插入LOITER命令
        insert_after_waypoint = avoidance_info.get('insert_after_waypoint', 2)
        wait_time = avoidance_info.get('wait_time', 5.0)
        
        # 找到插入點（考慮前面的HOME、速度、起飛、懸停命令）
        insert_index = min(insert_after_waypoint + 4, len(commands))
        
        # 創建LOITER命令
        loiter_cmd = QGCCommand(
            sequence=insert_index,
            frame=3,
            command=self.CMD_NAV_LOITER_TIME,
            param1=wait_time,
            x=0, y=0, z=0
        )
        
        # 插入LOITER命令並重新編號後續命令
        commands.insert(insert_index, loiter_cmd)
        self._renumber_sequences(commands)
        
        # 添加RTL命令
        rtl_cmd = QGCCommand(
            sequence=len(commands),
            frame=3,
            command=self.CMD_NAV_RETURN_TO_LAUNCH,
            x=0, y=0, z=0
        )
        commands.append(rtl_cmd)
        
        logger.info(f"生成避讓任務 {drone_id}: 在航點 {insert_after_waypoint} 後等待 {wait_time:.1f}秒")
        return commands
    
    def _increment_sequence(self):
        """增加序列計數器"""
        self.sequence_counter += 1
    
    def _renumber_sequences(self, commands: List[QGCCommand]):
        """重新編號命令序列"""
        for i, cmd in enumerate(commands):
            cmd.sequence = i

class MissionFileWriter:
    """任務檔案寫入器"""
    
    def __init__(self):
        self.generator = QGCMissionGenerator()
    
    def write_qgc_file(self, file_path: Union[str, Path], commands: List[QGCCommand]) -> bool:
        """寫入QGC waypoint檔案"""
        file_path = Path(file_path)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("QGC WPL 110\n")
                
                for cmd in commands:
                    f.write(cmd.to_qgc_line() + "\n")
            
            logger.info(f"成功寫入QGC檔案: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"寫入QGC檔案失敗: {file_path.name}, 錯誤: {e}")
            return False
    
    def write_mission_set(self, output_dir: Union[str, Path], 
                         missions: Dict[str, List[Dict]], 
                         collision_info: Dict[str, Dict] = None) -> Dict[str, bool]:
        """批次寫入任務集合"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for drone_id, waypoints in missions.items():
            try:
                # 選擇任務類型
                if collision_info and drone_id in collision_info:
                    commands = self.generator.create_collision_avoidance_mission(
                        drone_id, waypoints, collision_info[drone_id]
                    )
                    suffix = "_avoidance"
                else:
                    commands = self.generator.create_standard_mission(drone_id, waypoints)
                    suffix = "_standard"
                
                # 生成檔案名
                filename = f"{drone_id}_{timestamp}{suffix}.waypoints"
                file_path = output_dir / filename
                
                # 寫入檔案
                success = self.write_qgc_file(file_path, commands)
                results[drone_id] = success
                
            except Exception as e:
                logger.error(f"處理 {drone_id} 任務失敗: {e}")
                results[drone_id] = False
        
        return results

class TestMissionCreator:
    """測試任務創建器"""
    
    def __init__(self, origin_lat: float = 24.0, origin_lon: float = 121.0):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
    def create_2x2_formation_missions(self, formation_spacing: float = 100.0) -> Dict[str, List[Dict]]:
        """創建2x2編隊測試任務"""
        missions = {}
        
        # 編隊偏移（米制單位轉換為度）
        lat_offset = formation_spacing / 111111.0
        lon_offset = formation_spacing / (111111.0 * np.cos(np.radians(self.origin_lat)))
        
        # 四個區域的偏移
        region_offsets = [
            (-lat_offset, -lon_offset),  # 西南 - Drone_1
            (lat_offset, -lon_offset),   # 東南 - Drone_2
            (-lat_offset, lon_offset),   # 西北 - Drone_3  
            (lat_offset, lon_offset)     # 東北 - Drone_4
        ]
        
        for i in range(4):
            drone_id = f"Drone_{i+1}"
            lat_offset, lon_offset = region_offsets[i]
            
            # 起飛點（統一在原點東側50米）
            takeoff_lat = self.origin_lat
            takeoff_lon = self.origin_lon + (50.0 / (111111.0 * np.cos(np.radians(self.origin_lat))))
            
            waypoints = [
                # HOME點
                {
                    'lat': takeoff_lat,
                    'lon': takeoff_lon,
                    'alt': 0,
                    'cmd': 179
                }
            ]
            
            # 任務區域（矩形飛行）
            mission_lat = self.origin_lat + lat_offset
            mission_lon = self.origin_lon + lon_offset
            
            # 矩形航點
            rectangle_points = [
                (mission_lat, mission_lon, 15),
                (mission_lat + lat_offset * 0.3, mission_lon, 15),
                (mission_lat + lat_offset * 0.3, mission_lon + lon_offset * 0.3, 15),
                (mission_lat, mission_lon + lon_offset * 0.3, 15),
                (mission_lat, mission_lon, 15)  # 回到起點
            ]
            
            for lat, lon, alt in rectangle_points:
                waypoints.append({
                    'lat': lat,
                    'lon': lon,
                    'alt': alt,
                    'cmd': 16
                })
            
            missions[drone_id] = waypoints
        
        logger.info(f"創建2x2編隊測試任務: {len(missions)} 架無人機")
        return missions
    
    def create_race_track_mission(self, drone_id: str, track_size: float = 200.0) -> List[Dict]:
        """創建賽道測試任務"""
        # 橢圓形賽道
        center_lat = self.origin_lat
        center_lon = self.origin_lon
        
        # 轉換為度
        lat_radius = track_size / (2 * 111111.0)
        lon_radius = track_size / (2 * 111111.0 * np.cos(np.radians(center_lat)))
        
        waypoints = [
            # HOME點
            {
                'lat': center_lat,
                'lon': center_lon,
                'alt': 0,
                'cmd': 179
            }
        ]
        
        # 生成橢圓航點
        num_points = 12
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            lat = center_lat + lat_radius * np.sin(angle)
            lon = center_lon + lon_radius * np.cos(angle)
            
            waypoints.append({
                'lat': lat,
                'lon': lon,
                'alt': 20.0,
                'cmd': 16
            })
        
        logger.info(f"創建賽道任務 {drone_id}: {len(waypoints)} 個航點")
        return waypoints

class CollisionAnalyzer:
    """碰撞分析器 - 分析任務衝突並生成避讓策略"""
    
    def __init__(self, safety_distance: float = 5.0):
        self.safety_distance = safety_distance
        
    def analyze_missions(self, missions: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """分析任務集合的碰撞風險"""
        collision_info = {}
        
        # 簡化的碰撞分析：檢查航點距離
        drone_ids = list(missions.keys())
        
        for i in range(len(drone_ids)):
            for j in range(i + 1, len(drone_ids)):
                drone1_id = drone_ids[i]
                drone2_id = drone_ids[j]
                
                waypoints1 = missions[drone1_id]
                waypoints2 = missions[drone2_id]
                
                # 檢查每對航點的距離
                conflicts = self._find_waypoint_conflicts(waypoints1, waypoints2)
                
                if conflicts:
                    # 數字小的無人機有優先權
                    priority_drone = drone1_id if int(drone1_id.split('_')[1]) < int(drone2_id.split('_')[1]) else drone2_id
                    waiting_drone = drone2_id if priority_drone == drone1_id else drone1_id
                    
                    # 選擇第一個衝突點作為避讓點
                    first_conflict = conflicts[0]
                    wait_time = 10.0  # 默認等待時間
                    
                    collision_info[waiting_drone] = {
                        'insert_after_waypoint': first_conflict['waypoint_index'],
                        'wait_time': wait_time,
                        'conflict_with': priority_drone,
                        'conflict_distance': first_conflict['distance']
                    }
                    
                    logger.info(f"發現衝突: {drone1_id} vs {drone2_id}, {waiting_drone} 將等待 {wait_time}秒")
        
        return collision_info
    
    def _find_waypoint_conflicts(self, waypoints1: List[Dict], waypoints2: List[Dict]) -> List[Dict]:
        """找出航點衝突"""
        conflicts = []
        
        # 跳過HOME點，從任務航點開始檢查
        mission_wp1 = waypoints1[1:] if len(waypoints1) > 1 else []
        mission_wp2 = waypoints2[1:] if len(waypoints2) > 1 else []
        
        for i, wp1 in enumerate(mission_wp1):
            for j, wp2 in enumerate(mission_wp2):
                distance = self._calculate_distance(wp1, wp2)
                
                if distance < self.safety_distance * 2:  # 使用更大的安全邊界
                    conflicts.append({
                        'waypoint_index': i + 1,  # +1 因為跳過了HOME點
                        'distance': distance,
                        'waypoint1': wp1,
                        'waypoint2': wp2
                    })
        
        return conflicts
    
    def _calculate_distance(self, wp1: Dict, wp2: Dict) -> float:
        """計算兩個航點間的距離（簡化版）"""
        # 使用Haversine公式
        R = 6371000  # 地球半徑（米）
        
        lat1, lon1 = np.radians(wp1['lat']), np.radians(wp1['lon'])
        lat2, lon2 = np.radians(wp2['lat']), np.radians(wp2['lon'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        distance = R * c
        
        # 加上高度差
        alt1 = wp1.get('alt', 0)
        alt2 = wp2.get('alt', 0)
        distance_3d = np.sqrt(distance**2 + (alt2 - alt1)**2)
        
        return distance_3d

# 便利函數
def generate_test_missions(origin_lat: float = 24.0, origin_lon: float = 121.0) -> Dict[str, List[Dict]]:
    """快速生成測試任務"""
    creator = TestMissionCreator(origin_lat, origin_lon)
    return creator.create_2x2_formation_missions()

def export_missions_with_collision_avoidance(missions: Dict[str, List[Dict]], 
                                           output_dir: str,
                                           safety_distance: float = 5.0) -> Dict[str, bool]:
    """導出帶碰撞避讓的任務檔案"""
    # 分析碰撞
    analyzer = CollisionAnalyzer(safety_distance)
    collision_info = analyzer.analyze_missions(missions)
    
    # 寫入檔案
    writer = MissionFileWriter()
    results = writer.write_mission_set(output_dir, missions, collision_info)
    
    return results

def quick_generate_qgc_file(drone_id: str, waypoints: List[Dict], output_path: str) -> bool:
    """快速生成單個QGC檔案"""
    generator = QGCMissionGenerator()
    commands = generator.create_standard_mission(drone_id, waypoints)
    
    writer = MissionFileWriter()
    return writer.write_qgc_file(output_path, commands)