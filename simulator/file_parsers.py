#!/usr/bin/env python3
"""
任務檔案解析器
支援QGC waypoint檔案、CSV軌跡檔案和其他格式的解析
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Waypoint:
    """航點數據結構"""
    lat: float
    lon: float
    alt: float
    cmd: int = 16  # MAVLink命令類型
    param1: float = 0.0
    param2: float = 0.0
    param3: float = 0.0
    param4: float = 0.0
    autocontinue: int = 1
    frame: int = 3  # MAV_FRAME_GLOBAL_RELATIVE_ALT

@dataclass 
class MissionMetadata:
    """任務元數據"""
    name: str
    description: str = ""
    creation_time: str = ""
    drone_id: str = ""
    total_waypoints: int = 0
    estimated_duration: float = 0.0
    max_altitude: float = 0.0
    total_distance: float = 0.0

class QGCWaypointParser:
    """QGroundControl航點檔案解析器"""
    
    def __init__(self):
        self.supported_commands = {
            16: "NAV_WAYPOINT",           # 導航航點
            17: "NAV_LOITER_UNLIM",       # 無限盤旋
            18: "NAV_LOITER_TURNS",       # 指定圈數盤旋
            19: "NAV_LOITER_TIME",        # 指定時間盤旋
            20: "NAV_RETURN_TO_LAUNCH",   # 返回起飛點
            21: "NAV_LAND",               # 降落
            22: "NAV_TAKEOFF",            # 起飛
            179: "SET_HOME"               # 設置起飛點
        }
    
    def parse_file(self, file_path: Union[str, Path]) -> Tuple[List[Waypoint], MissionMetadata]:
        """解析QGC waypoint檔案"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"檔案不存在: {file_path}")
        
        waypoints = []
        metadata = MissionMetadata(name=file_path.stem)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 檢查檔案格式
            if not lines or not lines[0].strip().startswith("QGC WPL"):
                raise ValueError("不是有效的QGC waypoint檔案")
            
            # 解析每一行
            for line_num, line in enumerate(lines[1:], start=2):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                try:
                    waypoint = self._parse_qgc_line(line)
                    if waypoint:
                        waypoints.append(waypoint)
                except Exception as e:
                    logger.warning(f"跳過第 {line_num} 行，解析失敗: {e}")
            
            # 更新元數據
            metadata.total_waypoints = len(waypoints)
            if waypoints:
                metadata.max_altitude = max(wp.alt for wp in waypoints)
                metadata.total_distance = self._calculate_total_distance(waypoints)
                metadata.estimated_duration = self._estimate_mission_duration(waypoints)
            
            logger.info(f"成功解析QGC檔案: {file_path.name}, {len(waypoints)} 個航點")
            
        except Exception as e:
            logger.error(f"解析QGC檔案失敗: {file_path.name}, 錯誤: {e}")
            raise
        
        return waypoints, metadata
    
    def _parse_qgc_line(self, line: str) -> Optional[Waypoint]:
        """解析QGC檔案的單行數據"""
        parts = line.split('\t')
        
        # QGC格式至少需要12個欄位
        if len(parts) < 12:
            return None
        
        try:
            seq = int(parts[0])
            current = int(parts[1])
            frame = int(parts[2])
            cmd = int(parts[3])
            param1 = float(parts[4]) if parts[4] else 0.0
            param2 = float(parts[5]) if parts[5] else 0.0
            param3 = float(parts[6]) if parts[6] else 0.0
            param4 = float(parts[7]) if parts[7] else 0.0
            lat = float(parts[8])
            lon = float(parts[9])
            alt = float(parts[10])
            autocontinue = int(parts[11]) if len(parts) > 11 else 1
            
            # 過濾有效的航點命令
            if cmd in self.supported_commands and lat != 0 and lon != 0:
                return Waypoint(
                    lat=lat, lon=lon, alt=alt, cmd=cmd,
                    param1=param1, param2=param2, param3=param3, param4=param4,
                    autocontinue=autocontinue, frame=frame
                )
                
        except (ValueError, IndexError) as e:
            logger.debug(f"跳過無效行: {line[:50]}...")
            return None
        
        return None
    
    def _calculate_total_distance(self, waypoints: List[Waypoint]) -> float:
        """計算總飛行距離"""
        if len(waypoints) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(waypoints)):
            prev_wp = waypoints[i-1]
            curr_wp = waypoints[i]
            
            # 使用Haversine公式計算地球表面距離
            distance = self._haversine_distance(
                prev_wp.lat, prev_wp.lon, curr_wp.lat, curr_wp.lon
            )
            
            # 加上高度差
            height_diff = abs(curr_wp.alt - prev_wp.alt)
            total_distance += np.sqrt(distance**2 + height_diff**2)
        
        return total_distance
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """使用Haversine公式計算地球表面兩點間距離"""
        R = 6371000  # 地球半徑（米）
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = (np.sin(dlat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _estimate_mission_duration(self, waypoints: List[Waypoint], 
                                  cruise_speed: float = 8.0) -> float:
        """估算任務執行時間"""
        total_distance = self._calculate_total_distance(waypoints)
        
        # 簡單的時間估算：距離 / 速度 + 起飛降落時間
        flight_time = total_distance / cruise_speed
        overhead_time = 30.0  # 起飛、降落、懸停等開銷時間
        
        return flight_time + overhead_time

class CSVTrajectoryParser:
    """CSV軌跡檔案解析器"""
    
    def __init__(self):
        self.column_mappings = {
            # 標準列名映射
            'latitude': 'lat',
            'longitude': 'lon', 
            'altitude': 'alt',
            'lat': 'lat',
            'lon': 'lon',
            'lng': 'lon',
            'alt': 'alt',
            'height': 'alt',
            'elevation': 'alt',
            # 座標列名映射
            'x': 'x',
            'y': 'y', 
            'z': 'z',
            # 時間列名映射
            'time': 'time',
            'timestamp': 'time',
            'duration': 'time',
            # 其他屬性
            'speed': 'speed',
            'heading': 'heading',
            'yaw': 'yaw'
        }
    
    def parse_file(self, file_path: Union[str, Path], 
                  coordinate_system: str = 'gps') -> Tuple[List[Dict], MissionMetadata]:
        """
        解析CSV軌跡檔案
        
        Args:
            file_path: 檔案路徑
            coordinate_system: 座標系統類型 ('gps' 或 'local')
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"檔案不存在: {file_path}")
        
        try:
            # 嘗試不同的編碼格式
            df = self._read_csv_with_encoding(file_path)
            
            # 標準化列名
            df = self._normalize_column_names(df)
            
            # 驗證必要欄位
            trajectory_points = self._validate_and_convert(df, coordinate_system)
            
            # 創建元數據
            metadata = MissionMetadata(
                name=file_path.stem,
                total_waypoints=len(trajectory_points),
                max_altitude=max(point.get('alt', point.get('z', 0)) for point in trajectory_points),
                estimated_duration=trajectory_points[-1].get('time', 0) if trajectory_points else 0
            )
            
            logger.info(f"成功解析CSV檔案: {file_path.name}, {len(trajectory_points)} 個軌跡點")
            
            return trajectory_points, metadata
            
        except Exception as e:
            logger.error(f"解析CSV檔案失敗: {file_path.name}, 錯誤: {e}")
            raise
    
    def _read_csv_with_encoding(self, file_path: Path) -> pd.DataFrame:
        """嘗試多種編碼格式讀取CSV"""
        encodings = ['utf-8', 'utf-8-sig', 'big5', 'gbk', 'latin-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                logger.debug(f"成功使用 {encoding} 編碼讀取檔案")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                if encoding == encodings[-1]:  # 最後一個編碼也失敗
                    raise e
                continue
        
        raise ValueError("無法使用任何支援的編碼格式讀取檔案")
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """標準化列名"""
        # 轉換為小寫並移除空格
        df.columns = [col.lower().strip() for col in df.columns]
        
        # 應用列名映射
        rename_dict = {}
        for old_name in df.columns:
            if old_name in self.column_mappings:
                rename_dict[old_name] = self.column_mappings[old_name]
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.debug(f"列名映射: {rename_dict}")
        
        return df
    
    def _validate_and_convert(self, df: pd.DataFrame, 
                             coordinate_system: str) -> List[Dict]:
        """驗證並轉換數據"""
        trajectory_points = []
        
        if coordinate_system == 'gps':
            required_cols = ['lat', 'lon']
            alt_col = 'alt'
        else:  # local coordinate system
            required_cols = ['x', 'y']
            alt_col = 'z'
        
        # 檢查必要欄位
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"缺少必要欄位: {missing_cols}")
        
        # 如果沒有高度欄位，使用默認值
        if alt_col not in df.columns:
            df[alt_col] = 10.0  # 默認10米高度
            logger.warning(f"未找到高度欄位 '{alt_col}'，使用默認值 10.0m")
        
        # 如果沒有時間欄位，生成時間序列
        if 'time' not in df.columns:
            df['time'] = np.arange(len(df)) * 1.0  # 每秒一個點
            logger.warning("未找到時間欄位，生成默認時間序列")
        
        # 轉換數據
        for idx, row in df.iterrows():
            try:
                point = {
                    'time': float(row.get('time', idx)),
                }
                
                if coordinate_system == 'gps':
                    point.update({
                        'lat': float(row['lat']),
                        'lon': float(row['lon']),
                        'alt': float(row[alt_col])
                    })
                else:
                    point.update({
                        'x': float(row['x']),
                        'y': float(row['y']),
                        'z': float(row[alt_col])
                    })
                
                # 添加可選欄位
                optional_fields = ['speed', 'heading', 'yaw']
                for field in optional_fields:
                    if field in row and pd.notna(row[field]):
                        point[field] = float(row[field])
                
                trajectory_points.append(point)
                
            except (ValueError, KeyError) as e:
                logger.warning(f"跳過第 {idx+1} 行，數據無效: {e}")
                continue
        
        if not trajectory_points:
            raise ValueError("沒有有效的軌跡點數據")
        
        return trajectory_points

class KMLParser:
    """KML檔案解析器（Google Earth格式）"""
    
    def parse_file(self, file_path: Union[str, Path]) -> Tuple[List[Waypoint], MissionMetadata]:
        """解析KML檔案"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"檔案不存在: {file_path}")
        
        waypoints = []
        metadata = MissionMetadata(name=file_path.stem)
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # KML命名空間
            namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}
            
            # 查找所有Placemark
            placemarks = root.findall('.//kml:Placemark', namespaces)
            
            for placemark in placemarks:
                coord_elem = placemark.find('.//kml:coordinates', namespaces)
                if coord_elem is not None:
                    coords_text = coord_elem.text.strip()
                    
                    # 解析座標 (經度,緯度,高度)
                    for coord_line in coords_text.split('\n'):
                        coord_line = coord_line.strip()
                        if coord_line:
                            parts = coord_line.split(',')
                            if len(parts) >= 2:
                                lon = float(parts[0])
                                lat = float(parts[1])
                                alt = float(parts[2]) if len(parts) > 2 else 10.0
                                
                                waypoint = Waypoint(lat=lat, lon=lon, alt=alt)
                                waypoints.append(waypoint)
            
            metadata.total_waypoints = len(waypoints)
            logger.info(f"成功解析KML檔案: {file_path.name}, {len(waypoints)} 個航點")
            
        except Exception as e:
            logger.error(f"解析KML檔案失敗: {file_path.name}, 錯誤: {e}")
            raise
        
        return waypoints, metadata

class UniversalMissionParser:
    """通用任務檔案解析器"""
    
    def __init__(self):
        self.parsers = {
            '.waypoints': QGCWaypointParser(),
            '.txt': QGCWaypointParser(),  # 有些QGC檔案使用.txt
            '.csv': CSVTrajectoryParser(),
            '.kml': KMLParser()
        }
    
    def parse_file(self, file_path: Union[str, Path], 
                  **kwargs) -> Tuple[List[Union[Waypoint, Dict]], MissionMetadata]:
        """
        自動識別並解析任務檔案
        
        Args:
            file_path: 檔案路徑
            **kwargs: 傳遞給具體解析器的參數
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.parsers:
            raise ValueError(f"不支援的檔案格式: {file_ext}")
        
        parser = self.parsers[file_ext]
        
        try:
            if isinstance(parser, CSVTrajectoryParser):
                return parser.parse_file(file_path, **kwargs)
            else:
                return parser.parse_file(file_path)
        except Exception as e:
            logger.error(f"解析檔案失敗: {file_path.name}")
            raise

class BatchParser:
    """批次解析器 - 處理多個檔案"""
    
    def __init__(self):
        self.universal_parser = UniversalMissionParser()
    
    def parse_multiple_files(self, file_paths: List[Union[str, Path]], 
                           max_files: int = 4) -> Dict[str, Tuple[List, MissionMetadata]]:
        """批次解析多個檔案"""
        results = {}
        
        for i, file_path in enumerate(file_paths[:max_files]):
            try:
                drone_id = f"Drone_{i+1}"
                waypoints, metadata = self.universal_parser.parse_file(file_path)
                metadata.drone_id = drone_id
                
                results[drone_id] = (waypoints, metadata)
                logger.info(f"成功解析 {drone_id}: {Path(file_path).name}")
                
            except Exception as e:
                logger.error(f"解析檔案失敗: {Path(file_path).name}, {e}")
                continue
        
        return results
    
    def validate_mission_compatibility(self, missions: Dict[str, Tuple]) -> Dict[str, str]:
        """驗證任務相容性"""
        issues = {}
        
        if len(missions) < 2:
            return issues
        
        # 檢查時間長度差異
        durations = []
        for drone_id, (waypoints, metadata) in missions.items():
            durations.append((drone_id, metadata.estimated_duration))
        
        max_duration = max(durations, key=lambda x: x[1])
        min_duration = min(durations, key=lambda x: x[1])
        
        if max_duration[1] - min_duration[1] > 60:  # 超過1分鐘差異
            issues['time_mismatch'] = f"任務時間差異過大: {max_duration[0]}({max_duration[1]:.1f}s) vs {min_duration[0]}({min_duration[1]:.1f}s)"
        
        # 檢查高度差異
        altitudes = []
        for drone_id, (waypoints, metadata) in missions.items():
            altitudes.append((drone_id, metadata.max_altitude))
        
        max_alt = max(altitudes, key=lambda x: x[1])
        min_alt = min(altitudes, key=lambda x: x[1])
        
        if max_alt[1] - min_alt[1] > 50:  # 超過50米高度差
            issues['altitude_mismatch'] = f"飛行高度差異過大: {max_alt[0]}({max_alt[1]:.1f}m) vs {min_alt[0]}({min_alt[1]:.1f}m)"
        
        return issues

# 便利函數
def quick_parse_qgc(file_path: str) -> List[Dict]:
    """快速解析QGC檔案，返回簡化格式"""
    parser = QGCWaypointParser()
    waypoints, _ = parser.parse_file(file_path)
    
    return [
        {'lat': wp.lat, 'lon': wp.lon, 'alt': wp.alt, 'cmd': wp.cmd}
        for wp in waypoints
    ]

def quick_parse_csv(file_path: str, coordinate_system: str = 'gps') -> List[Dict]:
    """快速解析CSV檔案，返回簡化格式"""
    parser = CSVTrajectoryParser()
    trajectory, _ = parser.parse_file(file_path, coordinate_system=coordinate_system)
    return trajectory

def auto_parse_mission_files(file_paths: List[str], max_drones: int = 4) -> Dict[str, List[Dict]]:
    """自動解析任務檔案，返回標準化格式"""
    batch_parser = BatchParser()
    results = batch_parser.parse_multiple_files(file_paths, max_drones)
    
    # 轉換為標準化格式
    missions = {}
    for drone_id, (waypoints, metadata) in results.items():
        if isinstance(waypoints[0], Waypoint):
            # QGC格式轉換
            missions[drone_id] = [
                {'lat': wp.lat, 'lon': wp.lon, 'alt': wp.alt, 'cmd': wp.cmd}
                for wp in waypoints
            ]
        else:
            # CSV格式已經是Dict
            missions[drone_id] = waypoints
    
    return missions