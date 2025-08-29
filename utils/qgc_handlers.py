"""
QGroundControl (QGC) 檔案處理模組
提供QGC waypoint檔案的導入、生成和修改功能
支援精確的LOITER插入和任務修正
"""

import logging
import os
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class Waypoint:
    """航點數據結構"""
    sequence: int
    current: int
    coordinate_frame: int
    command: int
    param1: float
    param2: float
    param3: float
    param4: float
    latitude: float
    longitude: float
    altitude: float
    autocontinue: int
    
    def __post_init__(self):
        """驗證航點數據"""
        if not (-90 <= self.latitude <= 90):
            logger.warning(f"緯度超出有效範圍: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            logger.warning(f"經度超出有效範圍: {self.longitude}")

@dataclass
class MissionModification:
    """任務修改配置"""
    drone_id: str
    insert_after_waypoint: int
    loiter_time: float
    conflict_with: str
    original_waypoint_count: int


class QGCWaypointParser:
    """
    QGC航點檔案解析器
    支援標準QGC WPL 110格式的讀取和寫入
    """
    
    # QGC命令類型映射
    COMMAND_TYPES = {
        16: "NAV_WAYPOINT",           # 普通航點
        17: "NAV_LOITER_UNLIM",       # 無限循環
        18: "NAV_LOITER_TURNS",       # 指定圈數循環
        19: "NAV_LOITER_TIME",        # 指定時間循環
        20: "NAV_RETURN_TO_LAUNCH",   # 返航
        21: "NAV_LAND",               # 降落
        22: "NAV_TAKEOFF",            # 起飛
        178: "DO_CHANGE_SPEED",       # 改變速度
        179: "HOME",                  # HOME點
        183: "DO_SET_SERVO",          # 設置伺服器
        252: "DO_SET_CAM_TRIGG_DIST", # 相機觸發距離
    }
    
    def __init__(self):
        """初始化解析器"""
        self.sequence_counter = 0
        
    def parse_waypoint_file(self, file_path: str) -> List[Waypoint]:
        """
        解析QGC waypoint檔案
        
        Args:
            file_path: QGC檔案路徑
            
        Returns:
            航點列表
            
        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 檔案格式錯誤
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"QGC檔案不存在: {file_path}")
        
        waypoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                
            # 檢查檔案格式
            if not lines or not lines[0].strip().startswith("QGC WPL"):
                raise ValueError("無效的QGC檔案格式")
            
            logger.info(f"開始解析QGC檔案: {file_path}")
            
            # 解析航點數據（跳過標題行）
            for line_num, line in enumerate(lines[1:], start=2):
                line = line.strip()
                
                # 跳過空行和註釋
                if not line or line.startswith('#'):
                    continue
                
                try:
                    waypoint = self._parse_waypoint_line(line)
                    if waypoint:
                        waypoints.append(waypoint)
                        
                except Exception as e:
                    logger.warning(f"解析第{line_num}行失敗: {e}")
                    continue
            
            # 過濾有效航點（只保留導航命令）
            nav_waypoints = [wp for wp in waypoints if wp.command in [16, 22]]  # NAV_WAYPOINT, NAV_TAKEOFF
            
            logger.info(f"成功解析 {len(nav_waypoints)} 個導航航點")
            return nav_waypoints
            
        except Exception as e:
            logger.error(f"解析QGC檔案失敗: {e}")
            raise e
    
    def _parse_waypoint_line(self, line: str) -> Optional[Waypoint]:
        """
        解析單個航點行
        
        Args:
            line: 航點數據行
            
        Returns:
            航點對象或None
        """
        parts = line.split('\t')
        
        if len(parts) < 12:
            logger.warning(f"航點數據不完整: {line}")
            return None
        
        try:
            waypoint = Waypoint(
                sequence=int(parts[0]),
                current=int(parts[1]),
                coordinate_frame=int(parts[2]),
                command=int(parts[3]),
                param1=float(parts[4]),
                param2=float(parts[5]),
                param3=float(parts[6]),
                param4=float(parts[7]),
                latitude=float(parts[8]),
                longitude=float(parts[9]),
                altitude=float(parts[10]),
                autocontinue=int(parts[11])
            )
            
            # 驗證關鍵數據
            if waypoint.latitude == 0 and waypoint.longitude == 0 and waypoint.command != 179:
                logger.warning("發現零坐標航點（非HOME點）")
                return None
            
            return waypoint
            
        except (ValueError, IndexError) as e:
            logger.warning(f"解析航點數據失敗: {e}")
            return None


class QGCWaypointGenerator:
    """
    QGC航點檔案生成器
    支援創建完整的任務檔案，包括精確的LOITER插入
    """
    
    def __init__(self):
        """初始化生成器"""
        self.sequence_counter = 0
        
    def generate_complete_mission(self, drone_id: str, waypoints: List[Dict], 
                                 modifications: Optional[List[MissionModification]] = None) -> List[str]:
        """
        生成完整任務檔案
        
        Args:
            drone_id: 無人機ID
            waypoints: 航點列表 [{'lat': float, 'lon': float, 'alt': float, 'cmd': int}, ...]
            modifications: 任務修改列表（LOITER插入等）
            
        Returns:
            QGC檔案行列表
        """
        lines = ["QGC WPL 110"]
        self.sequence_counter = 0
        
        if not waypoints:
            logger.warning(f"{drone_id}: 沒有航點數據")
            return lines
        
        logger.info(f"為 {drone_id} 生成任務檔案，包含 {len(waypoints)} 個航點")
        
        # HOME點
        home_wp = waypoints[0]
        lines.append(self._create_home_waypoint(home_wp))
        
        # 速度設置（8 m/s）
        lines.append(self._create_speed_command(8.0))
        
        # 起飛命令
        lines.append(self._create_takeoff_command(home_wp, altitude=10.0))
        
        # 起飛後懸停等待
        lines.append(self._create_loiter_time_command(2.0))
        
        # 處理任務航點並插入LOITER
        loiter_insertions = self._process_modifications(modifications)
        
        # 添加任務航點（從waypoints[1:]開始，因為waypoints[0]是HOME）
        for wp_index, wp in enumerate(waypoints[1:], start=2):
            # 添加航點
            lines.append(self._create_navigation_waypoint(wp))
            
            # 檢查是否需要在此航點後插入LOITER
            if wp_index in loiter_insertions:
                loiter_time = loiter_insertions[wp_index]
                lines.append(self._create_loiter_time_command(loiter_time))
                logger.info(f"{drone_id}: 在航點 {wp_index} 後插入 {loiter_time:.1f}s LOITER")
        
        # RTL (Return to Launch)
        lines.append(self._create_rtl_command())
        
        return lines
    
    def _process_modifications(self, modifications: Optional[List[MissionModification]]) -> Dict[int, float]:
        """
        處理任務修改，生成LOITER插入映射
        
        Args:
            modifications: 修改列表
            
        Returns:
            航點索引到等待時間的映射
        """
        loiter_insertions = {}
        
        if modifications:
            for mod in modifications:
                wp_index = mod.insert_after_waypoint
                loiter_time = mod.loiter_time
                
                # 累加同一航點的等待時間
                if wp_index in loiter_insertions:
                    loiter_insertions[wp_index] += loiter_time
                else:
                    loiter_insertions[wp_index] = loiter_time
                    
                logger.info(f"修改處理: 航點 {wp_index} 後等待 {loiter_time:.1f}s "
                           f"(避讓 {mod.conflict_with})")
        
        return loiter_insertions
    
    def _create_home_waypoint(self, waypoint: Dict) -> str:
        """創建HOME航點"""
        return (f"{self.sequence_counter}\t1\t0\t179\t0\t0\t0\t0\t"
                f"{waypoint['lat']:.8f}\t{waypoint['lon']:.8f}\t"
                f"{waypoint.get('alt', 0):.2f}\t1")
    
    def _create_speed_command(self, speed: float) -> str:
        """創建速度設置命令"""
        self.sequence_counter += 1
        return f"{self.sequence_counter}\t0\t3\t178\t0\t{speed:.1f}\t0\t0\t0\t0\t0\t1"
    
    def _create_takeoff_command(self, waypoint: Dict, altitude: float = 10.0) -> str:
        """創建起飛命令"""
        self.sequence_counter += 1
        return (f"{self.sequence_counter}\t0\t3\t22\t0\t0\t0\t0\t"
                f"{waypoint['lat']:.8f}\t{waypoint['lon']:.8f}\t{altitude:.2f}\t1")
    
    def _create_loiter_time_command(self, time_seconds: float) -> str:
        """創建定時等待命令"""
        self.sequence_counter += 1
        return f"{self.sequence_counter}\t0\t3\t19\t{time_seconds:.1f}\t0\t0\t0\t0\t0\t0\t1"
    
    def _create_navigation_waypoint(self, waypoint: Dict) -> str:
        """創建導航航點"""
        self.sequence_counter += 1
        return (f"{self.sequence_counter}\t0\t3\t16\t0\t0\t0\t0\t"
                f"{waypoint['lat']:.8f}\t{waypoint['lon']:.8f}\t"
                f"{waypoint.get('alt', 15):.2f}\t1")
    
    def _create_rtl_command(self) -> str:
        """創建返航命令"""
        self.sequence_counter += 1
        return f"{self.sequence_counter}\t0\t3\t20\t0\t0\t0\t0\t0\t0\t0\t1"
    
    def generate_mission_with_conflicts(self, drone_id: str, waypoints: List[Dict], 
                                      conflicts: List[Dict]) -> List[str]:
        """
        根據衝突分析生成修正後的任務檔案
        
        Args:
            drone_id: 無人機ID
            waypoints: 原始航點列表
            conflicts: 衝突分析結果
            
        Returns:
            修正後的QGC檔案行列表
        """
        # 找出影響此無人機的衝突
        relevant_conflicts = [c for c in conflicts if c.get('waiting_drone') == drone_id]
        
        if not relevant_conflicts:
            # 沒有衝突，生成標準任務
            return self.generate_complete_mission(drone_id, waypoints)
        
        # 選擇最早發生的衝突進行處理
        earliest_conflict = min(relevant_conflicts, key=lambda c: c.get('conflict_time', 0))
        
        # 創建修改配置
        modification = MissionModification(
            drone_id=drone_id,
            insert_after_waypoint=earliest_conflict.get('waypoint2_index', 2),
            loiter_time=earliest_conflict.get('recommended_wait_time', 5.0),
            conflict_with=earliest_conflict.get('priority_drone', 'Unknown'),
            original_waypoint_count=len(waypoints)
        )
        
        logger.info(f"為 {drone_id} 生成帶避讓的任務: 在航點 {modification.insert_after_waypoint} "
                   f"後等待 {modification.loiter_time:.1f}s (避讓 {modification.conflict_with})")
        
        return self.generate_complete_mission(drone_id, waypoints, [modification])


class CSVWaypointHandler:
    """
    CSV格式航點檔案處理器
    支援多種CSV格式的航點數據導入
    """
    
    # 常見的列名映射
    COLUMN_MAPPINGS = {
        'latitude': ['lat', 'latitude', 'Lat', 'Latitude', 'LAT'],
        'longitude': ['lon', 'longitude', 'Lon', 'Longitude', 'LON', 'lng'],
        'altitude': ['alt', 'altitude', 'Alt', 'Altitude', 'ALT', 'z', 'height']
    }
    
    def __init__(self):
        """初始化CSV處理器"""
        pass
    
    def parse_csv_file(self, file_path: str) -> List[Dict]:
        """
        解析CSV航點檔案
        
        Args:
            file_path: CSV檔案路徑
            
        Returns:
            航點字典列表
            
        Raises:
            FileNotFoundError: 檔案不存在
            ValueError: 檔案格式錯誤
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV檔案不存在: {file_path}")
        
        try:
            # 嘗試不同的編碼
            encodings = ['utf-8', 'gbk', 'big5', 'latin1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.info(f"成功以 {encoding} 編碼讀取CSV檔案")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("無法識別CSV檔案編碼")
            
            # 標準化列名
            df = self._standardize_columns(df)
            
            # 驗證必要列
            required_columns = ['latitude', 'longitude', 'altitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"CSV檔案缺少必要列: {missing_columns}")
            
            # 轉換為航點列表
            waypoints = []
            for index, row in df.iterrows():
                try:
                    waypoint = {
                        'lat': float(row['latitude']),
                        'lon': float(row['longitude']),
                        'alt': float(row['altitude']),
                        'cmd': 16  # 預設為NAV_WAYPOINT
                    }
                    
                    # 驗證坐標
                    if self._validate_coordinates(waypoint['lat'], waypoint['lon']):
                        waypoints.append(waypoint)
                    else:
                        logger.warning(f"第{index+1}行坐標無效，已跳過")
                        
                except (ValueError, KeyError) as e:
                    logger.warning(f"處理第{index+1}行數據失敗: {e}")
                    continue
            
            logger.info(f"成功從CSV檔案解析 {len(waypoints)} 個航點")
            return waypoints
            
        except Exception as e:
            logger.error(f"解析CSV檔案失敗: {e}")
            raise e
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        標準化CSV列名
        
        Args:
            df: 原始DataFrame
            
        Returns:
            標準化後的DataFrame
        """
        df_copy = df.copy()
        
        for standard_name, possible_names in self.COLUMN_MAPPINGS.items():
            for possible_name in possible_names:
                if possible_name in df_copy.columns:
                    df_copy = df_copy.rename(columns={possible_name: standard_name})
                    break
        
        return df_copy
    
    def _validate_coordinates(self, lat: float, lon: float) -> bool:
        """
        驗證地理坐標
        
        Args:
            lat: 緯度
            lon: 經度
            
        Returns:
            坐標是否有效
        """
        return (-90 <= lat <= 90) and (-180 <= lon <= 180) and (lat != 0 or lon != 0)


class MissionFileExporter:
    """
    任務檔案導出器
    支援批量導出修改後的任務檔案
    """
    
    def __init__(self):
        """初始化導出器"""
        self.generator = QGCWaypointGenerator()
    
    def export_modified_missions(self, modified_missions: Dict[str, List[str]], 
                               export_directory: str, 
                               timestamp_suffix: bool = True) -> Dict[str, str]:
        """
        批量導出修改後的任務檔案
        
        Args:
            modified_missions: 修改後的任務字典 {drone_id: mission_lines}
            export_directory: 導出目錄
            timestamp_suffix: 是否添加時間戳後綴
            
        Returns:
            導出結果字典 {drone_id: file_path}
            
        Raises:
            OSError: 目錄創建或檔案寫入失敗
        """
        if not modified_missions:
            logger.warning("沒有修改後的任務需要導出")
            return {}
        
        # 確保導出目錄存在
        os.makedirs(export_directory, exist_ok=True)
        
        export_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp_suffix else ""
        
        for drone_id, mission_lines in modified_missions.items():
            try:
                # 生成檔案名
                if timestamp_suffix:
                    filename = f"{drone_id}_modified_{timestamp}.waypoints"
                else:
                    filename = f"{drone_id}_modified.waypoints"
                
                file_path = os.path.join(export_directory, filename)
                
                # 寫入檔案
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(mission_lines))
                
                export_results[drone_id] = file_path
                logger.info(f"成功導出 {drone_id} 修改後任務: {file_path}")
                
            except Exception as e:
                logger.error(f"導出 {drone_id} 任務檔案失敗: {e}")
                continue
        
        logger.info(f"批量導出完成: {len(export_results)}/{len(modified_missions)} 個檔案成功")
        return export_results
    
    def create_mission_summary(self, missions: Dict[str, List[str]], 
                             output_path: str) -> bool:
        """
        創建任務摘要報告
        
        Args:
            missions: 任務字典
            output_path: 輸出路徑
            
        Returns:
            是否成功創建
        """
        try:
            summary_lines = []
            summary_lines.append("任務摘要報告")
            summary_lines.append("=" * 50)
            summary_lines.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            summary_lines.append(f"任務數量: {len(missions)}")
            summary_lines.append("")
            
            for drone_id, mission_lines in missions.items():
                waypoint_count = len([line for line in mission_lines if '\t16\t' in line])  # NAV_WAYPOINT
                loiter_count = len([line for line in mission_lines if '\t19\t' in line])    # LOITER_TIME
                
                summary_lines.append(f"{drone_id}:")
                summary_lines.append(f"  - 總命令數: {len(mission_lines) - 1}")  # 減去標題行
                summary_lines.append(f"  - 導航航點: {waypoint_count}")
                summary_lines.append(f"  - 等待命令: {loiter_count}")
                summary_lines.append("")
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_lines))
            
            logger.info(f"任務摘要已創建: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"創建任務摘要失敗: {e}")
            return False


# 便利函數
def parse_mission_file(file_path: str) -> List[Dict]:
    """
    自動檢測並解析任務檔案（QGC或CSV）
    
    Args:
        file_path: 檔案路徑
        
    Returns:
        航點字典列表
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        handler = CSVWaypointHandler()
        return handler.parse_csv_file(file_path)
    else:
        # 假設是QGC格式
        parser = QGCWaypointParser()
        waypoints = parser.parse_waypoint_file(file_path)
        
        # 轉換為字典格式
        return [{
            'lat': wp.latitude,
            'lon': wp.longitude,
            'alt': wp.altitude,
            'cmd': wp.command
        } for wp in waypoints]