#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Parser Module
Handles QGC waypoint and CSV file parsing for drone missions
"""

import logging
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class QGCFileParser:
    """
    QGC waypoint file parser with enhanced error handling
    Supports QGroundControl .waypoints format
    """
    
    def __init__(self):
        self.valid_commands = {
            16: "NAV_WAYPOINT",
            22: "NAV_TAKEOFF", 
            179: "NAV_SET_HOME",
            178: "DO_CHANGE_SPEED",
            19: "NAV_LOITER_TIME",
            20: "NAV_RETURN_TO_LAUNCH"
        }
        
    def parse_file(self, file_path: str) -> List[Dict]:
        """
        Parse QGC waypoint file
        
        Args:
            file_path: Path to QGC waypoint file
            
        Returns:
            List of waypoint dictionaries
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"QGC file not found: {file_path}")
        
        logger.info(f"Parsing QGC file: {file_path}")
        
        waypoints = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Validate header
            if not lines or not lines[0].strip().startswith('QGC WPL'):
                raise ValueError("Invalid QGC file format: missing or invalid header")
            
            # Parse waypoints
            for line_num, line in enumerate(lines[1:], start=2):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                try:
                    waypoint = self._parse_qgc_line(line, line_num)
                    if waypoint:
                        waypoints.append(waypoint)
                        
                except Exception as e:
                    logger.warning(f"Skipping invalid line {line_num} in {file_path}: {e}")
                    continue
            
            if not waypoints:
                raise ValueError(f"No valid waypoints found in {file_path}")
            
            logger.info(f"Successfully parsed {len(waypoints)} waypoints from {file_path}")
            
            # Validate waypoint sequence
            self._validate_waypoint_sequence(waypoints)
            
            return waypoints
            
        except Exception as e:
            logger.error(f"Failed to parse QGC file {file_path}: {e}")
            raise
    
    def _parse_qgc_line(self, line: str, line_num: int) -> Optional[Dict]:
        """
        Parse a single line from QGC waypoint file
        
        Args:
            line: Line content
            line_num: Line number for error reporting
            
        Returns:
            Waypoint dictionary or None if line should be skipped
        """
        parts = line.split('\t')
        
        if len(parts) < 12:
            raise ValueError(f"Insufficient columns ({len(parts)}, expected ≥12)")
        
        try:
            seq = int(parts[0])
            current_wp = int(parts[1])
            coord_frame = int(parts[2])
            cmd = int(parts[3])
            param1 = float(parts[4])
            param2 = float(parts[5])
            param3 = float(parts[6])
            param4 = float(parts[7])
            lat = float(parts[8])
            lon = float(parts[9])
            alt = float(parts[10])
            autocontinue = int(parts[11])
            
            # Only process navigation waypoints and takeoff commands
            if cmd in [16, 22]:  # NAV_WAYPOINT, NAV_TAKEOFF
                if lat == 0 and lon == 0:
                    logger.debug(f"Skipping waypoint with zero coordinates at line {line_num}")
                    return None
                
                waypoint = {
                    'seq': seq,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt,
                    'cmd': cmd,
                    'param1': param1,
                    'param2': param2,
                    'param3': param3,
                    'param4': param4,
                    'coord_frame': coord_frame,
                    'autocontinue': autocontinue
                }
                
                logger.debug(f"Parsed waypoint {seq}: {self.valid_commands.get(cmd, f'CMD_{cmd}')} "
                           f"at ({lat:.6f}, {lon:.6f}, {alt:.1f}m)")
                
                return waypoint
            
            else:
                logger.debug(f"Skipping non-navigation command {cmd} ({self.valid_commands.get(cmd, 'UNKNOWN')}) "
                           f"at line {line_num}")
                return None
                
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid data format: {e}")
    
    def _validate_waypoint_sequence(self, waypoints: List[Dict]) -> None:
        """
        Validate waypoint sequence and coordinates
        
        Args:
            waypoints: List of waypoint dictionaries
        """
        if not waypoints:
            return
        
        # Check for HOME waypoint
        has_home = any(wp.get('cmd') == 179 or wp.get('seq') == 0 for wp in waypoints)
        if not has_home:
            logger.warning("No HOME waypoint found in mission")
        
        # Validate coordinate ranges
        for i, wp in enumerate(waypoints):
            lat, lon, alt = wp['lat'], wp['lon'], wp['alt']
            
            if not (-90 <= lat <= 90):
                logger.warning(f"Waypoint {i}: Invalid latitude {lat}")
            
            if not (-180 <= lon <= 180):
                logger.warning(f"Waypoint {i}: Invalid longitude {lon}")
            
            if alt < -1000 or alt > 10000:
                logger.warning(f"Waypoint {i}: Unusual altitude {alt}m")
        
        logger.debug(f"Waypoint sequence validation completed for {len(waypoints)} waypoints")


class CSVFileParser:
    """
    CSV file parser with flexible column mapping
    Supports various CSV formats for waypoint data
    """
    
    def __init__(self):
        self.column_mappings = {
            # Standard mappings
            'latitude': 'lat',
            'longitude': 'lon', 
            'altitude': 'alt',
            
            # Alternative names
            'lat': 'lat',
            'lon': 'lon',
            'lng': 'lon',
            'alt': 'alt',
            'height': 'alt',
            'elevation': 'alt',
            
            # Coordinate system variants
            'x': 'lon',  # Assuming X is longitude
            'y': 'lat',  # Assuming Y is latitude
            'z': 'alt',  # Assuming Z is altitude
            
            # Capitalized variants
            'Latitude': 'lat',
            'Longitude': 'lon',
            'Altitude': 'alt',
            'Lat': 'lat',
            'Lon': 'lon',
            'Alt': 'alt',
            'LAT': 'lat',
            'LON': 'lon',
            'ALT': 'alt',
            
            # WGS84 variants
            'wgs84_lat': 'lat',
            'wgs84_lon': 'lon',
            'wgs84_alt': 'alt'
        }
    
    def parse_file(self, file_path: str) -> List[Dict]:
        """
        Parse CSV waypoint file with automatic column detection
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            List of waypoint dictionaries
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        logger.info(f"Parsing CSV file: {file_path}")
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    logger.debug(f"Successfully read CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any standard encoding")
            
            logger.info(f"CSV file loaded: {len(df)} rows, {len(df.columns)} columns")
            logger.debug(f"Original columns: {list(df.columns)}")
            
            # Clean column names (remove whitespace)
            df.columns = df.columns.str.strip()
            
            # Apply column mappings
            df = self._apply_column_mappings(df)
            
            # Validate required columns
            self._validate_required_columns(df)
            
            # Convert to waypoint list
            waypoints = self._dataframe_to_waypoints(df)
            
            # Validate waypoint data
            self._validate_waypoint_data(waypoints)
            
            logger.info(f"Successfully parsed {len(waypoints)} waypoints from CSV")
            
            return waypoints
            
        except Exception as e:
            logger.error(f"Failed to parse CSV file {file_path}: {e}")
            raise
    
    def _apply_column_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply column name mappings to standardize column names
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized column names
        """
        # Create mapping for existing columns
        rename_dict = {}
        
        for col in df.columns:
            if col in self.column_mappings:
                new_name = self.column_mappings[col]
                rename_dict[col] = new_name
                logger.debug(f"Mapping column '{col}' -> '{new_name}'")
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.debug(f"Applied column mappings: {rename_dict}")
        
        return df
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns are present
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['lat', 'lon', 'alt']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            available_cols = list(df.columns)
            raise ValueError(
                f"Missing required columns: {missing_columns}. "
                f"Available columns: {available_cols}. "
                f"Supported column names: {list(self.column_mappings.keys())}"
            )
        
        logger.debug("All required columns present")
    
    def _dataframe_to_waypoints(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to waypoint dictionaries
        
        Args:
            df: Input DataFrame with standardized columns
            
        Returns:
            List of waypoint dictionaries
        """
        waypoints = []
        
        for index, row in df.iterrows():
            try:
                waypoint = {
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'alt': float(row['alt']),
                    'cmd': 16,  # Default to NAV_WAYPOINT
                    'seq': index
                }
                
                # Add optional fields if available
                for optional_field in ['speed', 'heading', 'radius', 'loiter_time']:
                    if optional_field in row and pd.notna(row[optional_field]):
                        waypoint[optional_field] = float(row[optional_field])
                
                waypoints.append(waypoint)
                
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid row {index}: {e}")
                continue
        
        return waypoints
    
    def _validate_waypoint_data(self, waypoints: List[Dict]) -> None:
        """
        Validate waypoint data values
        
        Args:
            waypoints: List of waypoint dictionaries
        """
        if not waypoints:
            raise ValueError("No valid waypoints found in CSV file")
        
        for i, wp in enumerate(waypoints):
            lat, lon, alt = wp['lat'], wp['lon'], wp['alt']
            
            # Validate coordinate ranges
            if not (-90 <= lat <= 90):
                raise ValueError(f"Waypoint {i}: Invalid latitude {lat} (must be -90 to 90)")
            
            if not (-180 <= lon <= 180):
                raise ValueError(f"Waypoint {i}: Invalid longitude {lon} (must be -180 to 180)")
            
            if alt < -1000 or alt > 50000:
                logger.warning(f"Waypoint {i}: Unusual altitude {alt}m")
        
        logger.debug(f"Validated {len(waypoints)} waypoints")
    
    def detect_file_format(self, file_path: str) -> Dict[str, any]:
        """
        Detect CSV file format and structure
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary with file format information
        """
        file_path = Path(file_path)
        
        try:
            # Read first few lines to detect format
            with open(file_path, 'r', encoding='utf-8') as f:
                sample_lines = [f.readline() for _ in range(5)]
            
            # Detect delimiter
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {}
            
            for delimiter in delimiters:
                count = sum(line.count(delimiter) for line in sample_lines)
                delimiter_counts[delimiter] = count
            
            detected_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            # Try reading with detected delimiter
            df_sample = pd.read_csv(file_path, delimiter=detected_delimiter, nrows=5)
            
            format_info = {
                'delimiter': detected_delimiter,
                'columns': list(df_sample.columns),
                'column_count': len(df_sample.columns),
                'estimated_rows': sum(1 for _ in open(file_path)) - 1,  # Subtract header
                'has_header': True,  # Assume header for now
                'mappable_columns': []
            }
            
            # Check which columns can be mapped
            for col in df_sample.columns:
                if col.strip() in self.column_mappings:
                    format_info['mappable_columns'].append(col.strip())
            
            logger.info(f"Detected CSV format: {format_info}")
            
            return format_info
            
        except Exception as e:
            logger.error(f"Failed to detect CSV format for {file_path}: {e}")
            return {}


class FileParserFactory:
    """
    Factory class for creating appropriate file parsers
    """
    
    @staticmethod
    def create_parser(file_path: str):
        """
        Create appropriate parser based on file extension
        
        Args:
            file_path: Path to file
            
        Returns:
            Appropriate parser instance
            
        Raises:
            ValueError: If file type is not supported
        """
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == '.waypoints':
            return QGCFileParser()
        elif suffix in ['.csv', '.txt']:
            return CSVFileParser()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    @staticmethod
    def parse_mission_file(file_path: str) -> List[Dict]:
        """
        Parse mission file using appropriate parser
        
        Args:
            file_path: Path to mission file
            
        Returns:
            List of waypoint dictionaries
        """
        parser = FileParserFactory.create_parser(file_path)
        return parser.parse_file(file_path)


# Example usage
if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    # Test QGC parser
    try:
        qgc_parser = QGCFileParser()
        print("QGC Parser created successfully")
    except Exception as e:
        print(f"QGC Parser error: {e}")
    
    # Test CSV parser
    try:
        csv_parser = CSVFileParser()
        print("CSV Parser created successfully")
        print(f"Supported column mappings: {list(csv_parser.column_mappings.keys())}")
    except Exception as e:
        print(f"CSV Parser error: {e}")
    
    # Test factory
    try:
        # Test with different file extensions
        for ext in ['.waypoints', '.csv', '.txt']:
            test_file = f"test{ext}"
            parser = FileParserFactory.create_parser(test_file)
            print(f"Created parser for {ext}: {type(parser).__name__}")
    except Exception as e:
        print(f"Factory test error: {e}")