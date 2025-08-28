"""
Core module for drone simulator
核心模組：包含碰撞檢測、座標轉換和飛行管理功能
"""

# 核心模組版本
__version__ = "5.2.0"

# 導入核心類別（如果存在的話）
try:
    from core.collision_avoidance import CollisionDetector, GPUCollisionDetector
    from core.coordinate_system import CoordinateConverter, GPUCoordinateConverter  
    from core.flight_manager import FlightManager
    from core.collision_logger import CollisionLogger
    
    # 向後相容的別名
    CollisionAvoidance = CollisionDetector
    CoordinateSystem = CoordinateConverter
    
    __all__ = [
        'CollisionDetector',
        'GPUCollisionDetector', 
        'CoordinateConverter',
        'GPUCoordinateConverter',
        'FlightManager',
        'CollisionLogger',
        # 向後相容
        'CollisionAvoidance',
        'CoordinateSystem'
    ]
    
except ImportError as e:
    # 如果某些模組不存在，提供基本功能
    __all__ = []
    
    # 基本的佔位類別
    class CollisionDetector:
        """基本碰撞檢測器佔位類"""
        def __init__(self):
            pass
        
        def detect_collisions(self, positions):
            return []
    
    class CoordinateConverter:
        """基本座標轉換器佔位類"""
        def __init__(self):
            pass
        
        def convert_to_meters(self, lat, lon, alt):
            return [0, 0, alt]
    
    class FlightManager:
        """基本飛行管理器佔位類"""
        def __init__(self):
            pass
    
    # 向後相容
    CollisionAvoidance = CollisionDetector
    CoordinateSystem = CoordinateConverter
    GPUCollisionDetector = CollisionDetector
    GPUCoordinateConverter = CoordinateConverter
    
    __all__ = [
        'CollisionDetector',
        'CoordinateConverter', 
        'FlightManager',
        'CollisionAvoidance',
        'CoordinateSystem',
        'GPUCollisionDetector',
        'GPUCoordinateConverter'
    ]

# 模組資訊
def get_core_info():
    """獲取核心模組資訊"""
    return {
        'version': __version__,
        'available_modules': __all__,
        'description': '無人機模擬器核心功能模組'
    }