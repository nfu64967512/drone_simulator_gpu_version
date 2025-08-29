# examples/basic_simulation.py
"""
基本模擬使用範例
展示如何使用API創建和運行基本的無人機模擬
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_simulation_config
from simulator.advanced_simulator_main import AdvancedDroneSimulator
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_basic_simulation():
    """運行基本模擬範例"""
    logger.info("開始基本模擬範例")
    
    # 載入配置
    config = get_simulation_config()
    
    # 創建模擬器
    simulator = AdvancedDroneSimulator(config)
    
    # 創建測試任務
    success = simulator.create_test_mission("square", 4)
    if not success:
        logger.error("測試任務創建失敗")
        return
    
    # 運行模擬
    simulator.play_simulation()
    
    # 模擬30秒
    import time
    for _ in range(300):  # 30秒，每次0.1秒
        update_result = simulator.update_simulation(0.1)
        
        if update_result['updated']:
            current_time = update_result['current_time']
            warnings = len(update_result.get('collision_warnings', []))
            
            if int(current_time) % 5 == 0:  # 每5秒輸出一次狀態
                logger.info(f"時間: {current_time:.1f}s, 碰撞警告: {warnings}")
        
        if not simulator.is_playing:
            break
            
        time.sleep(0.1)
    
    # 導出修正任務
    if simulator.modified_missions:
        results = simulator.export_modified_missions("exports/basic_example")
        logger.info(f"導出 {len(results)} 個修正任務檔案")
    
    # 清理
    simulator.cleanup()
    logger.info("基本模擬範例完成")

if __name__ == "__main__":
    run_basic_simulation()

# =============================================
# examples/gpu_performance_demo.py
"""
GPU性能演示範例
比較GPU和CPU模式的性能差異
"""

import sys
import os
import time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_utils import GPUSystemChecker, PerformanceBenchmark
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gpu_performance_demo():
    """GPU性能演示"""
    logger.info("GPU性能演示開始")
    
    # 檢查系統
    checker = GPUSystemChecker()
    gpu_info = checker.get_gpu_info()
    
    print(f"GPU可用: {'是' if gpu_info['available'] else '否'}")
    if gpu_info['available']:
        print(f"GPU名稱: {gpu_info['name']}")
        print(f"GPU記憶體: {gpu_info['total_memory_mb']:.0f} MB")
    
    # 運行基準測試
    benchmark = PerformanceBenchmark()
    
    print("\n運行性能基準測試...")
    benchmark.benchmark_array_operations(size=1000, iterations=50)
    benchmark.benchmark_distance_matrix(num_points=500, iterations=20)
    benchmark.benchmark_memory_bandwidth(size_mb=100)
    
    # 顯示結果
    benchmark.print_summary()

if __name__ == "__main__":
    gpu_performance_demo()

# =============================================
# examples/batch_processing.py
"""
批次處理範例
展示如何批次處理多個任務檔案
"""

import sys
import os
import glob
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.qgc_handlers import parse_mission_file, MissionFileExporter
from config.settings import get_simulation_config
from simulator.advanced_simulator_main import AdvancedDroneSimulator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_process_missions(input_directory: str, output_directory: str):
    """
    批次處理任務檔案
    
    Args:
        input_directory: 輸入目錄
        output_directory: 輸出目錄
    """
    logger.info(f"開始批次處理: {input_directory} -> {output_directory}")
    
    # 查找所有支援的檔案
    input_path = Path(input_directory)
    mission_files = []
    
    for pattern in ['*.waypoints', '*.csv']:
        mission_files.extend(input_path.glob(pattern))
    
    if not mission_files:
        logger.warning(f"在 {input_directory} 中未找到任務檔案")
        return
    
    logger.info(f"找到 {len(mission_files)} 個任務檔案")
    
    # 創建模擬器
    config = get_simulation_config()
    simulator = AdvancedDroneSimulator(config)
    exporter = MissionFileExporter()
    
    processed_count = 0
    
    for mission_file in mission_files:
        try:
            logger.info(f"處理檔案: {mission_file.name}")
            
            # 載入檔案
            success = simulator.load_mission_files([str(mission_file)])
            if not success:
                logger.warning(f"載入失敗: {mission_file.name}")
                continue
            
            # 快速模擬分析（不運行完整模擬）
            if len(simulator.drones) >= 2:
                # 只進行軌跡衝突分析
                simulator._analyze_trajectory_conflicts()
            
            # 導出修正任務
            if simulator.modified_missions:
                output_dir = Path(output_directory) / mission_file.stem
                output_dir.mkdir(parents=True, exist_ok=True)
                
                results = simulator.export_modified_missions(str(output_dir))
                logger.info(f"導出 {len(results)} 個修正檔案到 {output_dir}")
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"處理 {mission_file.name} 時發生錯誤: {e}")
        
        finally:
            # 重置模擬器狀態
            simulator.reset_simulation()
    
    logger.info(f"批次處理完成: 成功處理 {processed_count}/{len(mission_files)} 個檔案")
    simulator.cleanup()

if __name__ == "__main__":
    # 使用範例
    input_dir = "input_missions"
    output_dir = "processed_missions"
    
    # 創建範例目錄
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    batch_process_missions(input_dir, output_dir)

# =============================================
# tests/test_gpu_utils.py
"""
GPU工具模組測試
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_utils import (
    get_array_module, 
    GPUSystemChecker,
    PerformanceBenchmark,
    auto_detect_backend
)

class TestGPUUtils(unittest.TestCase):
    """GPU工具測試類"""
    
    def setUp(self):
        """設置測試"""
        self.checker = GPUSystemChecker()
        
    def test_get_array_module(self):
        """測試陣列模組獲取"""
        # CPU模式
        xp_cpu = get_array_module(use_gpu=False)
        self.assertEqual(xp_cpu.__name__, 'numpy')
        
        # GPU模式（如果可用）
        xp_gpu = get_array_module(use_gpu=True)
        self.assertIn(xp_gpu.__name__, ['numpy', 'cupy'])
    
    def test_system_info(self):
        """測試系統信息獲取"""
        system_info = self.checker.get_system_info()
        
        # 檢查必要字段
        required_fields = [
            'platform', 'python_version', 'cpu_count', 
            'memory_gb', 'gpu_available'
        ]
        
        for field in required_fields:
            self.assertIn(field, system_info)
    
    def test_gpu_info(self):
        """測試GPU信息獲取"""
        gpu_info = self.checker.get_gpu_info()
        
        self.assertIn('available', gpu_info)
        self.assertIsInstance(gpu_info['available'], bool)
        
        if gpu_info['available']:
            self.assertIn('name', gpu_info)
            self.assertIn('total_memory_mb', gpu_info)
    
    def test_auto_detect_backend(self):
        """測試自動後端檢測"""
        backend = auto_detect_backend()
        self.assertIn(backend, ['gpu', 'cpu'])
    
    def test_performance_benchmark(self):
        """測試性能基準測試"""
        benchmark = PerformanceBenchmark()
        
        # 運行小規模測試
        benchmark.benchmark_array_operations(size=100, iterations=5)
        results = benchmark.get_results()
        
        self.assertIn('array_operations', results)
        self.assertIn('cpu_time_ms', results['array_operations'])

if __name__ == '__main__':
    unittest.main()

# =============================================
# tests/test_collision.py
"""
碰撞檢測系統測試
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.collision_avoidance import CollisionAvoidanceSystem
from config.settings import SafetyConfig

class TestCollisionDetection(unittest.TestCase):
    """碰撞檢測測試類"""
    
    def setUp(self):
        """設置測試"""
        safety_config = SafetyConfig(
            safety_distance=5.0,
            warning_distance=8.0,
            critical_distance=3.0
        )
        self.collision_system = CollisionAvoidanceSystem(safety_config, use_gpu=False)
    
    def test_collision_detection(self):
        """測試碰撞檢測"""
        # 創建測試位置（兩架無人機距離2米，應該觸發碰撞警告）
        positions = {
            'drone_1': {'x': 0, 'y': 0, 'z': 10},
            'drone_2': {'x': 2, 'y': 0, 'z': 10}
        }
        
        warnings, loiters = self.collision_system.check_realtime_collisions(positions, 0.0)
        
        # 應該有一個碰撞警告
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].distance, 2.0)
        self.assertEqual(warnings[0].severity.value, 'critical')
    
    def test_no_collision(self):
        """測試無碰撞情況"""
        # 創建測試位置（兩架無人機距離10米，不應該觸發警告）
        positions = {
            'drone_1': {'x': 0, 'y': 0, 'z': 10},
            'drone_2': {'x': 10, 'y': 0, 'z': 10}
        }
        
        warnings, loiters = self.collision_system.check_realtime_collisions(positions, 0.0)
        
        # 不應該有碰撞警告
        self.assertEqual(len(warnings), 0)
        self.assertEqual(len(loiters), 0)
    
    def test_multiple_drones(self):
        """測試多架無人機"""
        # 四架無人機，其中兩對會發生碰撞
        positions = {
            'drone_1': {'x': 0, 'y': 0, 'z': 10},
            'drone_2': {'x': 2, 'y': 0, 'z': 10},  # 與drone_1碰撞
            'drone_3': {'x': 20, 'y': 0, 'z': 10},
            'drone_4': {'x': 22, 'y': 0, 'z': 10}  # 與drone_3碰撞
        }
        
        warnings, loiters = self.collision_system.check_realtime_collisions(positions, 0.0)
        
        # 應該有兩個碰撞警告
        self.assertEqual(len(warnings), 2)

if __name__ == '__main__':
    unittest.main()

# =============================================
# tests/benchmark_tests.py
"""
基準性能測試
"""

import unittest
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_utils import PerformanceBenchmark, GPUSystemChecker
from core.collision_avoidance import GPUCollisionDetector
from core.coordinate_system import EarthCoordinateSystem

class BenchmarkTests(unittest.TestCase):
    """基準性能測試"""
    
    def test_collision_detection_performance(self):
        """測試碰撞檢測性能"""
        detector = GPUCollisionDetector(use_gpu=False)
        
        # 創建大量無人機位置
        num_drones = 100
        positions = {}
        
        for i in range(num_drones):
            positions[f'drone_{i}'] = {
                'x': i * 2.0,  # 2米間距
                'y': 0,
                'z': 10
            }
        
        # 測試性能
        start_time = time.time()
        warnings = detector.detect_collisions_vectorized(positions, 5.0)
        elapsed = time.time() - start_time
        
        print(f"碰撞檢測 {num_drones} 架無人機用時: {elapsed:.3f}s")
        print(f"發現 {len(warnings)} 個碰撞警告")
        
        # 性能要求：100架無人機檢測應在1秒內完成
        self.assertLess(elapsed, 1.0)
    
    def test_coordinate_transformation_performance(self):
        """測試坐標轉換性能"""
        coord_system = EarthCoordinateSystem(use_gpu=False)
        coord_system.set_origin(24.0, 121.0)
        
        # 創建大量坐標點
        num_points = 10000
        coordinates = [(24.0 + i * 0.001, 121.0 + i * 0.001) for i in range(num_points)]
        
        # 測試批量轉換性能
        start_time = time.time()
        results = coord_system.batch_coordinate_transform(coordinates, to_meters=True)
        elapsed = time.time() - start_time
        
        print(f"坐標轉換 {num_points} 個點用時: {elapsed:.3f}s")
        
        # 性能要求：10000個點轉換應在1秒內完成
        self.assertLess(elapsed, 1.0)
        self.assertEqual(len(results), num_points)

if __name__ == '__main__':
    unittest.main()