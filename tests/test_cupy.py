#!/usr/bin/env python3
"""
CuPy功能測試腳本
"""
import sys

def test_cupy_basic():
    """測試CuPy基本功能"""
    try:
        print("導入CuPy...")
        import cupy as cp
        
        print(f"CuPy版本: {cp.__version__}")
        print(f"CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
        
        # 測試基本陣列操作
        print("\n測試基本陣列操作:")
        a = cp.array([1, 2, 3, 4, 5])
        print(f"  原始陣列: {a}")
        
        b = a * 2
        print(f"  乘以2: {b}")
        
        c = cp.sum(a)
        print(f"  求和: {c}")
        
        # 測試大陣列
        print("\n測試大陣列:")
        large_array = cp.random.random((1000, 1000))
        result = cp.mean(large_array)
        print(f"  1000x1000陣列平均值: {result}")
        
        # 測試GPU記憶體
        print("\n測試GPU記憶體:")
        mempool = cp.get_default_memory_pool()
        print(f"  已使用記憶體: {mempool.used_bytes() / 1024**2:.1f} MB")
        print(f"  總記憶體: {mempool.total_bytes() / 1024**2:.1f} MB")
        
        # 清理記憶體
        mempool.free_all_blocks()
        print("  記憶體已清理")
        
        print("\n所有測試通過!")
        return True
        
    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cupy_basic()
