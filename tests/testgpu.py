import cupy as cp

# CuPy 版本
print("CuPy version:", cp.__version__)

# CUDA runtime 版本
print("CUDA runtime version:", cp.cuda.runtime.runtimeGetVersion())

# 驅動版本
print("CUDA driver version:", cp.cuda.runtime.driverGetVersion())

# GPU 數量
print("GPU count:", cp.cuda.runtime.getDeviceCount())

# GPU 名稱
for i in range(cp.cuda.runtime.getDeviceCount()):
    props = cp.cuda.runtime.getDeviceProperties(i)
    print(f"GPU {i}: {props['name'].decode()}")
