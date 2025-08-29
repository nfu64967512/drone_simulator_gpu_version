# 🚁 專業級無人機群模擬系統 - GPU加速版

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.0%2B%20%7C%2012.0%2B-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

專業級無人機群模擬系統，支援GPU/CPU計算後端靈活切換，提供高效能大規模無人機模擬。

## ✨ 主要特色

- 🚀 **CUDA並行計算**: 支援NVIDIA GPU CUDA加速，大幅提升計算效能
- ⚡ **批次處理**: 使用GPU陣列運算，支援數千架無人機的同時模擬
- 🧠 **智慧回退**: GPU不可用時自動切換到CPU模式，確保程式穩定運行
- 🎯 **模組化加速**: 可選擇性啟用不同模組的GPU加速
- 🔄 **彈性後端**: 支援自動、GPU、CPU、混合四種運算模式

## 📊 效能對比

| 功能模組 | CPU模式 | GPU模式 | 加速倍數 |
|---------|---------|---------|----------|
| 碰撞檢測 | 基準 | 10-50x | 🚀🚀🚀🚀🚀 |
| 座標轉換 | 基準 | 3-15x | 🚀🚀🚀 |
| 軌跡計算 | 基準 | 5-20x | 🚀🚀🚀🚀 |
| 視覺化渲染 | 基準 | 2-5x | 🚀🚀 |

## 🖥️ 系統需求

### 基本需求
- **Python**: 3.8+ (推薦 3.9-3.11)
- **作業系統**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **記憶體**: 最少4GB，推薦8GB+
- **硬碟空間**: 2GB

### GPU加速需求
- **GPU**: NVIDIA顯卡，支援CUDA Compute Capability 3.5+
- **CUDA**: 11.0+ 或 12.0+
- **顯存**: 2GB+ (推薦4GB+)
- **CUDA驅動**: 對應版本的NVIDIA驅動程式

## 🚀 快速開始

### 方法一：一鍵啟動（推薦新手）

```bash
# 下載專案
git clone https://github.com/nfu64967512/drone_simulator_gpu_version.git
cd drone_simulator_gpu_version

# 直接運行啟動器
python launch.py
```

啟動器會自動：
- ✅ 檢查Python版本和依賴
- 🔍 檢測GPU支援
- 📦 提供依賴安裝選項
- 🎛️ 提供後端選擇介面

### 方法二：手動安裝

```bash
# 1. 安裝基本依賴
pip install -r requirements.txt

# 2a. 安裝GPU支援 (CUDA 11.x)
pip install cupy-cuda11x

# 2b. 或安裝GPU支援 (CUDA 12.x)
pip install cupy-cuda12x

# 3. 啟動模擬器
python main.py
```

## 🎛️ 運行模式

```bash
# 自動選擇後端
python main.py --backend auto

# 強制GPU模式
python main.py --backend gpu --device 0

# 強制CPU模式
python main.py --backend cpu

# 性能測試
python main.py --test
```

## 📁 專案結構

```
drone_simulator/
├── main.py                     # 主程式入口
├── launch.py                   # 快速啟動腳本
├── requirements.txt            # 依賴套件
├── config/
│   └── settings.py            # GPU設定檔
├── utils/
│   ├── gpu_utils.py           # GPU/CPU統一計算工具
│   └── logging_config.py      # 日誌系統
├── core/
│   ├── collision_avoidance.py # GPU加速碰撞檢測
│   └── coordinate_system.py   # GPU加速座標轉換
├── simulator/
│   └── drone_simulator.py     # 主模擬器
└── gui/
    └── plot_manager.py        # 視覺化系統
```

## ⚙️ 設定選項

### 運算後端選擇
- 🔄 **自動**: 程式自動偵測最佳後端
- 🚀 **GPU**: 強制使用GPU（需要CUDA支援）
- 🖥️ **CPU**: 使用CPU確保相容性
- ⚙️ **混合**: 不同模組可使用不同後端

### GPU加速模組設定
在 `config/settings.py` 中可微調：

```python
# GPU加速設定
settings.gpu.accelerate_collision_detection = True
settings.gpu.accelerate_coordinate_conversion = True
settings.gpu.accelerate_trajectory_calculation = True
settings.gpu.accelerate_visualization = True

# 效能調校
settings.gpu.batch_size = 1000
settings.gpu.memory_pool = True
settings.performance.parallel_workers = 4
```

## 🧪 效能測試

```bash
# 命令列測試
python main.py --test

# 圖形化測試
python launch.py
# 選擇 "4. 🧪 性能測試"
```

測試包含：
- ✅ 基本陣列運算效能
- ✅ 距離矩陣計算效能
- ✅ 記憶體使用統計
- ✅ GPU/CPU效能對比

## 📊 即時監控

程式運行時會顯示：
- **FPS**: 即時畫面更新率
- **後端**: 當前使用的計算後端
- **記憶體**: GPU/CPU記憶體使用情況

## 🔧 常見問題

### GPU相關問題

**Q: 如何檢查CUDA版本？**
```bash
nvcc --version
```

**Q: GPU記憶體不足怎麼辦？**
- 降低 `batch_size` 設定
- 減少同時模擬的無人機數量
- 關閉其他使用GPU的程式

**Q: macOS系統支援GPU加速嗎？**
- macOS不支援CUDA，會自動使用CPU模式
- 所有功能在CPU模式下完整可用

### 系統相容性

**Linux額外依賴:**
```bash
# 可能需要安裝tkinter
sudo apt-get install python3-tk

# 安裝OpenGL支援
sudo apt-get install python3-opengl
```

**Windows注意事項:**
- 確保已安裝Microsoft Visual C++ Redistributable
- 部分防毒軟體可能阻擋CUDA操作

## 🎯 應用場景

- **大規模無人機群研究**: 支援數千架無人機模擬
- **演算法效能驗證**: GPU加速快速驗證
- **碰撞避免研究**: 高精度即時碰撞檢測
- **無人機編隊訓練**: 真實環境模擬
- **航線規劃驗證**: 大規模路徑最佳化
- **安全性測試**: 極限情境模擬

## 🎓 教學應用

- **程式設計教學**: GPU程式設計實例
- **計算機圖學**: 3D視覺化技術
- **並行計算**: CUDA程式設計學習

## 🛣️ 未來規劃

- 🔮 分散式計算: 多GPU和多機器支援
- 🧠 深度學習整合: AI驅動的無人機行為
- 🌐 WebGL渲染: 瀏覽器內高效能視覺化
- 👥 即時協作: 多用戶同時模擬
- ☁️ 雲端部署: 容器化GPU集群

## 📝 更新日誌

### v2.0 GPU加速版本
- ✅ 新增完整GPU/CPU後端切換
- ✅ CUDA並行碰撞檢測
- ✅ 批次座標轉換加速
- ✅ GPU陣列運算最佳化
- ✅ 智慧記憶體管理
- ✅ 友善的圖形化設定介面
- ✅ 詳細的效能監控

## 🤝 貢獻

歡迎提交Issues和Pull Requests！

## 📞 技術支援

- **GitHub Issues**: [提交問題](https://github.com/nfu64967512/drone_simulator_gpu_version/issues)
- **系統診斷**: 執行 `python launch.py` → 選擇 "6. 🔧 系統診斷"
- **日誌檔案**: 查看 `logs/` 目錄中的詳細日誌

提交問題時請提供：
- 作業系統和版本
- Python版本
- GPU型號和驅動版本
- CUDA版本
- 錯誤日誌內容

## 🙏 致謝

感謝以下開源專案的支持：
- [CuPy](https://cupy.dev/): GPU加速陣列運算
- [CUDA](https://developer.nvidia.com/cuda-zone): NVIDIA並行計算平台
- [NumPy](https://numpy.org/): 科學計算基礎
- [Matplotlib](https://matplotlib.org/): 視覺化支援

## 📄 授權

本專案採用 MIT 授權條款 - 詳見 [LICENSE](LICENSE) 檔案

---

🚁 **讓您的無人機模擬飛得更快更遠！**

💡 **提示**: 第一次使用建議執行 `python launch.py`，它會引導您完成所有設定步驟。