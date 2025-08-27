# 🚀 無人機群模擬器 GPU加速版本

專業級無人機群模擬系統，支援GPU/CPU計算後端靈活切換，提供高效能大規模無人機模擬。

## ✨ 主要特色

### 🔥 GPU加速功能
- **🚀 CUDA並行計算**: 支援NVIDIA GPU CUDA加速，大幅提升計算效能
- **⚡ 批次處理**: 使用GPU陣列運算，支援數千架無人機的同時模擬
- **🧠 智慧回退**: GPU不可用時自動切換到CPU模式，確保程式穩定運行
- **🎯 模組化加速**: 可選擇性啟用不同模組的GPU加速

### 📊 效能提升對比
| 功能模組 | CPU模式 | GPU模式 | 加速倍數 |
|---------|---------|---------|-----------|
| 碰撞檢測 | 基準 | 10-50x | 🚀🚀🚀🚀🚀 |
| 座標轉換 | 基準 | 3-15x | 🚀🚀🚀 |
| 軌跡計算 | 基準 | 5-20x | 🚀🚀🚀🚀 |
| 視覺化渲染 | 基準 | 2-5x | 🚀🚀 |

### 🎛️ 靈活的後端選擇
- **🔄 自動模式**: 自動檢測並選擇最佳計算後端
- **🚀 GPU模式**: 強制使用GPU加速（需要CUDA支援）
- **🖥️ CPU模式**: 純CPU計算，最大相容性
- **⚙️ 混合模式**: 不同模組可使用不同後端

## 🛠️ 安裝需求

### 基本需求
- **Python**: 3.8+ (推薦 3.9-3.11)
- **作業系統**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **記憶體**: 最少4GB，推薦8GB+
- **硬碟空間**: 2GB

### GPU加速需求 (可選)
- **GPU**: NVIDIA顯卡，支援CUDA Compute Capability 3.5+
- **CUDA**: 11.0+ 或 12.0+ 
- **顯存**: 2GB+ (推薦4GB+)
- **CUDA驅動**: 對應版本的NVIDIA驅動程式

## 🚀 快速開始

### 1️⃣ 一鍵啟動 (推薦)
```bash
# 下載專案
git clone https://github.com/nfu64967512/drone_simulator.git
cd drone_simulator

# 直接運行啟動器
python launch.py
```

啟動器會自動：
- ✅ 檢查Python版本和依賴
- 🔍 檢測GPU支援
- 📦 提供依賴安裝選項  
- 🎛️ 提供後端選擇介面

### 2️⃣ 手動安裝
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

### 3️⃣ 命令列啟動
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

## 📋 新增檔案結構

```
drone_simulator/
│
├── main.py                    # 🆕 支援GPU/CPU選擇的主程式
├── launch.py                  # 🆕 快速啟動腳本
├── requirements.txt           # 🆕 包含GPU依賴的需求檔案
│
├── config/
│   └── settings.py           # 🆕 GPU設定和後端選擇
│
├── utils/
│   ├── gpu_utils.py          # 🆕 GPU/CPU統一計算工具
│   └── logging_config.py     # 🆕 專業日誌系統
│
├── core/
│   ├── collision_avoidance.py  # 🆕 GPU加速碰撞檢測
│   └── coordinate_system.py    # 🆕 GPU加速座標轉換
│
├── simulator/
│   └── drone_simulator.py      # 🆕 GPU加速主模擬器
│
└── gui/
    └── plot_manager.py         # 🆕 GPU加速視覺化系統
```

## 🎮 使用指南

### 🚀 啟動模擬器

1. **圖形化啟動**
   ```bash
   python launch.py
   ```
   - 提供友善的選單介面
   - 自動檢測系統能力
   - 一鍵安裝依賴
   - 性能測試功能

2. **直接啟動**
   ```bash
   python main.py
   ```
   - 彈出後端選擇對話框
   - 顯示系統GPU資訊
   - 支援進階設定

### ⚙️ 設定GPU加速

程式會自動提供後端選擇介面，您可以：

- **🔄 選擇「自動」**: 程式自動偵測最佳後端
- **🚀 選擇「GPU」**: 強制使用GPU（需要CUDA支援）
- **🖥️ 選擇「CPU」**: 使用CPU確保相容性

### 🎛️ 進階設定

可在 `config/settings.py` 中微調：

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

## 🧪 性能測試

### 內建測試
```bash
python main.py --test
```

測試包含：
- ✅ 基本陣列運算效能
- ✅ 距離矩陣計算效能  
- ✅ 記憶體使用統計
- ✅ GPU/CPU效能對比

### 自訂測試
```bash
python launch.py
# 選擇 "4. 🧪 性能測試"
```

### 效能監控
程式運行時會顯示：
- **FPS**: 即時畫面更新率
- **後端**: 當前使用的計算後端
- **記憶體**: GPU/CPU記憶體使用情況

## 🔧 故障排除

### GPU相關問題

#### ❌ "CuPy未安裝" 
```bash
# 檢查CUDA版本
nvcc --version

# 安裝對應版本的CuPy
pip install cupy-cuda11x  # CUDA 11.x
pip install cupy-cuda12x  # CUDA 12.x
```

#### ❌ "GPU檢測失敗"
1. 確認NVIDIA驅動程式已安裝
2. 檢查CUDA工具包安裝
3. 重新啟動電腦
4. 嘗試 `nvidia-smi` 命令

#### ❌ "CUDA記憶體不足"
- 降低 `batch_size` 設定
- 減少同時模擬的無人機數量
- 關閉其他使用GPU的程式

### 相容性問題

#### 🍎 macOS用戶
- macOS不支援CUDA，會自動使用CPU模式
- 所有功能在CPU模式下完整可用

#### 🐧 Linux用戶
```bash
# 可能需要安裝tkinter
sudo apt-get install python3-tk

# 安裝OpenGL支援
sudo apt-get install python3-opengl
```

#### 🪟 Windows用戶  
- 確保已安裝Microsoft Visual C++ Redistributable
- 部分防毒軟體可能阻擋CUDA操作

## 📊 效能最佳化建議

### 🚀 GPU模式最佳化
- **使用較大的batch_size**: 提升GPU利用率
- **啟用記憶體池**: 減少記憶體分配開銷  
- **關閉debug模式**: 避免頻繁CPU-GPU數據傳輸

### 🖥️ CPU模式最佳化
- **增加並行工作數**: 充分利用多核心CPU
- **關閉不必要的視覺效果**: 減少計算負擔
- **使用較小的更新間隔**: 提升回應速度

## 🤝 技術支援

### 📞 獲得幫助
- **GitHub Issues**: [提交問題](https://github.com/nfu64967512/drone_simulator/issues)
- **系統診斷**: 執行 `python launch.py` → 選擇 "6. 🔧 系統診斷"
- **日誌檔案**: 查看 `logs/` 目錄中的詳細日誌

### 🐛 回報問題
請提供以下資訊：
- 作業系統和版本
- Python版本
- GPU型號和驅動版本 
- CUDA版本
- 錯誤日誌內容

## 🎯 使用場景

### 🏢 學術研究
- **大規模無人機群研究**: 支援數千架無人機模擬
- **演算法效能驗證**: GPU加速快速驗證
- **碰撞避免研究**: 高精度即時碰撞檢測

### 🏭 工業應用  
- **無人機編隊訓練**: 真實環境模擬
- **航線規劃驗證**: 大規模路徑最佳化
- **安全性測試**: 極限情境模擬

### 🎓 教育用途
- **程式設計教學**: GPU程式設計實例
- **計算機圖學**: 3D視覺化技術
- **並行計算**: CUDA程式設計學習

## 📈 未來發展

### 🚀 計畫中功能
- [ ] **分散式計算**: 多GPU和多機器支援
- [ ] **深度學習整合**: AI驅動的無人機行為
- [ ] **WebGL渲染**: 瀏覽器內高效能視覺化
- [ ] **即時協作**: 多用戶同時模擬

### 🔬 研究方向
- [ ] **量子模擬**: 量子演算法應用
- [ ] **邊緣計算**: 嵌入式GPU支援  
- [ ] **雲端部署**: 容器化GPU集群

## 📝 更新日誌

### v5.2.0 (GPU加速版本)
- ✅ 新增完整GPU/CPU後端切換
- ✅ CUDA並行碰撞檢測  
- ✅ 批次座標轉換加速
- ✅ GPU陣列運算最佳化
- ✅ 智慧記憶體管理
- ✅ 友善的圖形化設定介面
- ✅ 詳細的效能監控

---

## 🙏 致謝

感謝以下開源專案的支持：
- **CuPy**: GPU加速陣列運算
- **CUDA**: NVIDIA並行計算平台
- **NumPy**: 科學計算基礎
- **Matplotlib**: 視覺化支援
- **Python**: 程式語言基礎

---

**🚁 讓您的無人機模擬飛得更快更遠！**

> 💡 **提示**: 第一次使用建議執行 `python launch.py`，它會引導您完成所有設定步驟。