# 電影推薦問答機器人

這是一個基於 Streamlit 的電影推薦問答機器人，支援兩種推薦模型：

1. **Heuristic Exposure** - 基於啟發式策略的推薦模型
2. **RL Exposure** - 基於強化學習的推薦模型

## 系統要求

- Python 3.8+
- 確保 `RL_recommender` 項目位於父目錄中（即 `../RL_recommender`）

## 安裝步驟

1. 克隆此項目：
```bash
git clone <your-repo-url>
cd comparison-system
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

3. 確保 `RL_recommender` 項目在正確位置：
```
parent-directory/
├── comparison-system/    # 當前項目
└── RL_recommender/      # 推薦模型項目
```

## 使用方法

1. 啟動 Streamlit 應用：
```bash
streamlit run app.py
```

2. 在瀏覽器中打開顯示的 URL（通常是 `http://localhost:8501`）

3. 在側邊欄選擇您想使用的推薦模型

4. 點擊相應的「運行模型」按鈕

5. 等待模型執行完成，查看結果

## 模型說明

### Heuristic Exposure
- 使用社群偵測算法來分析用戶-項目網絡
- 透過啟發式策略增加推薦項目的多樣性
- 執行時間較短（通常 1-5 分鐘）

### RL Exposure
- 使用強化學習來優化推薦策略
- 動態調整用戶與項目的連結以優化社群結構
- 執行時間較長（可能需要 5-10 分鐘）

## 注意事項

- 首次運行時，系統會自動檢查依賴和文件是否存在
- 模型執行期間請耐心等待，不要關閉瀏覽器
- 如果遇到錯誤，請檢查 `RL_recommender` 項目是否在正確位置且包含所需文件

## 疑難排解

### 常見問題

1. **"RL_recommender 目錄不存在"**
   - 確保 `RL_recommender` 項目位於 `../RL_recommender`
   - 檢查路徑是否正確

2. **"模型文件不存在"**
   - 確保 `RL_recommender` 項目包含所需的 `.py` 文件
   - 檢查 `exposure_method/` 目錄是否存在

3. **依賴安裝問題**
   - 嘗試使用虛擬環境：
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ``` 