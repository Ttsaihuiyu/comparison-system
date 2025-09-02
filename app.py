import streamlit as st
import os
import torch

# 解決 streamlit 和 torch 的衝突
if "STREAMLIT_SERVER_ENABLE_FILE_WATCHER" not in os.environ:
    if "torch" in globals():
        torch.classes.__path__ = []
import time
from model_adapters import (
    run_heuristic_exposure, 
    run_heuristic_20epoch_exposure,
    run_raw_20epoch_exposure,
    check_model_dependencies, 
    load_mapping_files, 
    get_user_liked_movies, 
    get_recommendations_data, 
    get_heuristic_20epoch_recommendations_data,
    get_raw_20epoch_recommendations_data,
    display_recommendations,
    get_simulator_recommendations_data,
    display_simulator_recommendations
)

# 配置頁面 - 開啟自動重載
st.set_page_config(
    page_title="電影推薦系統比較",
    page_icon="🎬",
    layout="wide"
)

# 添加重載按鈕（開發時使用）
if st.sidebar.button("強制重載"):
    st.cache_data.clear()
    st.rerun()

# 初始化 session state
if 'current_user_id' not in st.session_state:
    st.session_state.current_user_id = None
if 'current_num_recommendations' not in st.session_state:
    st.session_state.current_num_recommendations = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# 檢查依賴
dependencies_ok, dependency_msg = check_model_dependencies()
if not dependencies_ok:
    st.error(f"系統檢查失敗: {dependency_msg}")
    st.stop()
else:
    st.success("系統檢查通過！")

# 主標題
st.title("電影推薦系統比較平台")
st.markdown("比較不同推薦模型的效果，幫您找到最適合的電影！")

# 用戶輸入區域
st.subheader("個人化電影推薦設置")

# 模型選擇 - 四個模型對應四組 embeddings
st.subheader("選擇推薦模型")
model_choice = st.radio(
    "請選擇要使用的推薦模型：",
    ["Heuristic 模型", "Heuristic 20 Epoch 模型", "Raw 模型", "Raw 20 Epoch 模型"],
    index=0,
    help="選擇不同的推薦模型來比較推薦效果"
)

col1, col2 = st.columns([2, 1])

with col1:
    user_id = st.number_input(
        "請輸入用戶ID：", 
        min_value=1, 
        max_value=5950, 
        value=1, 
        step=1,
        help="輸入您想要推薦電影的用戶ID（範圍：1-5950，對應訓練後的模型）"
    )

with col2:
    num_recommendations = st.slider(
        "推薦電影數量：", 
        min_value=1, 
        max_value=20, 
        value=10,
        help="選擇要推薦的電影數量"
    )

# 添加說明信息
st.info(f"即將為用戶 {user_id} 推薦 {num_recommendations} 部電影")

# Main content - 四個模型對應四組 embeddings
st.header(f"推薦模型: {model_choice}")

# 檢查是否需要重新生成推薦
need_new_recommendations = (
    st.session_state.current_user_id != user_id or 
    st.session_state.current_num_recommendations != num_recommendations or
    st.session_state.current_model != model_choice or
    st.session_state.get('current_data') is None
)

# 添加執行按鈕
if st.button(f"開始為用戶 {user_id} 推薦電影", key="model_btn", type="primary"):
    with st.spinner("正在分析您的偏好並生成推薦，請稍候..."):
        # 使用 user_id - 1 因為內部索引從0開始
        if model_choice == "Heuristic 模型":
            recommendations_data, result_msg = get_recommendations_data(user_id - 1, num_recommendations)
        elif model_choice == "Heuristic 20 Epoch 模型":
            recommendations_data, result_msg = get_heuristic_20epoch_recommendations_data(user_id - 1, num_recommendations)
        elif model_choice == "Raw 模型":
            recommendations_data, result_msg = get_simulator_recommendations_data(user_id - 1, num_recommendations)
        elif model_choice == "Raw 20 Epoch 模型":
            recommendations_data, result_msg = get_raw_20epoch_recommendations_data(user_id - 1, num_recommendations)
    
    if recommendations_data is not None:
        st.session_state.current_data = recommendations_data
        st.session_state.current_user_id = user_id
        st.session_state.current_num_recommendations = num_recommendations
        st.session_state.current_model = model_choice
        st.success("推薦完成！希望您會喜歡這些電影！")
    else:
        st.error(f"推薦過程中遇到問題: {result_msg}")

# 顯示推薦結果
if (st.session_state.get('current_data') is not None and not need_new_recommendations):
    if model_choice == "Raw 模型":
        display_simulator_recommendations(st, st.session_state.current_data)
    else:
        display_recommendations(st, st.session_state.current_data)

elif (st.session_state.get('current_data') is not None and need_new_recommendations):
    st.info("檢測到設置變更，請點擊推薦按鈕重新生成推薦結果")

# 添加更詳細的說明信息
st.sidebar.markdown("---")
st.sidebar.subheader("使用說明")
st.sidebar.write("""
1. **選擇推薦模型**：選擇四種不同的模型之一
2. **輸入用戶ID**：選擇要分析的用戶（1-5950）
3. **設置推薦數量**：選擇想要的推薦電影數量
4. **點擊推薦按鈕**：開始生成個性化推薦
5. **點擊愛心按鈕**：將喜歡的電影加入交互記錄
6. **檢視歷史**：查看用戶的歷史交互記錄
7. **手動比對**：自行比較推薦結果與歷史記錄

**四種模型對比**：
- **Heuristic 模型**：使用 heuristic embeddings + 動態更新
- **Heuristic 20 Epoch 模型**：使用 heuristic 20 epoch embeddings + 動態更新  
- **Raw 模型**：使用 raw embeddings
- **Raw 20 Epoch 模型**：使用 raw 20 epoch embeddings
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 提示：不同模型可能會給出不同的推薦結果，您可以比較它們的效果！")