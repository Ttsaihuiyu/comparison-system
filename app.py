import streamlit as st
import time
from model_adapters import (
    run_heuristic_exposure, 
    check_model_dependencies, 
    load_mapping_files, 
    get_user_liked_movies, 
    get_recommendations_data, 
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
if st.sidebar.button("🔄 強制重載"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# 顯示最後更新時間（確認是否有重載）
st.sidebar.text(f"最後更新: {time.strftime('%H:%M:%S')}")

st.title("🎬 電影推薦系統")

# 初始化 session state
if 'recommendations_data' not in st.session_state:
    st.session_state.recommendations_data = None
if 'simulator_recommendations_data' not in st.session_state:
    st.session_state.simulator_recommendations_data = None
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
    st.success("✅ 系統檢查通過！")

# 用戶輸入區域
st.subheader("🎯 個人化電影推薦設置")

# 模型選擇
st.subheader("🤖 選擇推薦模型")
model_choice = st.radio(
    "請選擇要使用的推薦模型：",
    ["Heuristic Exposure 模型 ", "原始模型  "],
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
st.info(f"🎯 即將為用戶 {user_id} 推薦 {num_recommendations} 部電影")

# Main content
if model_choice.startswith("Heuristic"):
    st.header("🎯 Heuristic Exposure 推薦模型")
   
    
    # 檢查是否需要重新生成推薦（用戶、推薦數量或模型改變時）
    need_new_recommendations = (
        st.session_state.current_user_id != user_id or 
        st.session_state.current_num_recommendations != num_recommendations or
        st.session_state.current_model != model_choice or
        (model_choice.startswith("Heuristic") and st.session_state.recommendations_data is None)
    )

    # 添加執行按鈕
    if st.button(f"🚀 開始為用戶 {user_id} 推薦電影", key="heuristic_btn", type="primary"):
        with st.spinner("🎬 正在分析您的偏好並生成推薦，請稍候..."):
            # 使用 user_id - 1 因為內部索引從0開始
            recommendations_data, result_msg = get_recommendations_data(user_id - 1, num_recommendations)
        
        if recommendations_data is not None:
            st.session_state.recommendations_data = recommendations_data
            st.session_state.current_user_id = user_id
            st.session_state.current_num_recommendations = num_recommendations
            st.session_state.current_model = model_choice
            st.success("🎉 推薦完成！希望您會喜歡這些電影！")
        else:
            st.error(f"⚠️ 推薦過程中遇到問題: {result_msg}")

    # 顯示推薦結果（如果存在且參數匹配）
    if (st.session_state.recommendations_data is not None and 
        model_choice.startswith("Heuristic") and
        not need_new_recommendations):
        
        display_recommendations(st, st.session_state.recommendations_data)

    elif (st.session_state.recommendations_data is not None and 
          model_choice.startswith("Heuristic") and
          need_new_recommendations):
        
        st.info("📝 檢測到設置變更，請點擊推薦按鈕重新生成推薦結果")

else:  # Simulator model
    st.header("🎯原始推薦模型")
    
    # 檢查是否需要重新生成推薦（用戶、推薦數量或模型改變時）
    need_new_recommendations = (
        st.session_state.current_user_id != user_id or 
        st.session_state.current_num_recommendations != num_recommendations or
        st.session_state.current_model != model_choice or
        (model_choice == "原始模型  " and st.session_state.simulator_recommendations_data is None)
    )
    
    # 添加執行按鈕
    if st.button(f"🚀 開始為用戶 {user_id} 推薦電影", key="simulator_btn", type="primary"):
        with st.spinner("🎬 正在分析您的偏好並生成推薦，請稍候..."):
            # 使用 user_id - 1 因為內部索引從0開始
            simulator_recommendations_data, result_msg = get_simulator_recommendations_data(user_id - 1, num_recommendations)
        
        if simulator_recommendations_data is not None:
            st.session_state.simulator_recommendations_data = simulator_recommendations_data
            st.session_state.current_user_id = user_id
            st.session_state.current_num_recommendations = num_recommendations
            st.session_state.current_model = model_choice
            st.success("🎉 推薦完成！希望您會喜歡這些電影！")
        else:
            st.error(f"⚠️ 推薦過程中遇到問題: {result_msg}")

    # 顯示推薦結果（如果存在且參數匹配）
    if (st.session_state.simulator_recommendations_data is not None and 
        model_choice == "原始模型  " and
        not need_new_recommendations):
        
        display_simulator_recommendations(st, st.session_state.simulator_recommendations_data)

    elif (st.session_state.simulator_recommendations_data is not None and 
          model_choice == "原始模型  " and
          need_new_recommendations):
        
        st.info("📝 檢測到設置變更，請點擊推薦按鈕重新生成推薦結果")

# 添加更詳細的說明信息
st.sidebar.markdown("---")
st.sidebar.subheader("📖 使用說明")
st.sidebar.write("""
1. **選擇推薦模型**：選擇 Heuristic 或 Simulator 模型
2. **輸入用戶ID**：選擇要分析的用戶（1-5950）
3. **設置推薦數量**：選擇想要的推薦電影數量
4. **點擊推薦按鈕**：開始生成個性化推薦
5. **點擊愛心按鈕**：將喜歡的電影加入交互記錄
6. **檢視歷史**：查看用戶的歷史交互記錄
7. **手動比對**：自行比較推薦結果與歷史記錄

⭐ **新功能**: 可以比較不同模型的推薦效果！
""")

 

st.sidebar.markdown("---")
st.sidebar.subheader("📊 數據說明")
st.sidebar.write("""
- **用戶數據**：5,950個用戶（訓練後）
- **電影數據**：3,191部電影（訓練後）
- **評分數據**：1,000,209個評分
- **評分範圍**：1-5星
- **數據來源**：MovieLens 1M數據集
- **輸出文件**：用戶交互記錄CSV
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 電影類型")
st.sidebar.write("""
包含18種電影類型：
動作、冒險、動畫、兒童、喜劇、犯罪、
紀錄片、劇情、奇幻、黑色電影、恐怖、
音樂劇、神秘、愛情、科幻、驚悚、戰爭、西部
""")

