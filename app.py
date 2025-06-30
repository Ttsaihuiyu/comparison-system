import streamlit as st
from model_adapters import run_heuristic_exposure, check_model_dependencies

st.set_page_config(
    page_title="電影推薦系統",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 電影推薦系統")
 

# 檢查依賴
dependencies_ok, dependency_msg = check_model_dependencies()
if not dependencies_ok:
    st.error(f"系統檢查失敗: {dependency_msg}")
    st.stop()
else:
    st.success("✅ 系統檢查通過！")

# 用戶輸入區域
st.subheader("🎯 個人化電影推薦設置")

col1, col2 = st.columns([2, 1])

with col1:
    user_id = st.number_input(
        "請輸入用戶ID：", 
        min_value=1, 
        max_value=6040, 
        value=1, 
        step=1,
        help="輸入您想要推薦電影的用戶ID（範圍：1-6040）"
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
st.header("🎯 Heuristic Exposure 推薦模型")

# 添加執行按鈕
if st.button(f"🚀 開始為用戶 {user_id} 推薦電影", key="heuristic_btn", type="primary"):
    output_container = st.container()
    with st.spinner("🎬 正在分析您的偏好並生成推薦，請稍候..."):
        # 使用 user_id - 1 因為內部索引從0開始
        result = run_heuristic_exposure(output_container, user_id - 1, num_recommendations)
    
    if "成功" in result:
         
        st.success("🎉 推薦完成！希望您會喜歡這些電影！")
    else:
        st.error("⚠️ 推薦過程中遇到問題，請檢查用戶ID是否正確。")

# 添加更詳細的說明信息
st.sidebar.markdown("---")
st.sidebar.subheader("📖 使用說明")
st.sidebar.write("""
1. **輸入用戶ID**：選擇要分析的用戶（1-6040）
2. **設置推薦數量**：選擇想要的推薦電影數量
3. **點擊推薦按鈕**：開始生成個性化推薦
4. **查看結果**：瀏覽用戶信息和推薦電影
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 推薦算法說明")
st.sidebar.info("""
**Heuristic Exposure 模型**
- 使用啟發式策略
- 通過社群偵測增加推薦多樣性
- 適合探索新類型電影
""")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 數據說明")
st.sidebar.write("""
- **用戶數據**：6,040個用戶
- **電影數據**：3,883部電影
- **評分數據**：1,000,209個評分
- **評分範圍**：1-5星
- **數據來源**：MovieLens 1M數據集
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎬 電影類型")
st.sidebar.write("""
包含18種電影類型：
動作、冒險、動畫、兒童、喜劇、犯罪、
紀錄片、劇情、奇幻、黑色電影、恐怖、
音樂劇、神秘、愛情、科幻、驚悚、戰爭、西部
""") 