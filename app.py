import streamlit as st
from model_adapters import run_heuristic_exposure, run_rl_exposure, check_model_dependencies

st.title("電影推薦系統")
st.write("你好！我是你的電影推薦小幫手。")

# 檢查依賴
dependencies_ok, dependency_msg = check_model_dependencies()
if not dependencies_ok:
    st.error(f"系統檢查失敗: {dependency_msg}")
    st.stop()
else:
    st.success("系統檢查通過！")

# Sidebar for model selection
st.sidebar.title("模型選擇")
model_choice = st.sidebar.selectbox(
    "請選擇您想使用的推薦模型：",
    ("Heuristic Exposure", "RL Exposure")
)

# Main content
if model_choice == "Heuristic Exposure":
    st.header("Heuristic Exposure 模型")
    st.write("此模型使用啟發式策略，透過社群偵測來增加曝光項目的多樣性。")
    
    # 添加執行按鈕
    if st.button("運行 Heuristic Exposure 模型", key="heuristic_btn"):
        # 創建一個容器來顯示進度和結果
        output_container = st.empty()
        with st.spinner("正在執行模型，請稍候..."):
            result = run_heuristic_exposure(output_container)
        
        # 顯示最終結果
        st.subheader("執行結果")
        if "模型執行失敗" in result or "模型執行超時" in result or "模型執行出錯" in result:
            st.error("模型執行遇到問題，請檢查錯誤信息。")
        else:
            st.success("模型執行完成！")
            
else:
    st.header("RL Exposure 模型")
    st.write("此模型使用強化學習來動態調整使用者與項目的連結，以優化社群結構。")
    
    # 添加執行按鈕
    if st.button("運行 RL Exposure 模型", key="rl_btn"):
        # 創建一個容器來顯示進度和結果
        output_container = st.empty()
        with st.spinner("正在執行模型，請稍候...（可能需要幾分鐘）"):
            result = run_rl_exposure(output_container)
        
        # 顯示最終結果
        st.subheader("執行結果")
        if "模型執行失敗" in result or "模型執行超時" in result or "模型執行出錯" in result:
            st.error("模型執行遇到問題，請檢查錯誤信息。")
        else:
            st.success("模型執行完成！")

# 添加一些說明信息
st.sidebar.markdown("---")
st.sidebar.subheader("使用說明")
st.sidebar.write("1. 選擇想要使用的推薦模型")
st.sidebar.write("2. 點擊相應的運行按鈕")
st.sidebar.write("3. 等待模型執行完成")
st.sidebar.write("4. 查看執行結果") 