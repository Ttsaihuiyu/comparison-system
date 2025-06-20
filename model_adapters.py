import os
import sys
import subprocess
import torch
import numpy as np
from pathlib import Path

# Add RL_recommender to Python path
RL_RECOMMENDER_PATH = Path("../RL_recommender").resolve()
sys.path.append(str(RL_RECOMMENDER_PATH))

def run_heuristic_exposure(output_container=None):
    """
    運行 Heuristic Exposure 模型的包裝函數
    
    參數:
    output_container: Streamlit 容器，用於顯示輸出
    
    返回:
    results: 模型運行結果
    """
    if output_container:
        output_container.text("正在啟動 Heuristic Exposure 模型...")
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        if output_container:
            output_container.text("正在執行模型...")
        
        # 運行 heuristic exposure 腳本
        result = subprocess.run(
            [sys.executable, "exposure_method/heuristic_exposure.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5分鐘超時
        )
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        if result.returncode == 0:
            if output_container:
                output_container.success("Heuristic Exposure 模型執行成功！")
                output_container.text("輸出結果：")
                output_container.code(result.stdout)
            return result.stdout
        else:
            error_msg = f"模型執行失敗: {result.stderr}"
            if output_container:
                output_container.error(error_msg)
            return error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "模型執行超時（超過5分鐘）"
        if output_container:
            output_container.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"模型執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        return error_msg
    finally:
        # 確保切換回原目錄
        os.chdir(original_dir)

def run_rl_exposure(output_container=None):
    """
    運行 RL Exposure 模型的包裝函數
    
    參數:
    output_container: Streamlit 容器，用於顯示輸出
    
    返回:
    results: 模型運行結果
    """
    if output_container:
        output_container.text("正在啟動 RL Exposure 模型...")
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        if output_container:
            output_container.text("正在執行模型...")
        
        # 運行 RL exposure 腳本
        result = subprocess.run(
            [sys.executable, "exposure_method/rl_exposure_main.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10分鐘超時（RL訓練可能需要更長時間）
        )
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        if result.returncode == 0:
            if output_container:
                output_container.success("RL Exposure 模型執行成功！")
                output_container.text("輸出結果：")
                output_container.code(result.stdout)
            return result.stdout
        else:
            error_msg = f"模型執行失敗: {result.stderr}"
            if output_container:
                output_container.error(error_msg)
            return error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "模型執行超時（超過10分鐘）"
        if output_container:
            output_container.error(error_msg)
        return error_msg
    except Exception as e:
        error_msg = f"模型執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        return error_msg
    finally:
        # 確保切換回原目錄
        os.chdir(original_dir)

def check_model_dependencies():
    """
    檢查模型依賴是否滿足
    
    返回:
    bool: 依賴是否滿足
    """
    try:
        # 檢查 RL_recommender 目錄是否存在
        if not RL_RECOMMENDER_PATH.exists():
            return False, f"RL_recommender 目錄不存在: {RL_RECOMMENDER_PATH}"
        
        # 檢查關鍵文件是否存在
        heuristic_file = RL_RECOMMENDER_PATH / "exposure_method" / "heuristic_exposure.py"
        rl_file = RL_RECOMMENDER_PATH / "exposure_method" / "rl_exposure_main.py"
        
        if not heuristic_file.exists():
            return False, f"Heuristic exposure 文件不存在: {heuristic_file}"
        
        if not rl_file.exists():
            return False, f"RL exposure 文件不存在: {rl_file}"
        
        return True, "所有依賴檢查通過"
        
    except Exception as e:
        return False, f"依賴檢查出錯: {str(e)}" 