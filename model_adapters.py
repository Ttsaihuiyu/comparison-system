import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add RL_recommender to Python path
RL_RECOMMENDER_PATH = Path("../RL_recommender").resolve()
sys.path.append(str(RL_RECOMMENDER_PATH))

def load_movie_info():
    """
    載入電影信息（ID、標題、類型）
    """
    movies_file = RL_RECOMMENDER_PATH / "raw" / "ml-1m" / "movies.dat"
    movies_dict = {}
    
    with open(movies_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                movie_id = int(parts[0])
                title = parts[1]
                genres = parts[2].split("|")
                movies_dict[movie_id] = {
                    'title': title,
                    'genres': genres
                }
    return movies_dict

def load_user_info():
    """
    載入用戶信息（ID、性別、年齡、職業）
    """
    users_file = RL_RECOMMENDER_PATH / "raw" / "ml-1m" / "users.dat"
    users_dict = {}
    
    # 年齡段映射
    age_map = {
        1: "18歲以下",
        18: "18-24歲",
        25: "25-34歲",
        35: "35-44歲",
        45: "45-49歲",
        50: "50-55歲",
        56: "56歲以上"
    }
    
    # 職業映射
    occupation_map = {
        0: "其他/未指定",
        1: "學術/教育工作者",
        2: "藝術家",
        3: "文職/行政",
        4: "大學生/研究生",
        5: "客戶服務",
        6: "醫生/醫療保健",
        7: "高級管理",
        8: "農民",
        9: "家庭主婦",
        10: "中小學生",
        11: "律師",
        12: "程序員",
        13: "退休",
        14: "銷售/市場",
        15: "科學家",
        16: "自由職業",
        17: "技術員/工程師",
        18: "工匠",
        19: "無業",
        20: "作家"
    }
    
    with open(users_file, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) >= 5:
                user_id = int(parts[0])
                gender = "男" if parts[1] == "M" else "女"
                age = age_map.get(int(parts[2]), f"{parts[2]}歲")
                occupation = occupation_map.get(int(parts[3]), "未知職業")
                zip_code = parts[4]
                
                users_dict[user_id] = {
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip_code': zip_code
                }
    return users_dict

def get_user_recommendations(user_embeddings, item_embeddings, user_id, num_recommendations=20):
    """
    為特定用戶生成電影推薦
    """
    with torch.no_grad():
        # 計算用戶和所有電影的相似度
        user_emb = user_embeddings[user_id].unsqueeze(0)  # shape: (1, emb_dim)
        scores = torch.mm(user_emb, item_embeddings.t()).squeeze()  # shape: (num_items,)
        
        # 獲取 top-k 推薦
        _, top_items = torch.topk(scores, num_recommendations)
        
        return top_items.cpu().numpy(), scores[top_items].cpu().numpy()

def run_heuristic_exposure(output_container=None, target_user_id=None, num_recommendations=20):
    """
    運行 Heuristic Exposure 模型 - 使用 user embedding 和 item embedding
    """
    if output_container:
        output_container.text("正在啟動 Heuristic Exposure 模型...")
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        if output_container:
            output_container.text("正在載入模型文件和數據...")
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/simulator/user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/simulator/item_emb.pt", map_location=device, weights_only=True)
        
        if output_container:
            output_container.text(f"成功載入模型: {user_embeddings.shape[0]} 個用戶, {item_embeddings.shape[0]} 個電影")
        
        # 檢查用戶ID是否有效
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
        
        # 為特定用戶生成推薦
        if output_container:
            output_container.text(f"正在為用戶 {target_user_id} 生成推薦...")
        
        recommended_items, scores = get_user_recommendations(
            user_embeddings, item_embeddings, target_user_id, num_recommendations
        )
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        if output_container:
            output_container.success("Heuristic 推薦完成！")
            
            # 顯示用戶信息
            user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
            if user_info:
                output_container.subheader(f"👤 用戶 {target_user_id} 的詳細信息")
                col1, col2, col3 = output_container.columns(3)
                with col1:
                    output_container.metric("性別", user_info['gender'])
                with col2:
                    output_container.metric("年齡", user_info['age'])
                with col3:
                    output_container.metric("職業", user_info['occupation'])
            
            output_container.subheader(f"🎯 為用戶 {target_user_id} 的 Heuristic 推薦結果")
            
            # 創建詳細的推薦結果表格
            recommendations_data = []
            for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
                movie_info = movies_info.get(item_id + 1, {})  # 電影ID從1開始
                recommendations_data.append({
                    '排名': i + 1,
                    '電影ID': item_id,
                    '電影名稱': movie_info.get('title', '未知電影'),
                    '類型': ' | '.join(movie_info.get('genres', ['未知'])),
                    '推薦分數': f"{score:.4f}"
                })
            
            recommendations_df = pd.DataFrame(recommendations_data)
            output_container.dataframe(recommendations_df, use_container_width=True)
        
        return f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"Heuristic 推薦執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        os.chdir(original_dir)
        return error_msg

def run_rl_exposure(output_container=None, target_user_id=None, num_recommendations=20):
    """
    運行 RL Exposure 模型 - 使用 user embedding 和 item embedding
    """
    if output_container:
        output_container.text("正在啟動 RL Exposure 模型...")
        
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        if output_container:
            output_container.text("正在載入模型文件和數據...")
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/simulator/user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/simulator/item_emb.pt", map_location=device, weights_only=True)
        
        if output_container:
            output_container.text(f"成功載入模型: {user_embeddings.shape[0]} 個用戶, {item_embeddings.shape[0]} 個電影")
        
        # 檢查用戶ID是否有效
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
        
        # 為特定用戶生成推薦
        if output_container:
            output_container.text(f"正在為用戶 {target_user_id} 生成推薦...")
        
        recommended_items, scores = get_user_recommendations(
            user_embeddings, item_embeddings, target_user_id, num_recommendations
        )
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        if output_container:
            output_container.success("RL 推薦完成！")
            
            # 顯示用戶信息
            user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
            if user_info:
                output_container.subheader(f"👤 用戶 {target_user_id} 的詳細信息")
                col1, col2, col3 = output_container.columns(3)
                with col1:
                    output_container.metric("性別", user_info['gender'])
                with col2:
                    output_container.metric("年齡", user_info['age'])
                with col3:
                    output_container.metric("職業", user_info['occupation'])
            
            output_container.subheader(f"🤖 為用戶 {target_user_id} 的 RL 推薦結果")
            
            # 創建詳細的推薦結果表格
            recommendations_data = []
            for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
                movie_info = movies_info.get(item_id + 1, {})  # 電影ID從1開始
                recommendations_data.append({
                    '排名': i + 1,
                    '電影ID': item_id,
                    '電影名稱': movie_info.get('title', '未知電影'),
                    '類型': ' | '.join(movie_info.get('genres', ['未知'])),
                    '推薦分數': f"{score:.4f}"
                })
            
            recommendations_df = pd.DataFrame(recommendations_data)
            output_container.dataframe(recommendations_df, use_container_width=True)
        
        return f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"RL 推薦執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        os.chdir(original_dir)
        return error_msg

def check_model_dependencies():
    """
    檢查模型依賴是否滿足
    """
    try:
        # 檢查 RL_recommender 目錄是否存在
        if not RL_RECOMMENDER_PATH.exists():
            return False, f"RL_recommender 目錄不存在: {RL_RECOMMENDER_PATH}"
        
        # 檢查模型文件
        user_emb_file = RL_RECOMMENDER_PATH / "model" / "simulator" / "user_emb.pt"
        item_emb_file = RL_RECOMMENDER_PATH / "model" / "simulator" / "item_emb.pt"
        
        if not user_emb_file.exists():
            return False, f"用戶嵌入文件不存在: {user_emb_file}"
            
        if not item_emb_file.exists():
            return False, f"項目嵌入文件不存在: {item_emb_file}"
        
        return True, "所有依賴檢查通過"
        
    except Exception as e:
        return False, f"依賴檢查出錯: {str(e)}" 