import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime

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

def get_user_recommendations(user_embeddings, item_embeddings, user_id, num_recommendations=20, exclude_ids=None):
    """
    為特定用戶生成電影推薦，可選擇排除已觀看電影
    """
    with torch.no_grad():
        # 計算用戶和所有電影的相似度
        user_emb = user_embeddings[user_id].unsqueeze(0)  # shape: (1, emb_dim)
        scores = torch.mm(user_emb, item_embeddings.t()).squeeze()  # shape: (num_items,)
        
        # 如果有要排除的電影，將其分數設為負無窮大
        if exclude_ids and isinstance(exclude_ids, list) and len(exclude_ids) > 0:
            scores[exclude_ids] = -float('inf')
        
        # 獲取 top-k 推薦
        top_scores, top_items = torch.topk(scores, num_recommendations)
        
        return top_items.cpu().numpy(), top_scores.cpu().numpy()

def find_similar_users(user_embeddings, target_user_id, num_similar_users=2):
    """
    找到與目標用戶最相似的用戶
    """
    with torch.no_grad():
        target_emb = user_embeddings[target_user_id].unsqueeze(0)  # shape: (1, emb_dim)
        
        # 計算與所有用戶的相似度（使用餘弦相似度）
        similarities = torch.cosine_similarity(target_emb, user_embeddings, dim=1)
        
        # 將目標用戶自己的相似度設為負無窮大，避免推薦自己
        similarities[target_user_id] = -float('inf')
        
        # 獲取最相似的用戶
        _, top_users = torch.topk(similarities, num_similar_users)
        similarity_scores = similarities[top_users]
        
        return top_users.cpu().numpy(), similarity_scores.cpu().numpy()

def get_user_watched_movies(user_id, reverse_uid_map, reverse_mid_map, num_movies=5):
    """
    獲取指定用戶看過的電影（按評分排序，取高分電影）
    """
    try:
        # 讀取數據
        train_df = pd.read_csv(RL_RECOMMENDER_PATH / 'data/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        val_df = pd.read_csv(RL_RECOMMENDER_PATH / 'data/val.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        test_df = pd.read_csv(RL_RECOMMENDER_PATH / 'data/test.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        combined = pd.concat([train_df, val_df, test_df])
        
        # 找到該用戶的所有交互記錄
        user_interactions = combined[combined['user_id'] == user_id].copy()
        
        if user_interactions.empty:
            return []
        
        # 按評分降序排序，選擇高分電影
        user_interactions = user_interactions.sort_values(['rating', 'timestamp'], ascending=[False, False])
        
        # 取前 num_movies 部電影
        top_movies = user_interactions.head(num_movies)
        
        # 轉換為原始電影ID
        watched_movies = []
        for _, row in top_movies.iterrows():
            original_movie_id = reverse_mid_map.get(row['movie_id'])
            if original_movie_id is not None:
                watched_movies.append({
                    'movie_id': original_movie_id,
                    'rating': row['rating']
                })
        
        return watched_movies
        
    except Exception as e:
        print(f"獲取用戶觀看電影失敗: {str(e)}")
        return []

def run_heuristic_exposure(output_container=None, target_user_id=None, num_recommendations=20):
    """
    運行 Heuristic Exposure 模型 - 使用 user embedding 和 item embedding
    """
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb.pt", map_location=device, weights_only=True)
        
        # 载入映射文件
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            error_msg = "無法載入映射文件"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
        
        # 獲取用戶歷史交互記錄，以便過濾推薦
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        # 檢查用戶ID是否有效
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
        
        # 為特定用戶生成推薦，並排除已觀看電影
        recommended_items, scores = get_user_recommendations(
            user_embeddings, item_embeddings, target_user_id, num_recommendations,
            exclude_ids=watched_movie_ids
        )
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        if output_container:
            output_container.success("Heuristic 推薦完成！")
            
            # 顯示用戶信息
            user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
            if user_info:
                output_container.subheader(f"👤 用戶 {target_user_id} 的詳細信息")
                # 使用表格形式確保完整顯示
                import pandas as pd
                user_data = pd.DataFrame({
                    '性別': [user_info['gender']],
                    '年齡': [user_info['age']],
                    '職業': [user_info['occupation']],
                    '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
                })
                output_container.dataframe(user_data, use_container_width=True, hide_index=True)

            output_container.subheader(f"🎯 為用戶 {target_user_id} 的 Heuristic 推薦結果")
            
            # 創建詳細的推薦結果表格
            recommendations_data = []
            for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
                # 正確的ID映射邏輯：將映射後的item_id轉換為原始電影ID
                original_movie_id = reverse_mid_map.get(item_id)
                if original_movie_id is None:
                    continue  # 跳過無法映射的電影
                
                movie_info = movies_info.get(original_movie_id, {})
                movie_title = movie_info.get('title', '未知電影')
                movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
                
                recommendations_data.append({
                    '排名': i + 1,
                    '電影ID': original_movie_id,
                    '電影名稱': movie_title,
                    '類型': movie_genres,
                    '推薦分數': f"{score:.4f}"
                })
            
            recommendations_df = pd.DataFrame(recommendations_data)
            
            # 添加表格標題
            col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 3])
            with col1:
                output_container.write("**排名**")
            with col2:
                output_container.write("**電影ID**")
            with col3:
                output_container.write("**電影名稱**")
            with col4:
                output_container.write("**類型**")
            with col5:
                output_container.write("**推薦分數**")
            with col6:
                output_container.write("**喜愛**")
            
            output_container.write("---")
            
            # 顯示推薦結果表格，並為每行添加愛心按鈕
            for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
                # 正確的ID映射邏輯：將映射後的item_id轉換為原始電影ID
                original_movie_id = reverse_mid_map.get(item_id)
                if original_movie_id is None:
                    continue  # 跳過無法映射的電影
                
                movie_info = movies_info.get(original_movie_id, {})
                movie_title = movie_info.get('title', '未知電影')
                movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
                
                col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 1])
                
                with col1:
                    output_container.write(f"**{i+1}**")
                with col2:
                    output_container.write(f"{original_movie_id}")  # 顯示原始電影ID
                with col3:
                    output_container.write(f"**{movie_title}**")
                with col4:
                    output_container.write(f"{movie_genres}")
                with col5:
                    output_container.write(f"{score:.4f}")
                with col6:
                    if output_container.button("❤️", key=f"heart_{target_user_id}_{i}", help="加入我的最愛"):
                        # 添加到用戶交互記錄，直接使用原始電影ID
                        success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                        
                        if success:
                            output_container.success(f"❤️ 已將《{movie_title}》添加到交互記錄！")
                            # 更新 session_state 中的交互記錄
                            import streamlit as st
                            if 'recommendations_data' in st.session_state:
                                # 重新獲取更新後的交互記錄
                                updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                                st.session_state.recommendations_data['user_interactions_df'] = updated_interactions_df
                                st.session_state.recommendations_data['watched_movie_ids'] = updated_watched_ids
                        else:
                            output_container.error("❌ 添加失敗，請稍後再試")
            
            # 顯示用戶歷史交互記錄
            if not user_interactions_df.empty:
                output_container.subheader(f"📚 用戶 {target_user_id} 的歷史交互記錄")
                output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
                
                # 按時間戳降序排列
                sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
                
                # 顯示所有交互記錄
                output_container.dataframe(sorted_interactions, use_container_width=True)
                
                # 顯示統計信息
                output_container.info(f"📊 共 {len(sorted_interactions)} 條交互記錄")
            else:
                output_container.warning("該用戶沒有歷史交互記錄")
        
        return f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"Heuristic 推薦執行出錯: {str(e)}"
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
        user_emb_file = RL_RECOMMENDER_PATH / "model" / "RS" / "user_emb.pt"
        item_emb_file = RL_RECOMMENDER_PATH / "model" / "RS" / "item_emb.pt"
        
        if not user_emb_file.exists():
            return False, f"用戶嵌入文件不存在: {user_emb_file}"
            
        if not item_emb_file.exists():
            return False, f"項目嵌入文件不存在: {item_emb_file}"
        
        return True, "所有依賴檢查通過"
        
    except Exception as e:
        return False, f"依賴檢查出錯: {str(e)}"

def load_mapping_files():
    """
    载入用户和电影的映射文件
    """
    try:
        # 载入映射关系
        uid_map_file = RL_RECOMMENDER_PATH / "mapping" / "uid_map.pkl"
        mid_map_file = RL_RECOMMENDER_PATH / "mapping" / "mid_map.pkl"
        
        with open(uid_map_file, 'rb') as f:
            uid_map = pickle.load(f)
        with open(mid_map_file, 'rb') as f:
            mid_map = pickle.load(f)
        
        # 创建反向映射（从新ID到旧ID）
        reverse_uid_map = {v: k for k, v in uid_map.items()}
        reverse_mid_map = {v: k for k, v in mid_map.items()}
        
        return uid_map, mid_map, reverse_uid_map, reverse_mid_map, True
    except Exception as e:
        print(f"载入映射文件失败: {str(e)}")
        return {}, {}, {}, {}, False

def get_user_interactions(user_id, reverse_uid_map, reverse_mid_map):
    """
    获取用户的历史交互记录
    """
    try:
        # 读取数据
        train_df = pd.read_csv(RL_RECOMMENDER_PATH / 'data/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        val_df = pd.read_csv(RL_RECOMMENDER_PATH / 'data/val.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        test_df = pd.read_csv(RL_RECOMMENDER_PATH / 'data/test.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        
        movies = pd.read_csv(RL_RECOMMENDER_PATH / 'raw/ml-1m/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python', encoding='latin-1')
        
        combined = pd.concat([train_df, val_df, test_df])
        
        # 找到该用户的所有交互记录
        user_interactions = combined[combined['user_id'] == user_id].copy()
        
        if user_interactions.empty:
            return pd.DataFrame(), []
        
        # 将movie_id转换回原始ID
        user_interactions['original_movie_id'] = user_interactions['movie_id'].map(reverse_mid_map)
        
        # 与movies数据合并
        result = pd.merge(
            user_interactions,
            movies,
            left_on='original_movie_id',
            right_on='movie_id',
            how='left'
        )
        
        # 选择要显示的栏位并重新命名
        result = result[['original_movie_id', 'title', 'genres', 'rating', 'timestamp']]
        result.columns = ['Movie_ID', 'Title', 'Genres', 'Rating', 'Timestamp']
        
        # 檢查重複電影並優先顯示重複的記錄
        movie_counts = result['Movie_ID'].value_counts()
        duplicated_movies = movie_counts[movie_counts > 1].index.tolist()
        
        if duplicated_movies:
            # 分離重複和非重複記錄
            duplicated_records = result[result['Movie_ID'].isin(duplicated_movies)]
            non_duplicated_records = result[~result['Movie_ID'].isin(duplicated_movies)]
            
            # 重複記錄按Movie_ID排序，非重複記錄按時間戳排序
            duplicated_records = duplicated_records.sort_values(['Movie_ID', 'Timestamp'])
            non_duplicated_records = non_duplicated_records.sort_values('Timestamp', ascending=False)
            
            # 重複記錄優先，然後是非重複記錄
            result = pd.concat([duplicated_records, non_duplicated_records], ignore_index=True)
        else:
            # 如果沒有重複，按時間戳降序排列
            result = result.sort_values('Timestamp', ascending=False)
        
        # 保存到CSV文件
        interaction_file = RL_RECOMMENDER_PATH / "interaction_collect" / f"user_{user_id}_interactions.csv"
        result.to_csv(interaction_file, index=False)
        
        # 返回用户看过的电影ID列表
        watched_movie_ids = user_interactions['movie_id'].tolist()
        
        return result, watched_movie_ids
        
    except Exception as e:
        print(f"获取用户交互记录失败: {str(e)}")
        return pd.DataFrame(), []

def add_to_liked_movies(user_id, movie_info, liked_movies_file):
    """
    将电影添加到用户喜好名单
    """
    # 创建新的喜好记录
    new_like = {
        'user_id': user_id,
        'movie_id': movie_info.get('movie_id'),
        'movie_title': movie_info.get('title'),
        'genres': movie_info.get('genres'),
        'liked_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 检查文件是否存在
    try:
        if os.path.exists(liked_movies_file):
            liked_df = pd.read_csv(liked_movies_file)
        else:
            liked_df = pd.DataFrame()
        
        # 添加新记录
        liked_df = pd.concat([liked_df, pd.DataFrame([new_like])], ignore_index=True)
        
        # 保存文件
        liked_df.to_csv(liked_movies_file, index=False)
        return True
        
    except Exception as e:
        print(f"保存喜好记录失败: {str(e)}")
        return False

def get_user_liked_movies(user_id):
    """
    获取用户的喜好电影名单
    """
    liked_movies_file = f"user_{user_id}_liked_movies.csv"
    
    try:
        if os.path.exists(liked_movies_file):
            return pd.read_csv(liked_movies_file)
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"读取喜好记录失败: {str(e)}")
        return pd.DataFrame()

def add_movie_to_interactions(user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map):
    """
    將電影添加到用戶的交互記錄中
    """
    try:
        print(f"🔍 開始處理: 用戶{user_id}, 原始電影ID{original_movie_id}")
        
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        print(f"📁 切換到目錄: {RL_RECOMMENDER_PATH}")
        
        # 讀取現有的交互記錄
        train_file = "data/train.dat"
        
        # 檢查電影ID映射 - 從原始ID映射到模型使用的ID
        mid_map_file = RL_RECOMMENDER_PATH / "mapping" / "mid_map.pkl"
        with open(mid_map_file, 'rb') as f:
            mid_map = pickle.load(f)
        
        # 將原始電影ID轉換為映射後的ID（用於存儲到train.dat）
        mapped_movie_id = mid_map.get(original_movie_id)
        if mapped_movie_id is None:
            print(f"❌ 原始電影ID {original_movie_id} 不在映射中")
            os.chdir(original_dir)
            return False
        
        print(f"✅ 電影ID映射成功: 原始ID{original_movie_id} -> 映射ID{mapped_movie_id}")
        
        # 生成新的交互記錄
        current_timestamp = int(datetime.now().timestamp())
        new_interaction = f"{user_id},{mapped_movie_id},5,{current_timestamp}\n"  # 給予5星評分
        print(f"📝 新交互記錄: {new_interaction.strip()}")
        
        # 將新記錄添加到訓練數據中
        with open(train_file, 'a', encoding='utf-8') as f:
            f.write(new_interaction)
        print(f"✅ 已添加到 {train_file}")
        
        # 同時保存到單獨的用戶交互文件
        interaction_file = f"interaction_collect/user_{user_id}_interactions.csv"
        
        # 創建新記錄的數據框（使用原始電影ID）
        new_record = pd.DataFrame({
            'Movie_ID': [original_movie_id],
            'Title': [movie_info.get('title', '未知電影')],
            'Genres': [' | '.join(movie_info.get('genres', ['未知']))],
            'Rating': [5],
            'Timestamp': [current_timestamp]
        })
        
        # 如果文件存在，讀取現有記錄並添加新記錄
        if os.path.exists(interaction_file):
            existing_df = pd.read_csv(interaction_file)
            # 檢查是否已經存在該電影記錄
            if original_movie_id not in existing_df['Movie_ID'].values:
                updated_df = pd.concat([new_record, existing_df], ignore_index=True)
                updated_df.to_csv(interaction_file, index=False)
                print(f"✅ 已更新用戶交互文件: {interaction_file}")
            else:
                print(f"⚠️ 電影已存在於用戶交互記錄中")
        else:
            new_record.to_csv(interaction_file, index=False)
            print(f"✅ 已創建新的用戶交互文件: {interaction_file}")
        
        os.chdir(original_dir)
        print(f"🎉 處理完成，返回成功")
        return True
        
    except Exception as e:
        print(f"❌ 添加電影到交互記錄失敗: {str(e)}")
        print(f"❌ 錯誤類型: {type(e).__name__}")
        import traceback
        print(f"❌ 詳細錯誤: {traceback.format_exc()}")
        os.chdir(original_dir)
        return False

def get_recommendations_data(target_user_id, num_recommendations=20):
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb.pt", map_location=device, weights_only=True)
        
        # 載入映射文件
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            os.chdir(original_dir)
            return None, "無法載入映射文件"
        
        # 先獲取用戶歷史交互記錄，以便在推薦中過濾
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        # 檢查用戶ID是否有效
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            os.chdir(original_dir)
            return None, error_msg
        
        # 為特定用戶生成推薦，並排除已觀看電影
        recommended_items, scores = get_user_recommendations(
            user_embeddings, item_embeddings, target_user_id, num_recommendations,
            exclude_ids=watched_movie_ids
        )
        
        # 準備返回數據
        recommendations_data = {
            'user_id': target_user_id,
            'recommended_items': recommended_items,
            'scores': scores,
            'movies_info': movies_info,
            'users_info': users_info,
            'user_interactions_df': user_interactions_df,
            'watched_movie_ids': watched_movie_ids,
            'reverse_uid_map': reverse_uid_map,
            'reverse_mid_map': reverse_mid_map,
            'user_embeddings': user_embeddings  # 添加用戶嵌入，用於社群推薦
        }
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        return recommendations_data, "成功生成推薦"
        
    except Exception as e:
        error_msg = f"推薦生成出錯: {str(e)}"
        os.chdir(original_dir)
        return None, error_msg

def display_recommendations(output_container, recommendations_data):
    """
    顯示推薦結果 - 從保存的數據中渲染界面
    """
    if not recommendations_data:
        output_container.error("沒有推薦數據可顯示")
        return
    
    target_user_id = recommendations_data['user_id']
    recommended_items = recommendations_data['recommended_items']
    scores = recommendations_data['scores']
    movies_info = recommendations_data['movies_info']
    users_info = recommendations_data['users_info']
    user_interactions_df = recommendations_data['user_interactions_df']
    watched_movie_ids = recommendations_data['watched_movie_ids']
    reverse_uid_map = recommendations_data['reverse_uid_map']
    reverse_mid_map = recommendations_data['reverse_mid_map']
    user_embeddings = recommendations_data['user_embeddings'] # 獲取用戶嵌入
    
    output_container.success("Heuristic 推薦完成！")
    
    # 顯示用戶信息
    user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
    if user_info:
        output_container.subheader(f"👤 用戶 {target_user_id} 的詳細信息")
        # 使用表格形式確保完整顯示
        import pandas as pd
        user_data = pd.DataFrame({
            '性別': [user_info['gender']],
            '年齡': [user_info['age']],
            '職業': [user_info['occupation']],
            '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
        })
        output_container.dataframe(user_data, use_container_width=True, hide_index=True)

    output_container.subheader(f"🎯 為用戶 {target_user_id} 的 Heuristic 推薦結果")
    
    # 添加表格標題
    col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 3])
    with col1:
        output_container.write("**排名**")
    with col2:
        output_container.write("**電影ID**")
    with col3:
        output_container.write("**電影名稱**")
    with col4:
        output_container.write("**類型**")
    with col5:
        output_container.write("**推薦分數**")
    with col6:
        output_container.write("**喜愛**")
    
    output_container.write("---")
    
    # 顯示推薦結果表格，並為每行添加愛心按鈕
    for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
        # 正確的ID映射邏輯：將映射後的item_id轉換為原始電影ID
        original_movie_id = reverse_mid_map.get(item_id)
        if original_movie_id is None:
            continue  # 跳過無法映射的電影
        
        movie_info = movies_info.get(original_movie_id, {})
        movie_title = movie_info.get('title', '未知電影')
        movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
        
        col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 3])
        
        with col1:
            output_container.write(f"**{i+1}**")
        with col2:
            output_container.write(f"{original_movie_id}")  # 顯示原始電影ID
        with col3:
            output_container.write(f"**{movie_title}**")
        with col4:
            output_container.write(f"{movie_genres}")
        with col5:
            output_container.write(f"{score:.4f}")
        with col6:
            if output_container.button("❤️", key=f"heart_{target_user_id}_{i}", help="加入我的最愛"):
                # 添加到用戶交互記錄，直接使用原始電影ID
                success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                
                if success:
                    output_container.success(f"❤️ 已將《{movie_title}》添加到交互記錄！")
                    # 更新 session_state 中的交互記錄
                    import streamlit as st
                    if 'recommendations_data' in st.session_state:
                        # 重新獲取更新後的交互記錄
                        updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                        st.session_state.recommendations_data['user_interactions_df'] = updated_interactions_df
                        st.session_state.recommendations_data['watched_movie_ids'] = updated_watched_ids
                else:
                    output_container.error("❌ 添加失敗，請稍後再試")
    
    # 顯示用戶歷史交互記錄
    if not user_interactions_df.empty:
        output_container.subheader(f"📚 用戶 {target_user_id} 的歷史交互記錄")
        output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
        
        # 按時間戳降序排列
        sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
        
        # 顯示所有交互記錄
        output_container.dataframe(sorted_interactions, use_container_width=True)
        
        # 顯示統計信息
        output_container.info(f"📊 共 {len(sorted_interactions)} 條交互記錄")
    else:
        output_container.warning("該用戶沒有歷史交互記錄")

    # 添加社群推薦功能
    output_container.markdown("---")
    output_container.subheader("👥 社群推薦 - 看過類似電影的用戶推薦")
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 找到最相似的兩個用戶（使用傳入的用戶嵌入）
        similar_users, similarity_scores = find_similar_users(user_embeddings, target_user_id, num_similar_users=2)
        
        # 為每個相似用戶顯示推薦
        for i, (similar_user_id, similarity_score) in enumerate(zip(similar_users, similarity_scores)):
            # 獲取該用戶看過的高分電影
            watched_movies = get_user_watched_movies(similar_user_id, reverse_uid_map, reverse_mid_map, num_movies=5)
            
            if watched_movies:
                output_container.subheader(f"🎬 用戶 {similar_user_id} 跟你看過類似的電影，所以你也可能喜歡看這些電影")
                output_container.info(f"相似度: {similarity_score:.4f}")
                
                # 創建推薦表格
                col1, col2, col3, col4, col5 = output_container.columns([1, 1, 4, 3, 2])
                with col1:
                    output_container.write("**排名**")
                with col2:
                    output_container.write("**電影ID**")
                with col3:
                    output_container.write("**電影名稱**")
                with col4:
                    output_container.write("**類型**")
                with col5:
                    output_container.write("**用戶評分**")
                
                output_container.write("---")
                
                # 顯示推薦電影
                for j, movie_data in enumerate(watched_movies):
                    movie_id = movie_data['movie_id']
                    rating = movie_data['rating']
                    
                    movie_info = movies_info.get(movie_id, {})
                    movie_title = movie_info.get('title', '未知電影')
                    movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
                    
                    col1, col2, col3, col4, col5 = output_container.columns([1, 1, 4, 3, 2])
                    
                    with col1:
                        output_container.write(f"**{j+1}**")
                    with col2:
                        output_container.write(f"{movie_id}")
                    with col3:
                        output_container.write(f"**{movie_title}**")
                    with col4:
                        output_container.write(f"{movie_genres}")
                    with col5:
                        output_container.write(f"⭐ {rating}")
                
                if i < len(similar_users) - 1:  # 如果不是最後一個用戶，添加分隔線
                    output_container.markdown("---")
            else:
                output_container.warning(f"用戶 {similar_user_id} 沒有足夠的觀看記錄")
        
        # 切換回原目錄
        os.chdir(original_dir)
        
    except Exception as e:
        output_container.error(f"社群推薦功能出錯: {str(e)}")
        # 確保切換回原目錄
        try:
            os.chdir(original_dir)
        except:
            pass

def run_simulator_exposure(output_container=None, target_user_id=None, num_recommendations=20):
    """
    運行 Simulator Exposure 模型 - 使用 simulator 目錄中的原始 user embedding 和 item embedding
    """
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 使用 simulator 目錄中的原始 embeddings
        user_embeddings = torch.load("model/simulator/user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/simulator/item_emb.pt", map_location=device, weights_only=True)
        
        # 载入映射文件
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            error_msg = "無法載入映射文件"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
        
        # 獲取用戶歷史交互記錄，以便過濾推薦
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        # 檢查用戶ID是否有效
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
        
        # 為特定用戶生成推薦，並排除已觀看電影
        recommended_items, scores = get_user_recommendations(
            user_embeddings, item_embeddings, target_user_id, num_recommendations,
            exclude_ids=watched_movie_ids
        )
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        if output_container:
            output_container.success("Simulator 推薦完成！")
            
            # 顯示用戶信息
            user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
            if user_info:
                output_container.subheader(f"👤 用戶 {target_user_id} 的詳細信息")
                # 使用表格形式確保完整顯示
                import pandas as pd
                user_data = pd.DataFrame({
                    '性別': [user_info['gender']],
                    '年齡': [user_info['age']],
                    '職業': [user_info['occupation']],
                    '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
                })
                output_container.dataframe(user_data, use_container_width=True, hide_index=True)

            output_container.subheader(f"🎯 為用戶 {target_user_id} 的 Simulator 推薦結果")
            
            # 創建詳細的推薦結果表格
            recommendations_data = []
            for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
                # 正確的ID映射邏輯：將映射後的item_id轉換為原始電影ID
                original_movie_id = reverse_mid_map.get(item_id)
                if original_movie_id is None:
                    continue  # 跳過無法映射的電影
                
                movie_info = movies_info.get(original_movie_id, {})
                movie_title = movie_info.get('title', '未知電影')
                movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
                
                recommendations_data.append({
                    '排名': i + 1,
                    '電影ID': original_movie_id,
                    '電影名稱': movie_title,
                    '類型': movie_genres,
                    '推薦分數': f"{score:.4f}"
                })
            
            recommendations_df = pd.DataFrame(recommendations_data)
            
            # 添加表格標題
            col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 3])
            with col1:
                output_container.write("**排名**")
            with col2:
                output_container.write("**電影ID**")
            with col3:
                output_container.write("**電影名稱**")
            with col4:
                output_container.write("**類型**")
            with col5:
                output_container.write("**推薦分數**")
            with col6:
                output_container.write("**喜愛**")
            
            output_container.write("---")
            
            # 顯示推薦結果表格，並為每行添加愛心按鈕
            for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
                # 正確的ID映射邏輯：將映射後的item_id轉換為原始電影ID
                original_movie_id = reverse_mid_map.get(item_id)
                if original_movie_id is None:
                    continue  # 跳過無法映射的電影
                
                movie_info = movies_info.get(original_movie_id, {})
                movie_title = movie_info.get('title', '未知電影')
                movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
                
                col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 1])
                
                with col1:
                    output_container.write(f"**{i+1}**")
                with col2:
                    output_container.write(f"{original_movie_id}")  # 顯示原始電影ID
                with col3:
                    output_container.write(f"**{movie_title}**")
                with col4:
                    output_container.write(f"{movie_genres}")
                with col5:
                    output_container.write(f"{score:.4f}")
                with col6:
                    if output_container.button("❤️", key=f"sim_heart_{target_user_id}_{i}", help="加入我的最愛"):
                        # 添加到用戶交互記錄，直接使用原始電影ID
                        success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                        
                        if success:
                            output_container.success(f"❤️ 已將《{movie_title}》添加到交互記錄！")
                            # 更新 session_state 中的交互記錄
                            import streamlit as st
                            if 'simulator_recommendations_data' in st.session_state:
                                # 重新獲取更新後的交互記錄
                                updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                                st.session_state.simulator_recommendations_data['user_interactions_df'] = updated_interactions_df
                                st.session_state.simulator_recommendations_data['watched_movie_ids'] = updated_watched_ids
                        else:
                            output_container.error("❌ 添加失敗，請稍後再試")
            
            # 顯示用戶歷史交互記錄
            if not user_interactions_df.empty:
                output_container.subheader(f"📚 用戶 {target_user_id} 的歷史交互記錄")
                output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
                
                # 按時間戳降序排列
                sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
                
                # 顯示所有交互記錄
                output_container.dataframe(sorted_interactions, use_container_width=True)
                
                # 顯示統計信息
                output_container.info(f"📊 共 {len(sorted_interactions)} 條交互記錄")
            else:
                output_container.warning("該用戶沒有歷史交互記錄")
        
        return f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"Simulator 推薦執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        os.chdir(original_dir)
        return error_msg

def get_simulator_recommendations_data(target_user_id, num_recommendations=20):
    """
    獲取 Simulator 推薦數據
    """
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 使用 simulator 目錄中的原始 embeddings
        user_embeddings = torch.load("model/simulator/user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/simulator/item_emb.pt", map_location=device, weights_only=True)
        
        # 載入映射文件
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            os.chdir(original_dir)
            return None, "無法載入映射文件"
        
        # 先獲取用戶歷史交互記錄，以便在推薦中過濾
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        # 檢查用戶ID是否有效
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            os.chdir(original_dir)
            return None, error_msg
        
        # 為特定用戶生成推薦，並排除已觀看電影
        recommended_items, scores = get_user_recommendations(
            user_embeddings, item_embeddings, target_user_id, num_recommendations,
            exclude_ids=watched_movie_ids
        )
        
        # 準備返回數據
        recommendations_data = {
            'user_id': target_user_id,
            'recommended_items': recommended_items,
            'scores': scores,
            'movies_info': movies_info,
            'users_info': users_info,
            'user_interactions_df': user_interactions_df,
            'watched_movie_ids': watched_movie_ids,
            'reverse_uid_map': reverse_uid_map,
            'reverse_mid_map': reverse_mid_map,
            'user_embeddings': user_embeddings  # 添加用戶嵌入，用於社群推薦
        }
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        return recommendations_data, "成功生成推薦"
        
    except Exception as e:
        error_msg = f"推薦生成出錯: {str(e)}"
        os.chdir(original_dir)
        return None, error_msg

def display_simulator_recommendations(output_container, recommendations_data):
    """
    顯示 Simulator 推薦結果 - 從保存的數據中渲染界面
    """
    if not recommendations_data:
        output_container.error("沒有推薦數據可顯示")
        return
    
    target_user_id = recommendations_data['user_id']
    recommended_items = recommendations_data['recommended_items']
    scores = recommendations_data['scores']
    movies_info = recommendations_data['movies_info']
    users_info = recommendations_data['users_info']
    user_interactions_df = recommendations_data['user_interactions_df']
    watched_movie_ids = recommendations_data['watched_movie_ids']
    reverse_uid_map = recommendations_data['reverse_uid_map']
    reverse_mid_map = recommendations_data['reverse_mid_map']
    user_embeddings = recommendations_data['user_embeddings'] # 獲取用戶嵌入
    
    output_container.success("Simulator 推薦完成！")
    
    # 顯示用戶信息
    user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
    if user_info:
        output_container.subheader(f"👤 用戶 {target_user_id} 的詳細信息")
        # 使用表格形式確保完整顯示
        import pandas as pd
        user_data = pd.DataFrame({
            '性別': [user_info['gender']],
            '年齡': [user_info['age']],
            '職業': [user_info['occupation']],
            '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
        })
        output_container.dataframe(user_data, use_container_width=True, hide_index=True)

    output_container.subheader(f"🎯 為用戶 {target_user_id} 的 Simulator 推薦結果")
    
    # 添加表格標題
    col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 3])
    with col1:
        output_container.write("**排名**")
    with col2:
        output_container.write("**電影ID**")
    with col3:
        output_container.write("**電影名稱**")
    with col4:
        output_container.write("**類型**")
    with col5:
        output_container.write("**推薦分數**")
    with col6:
        output_container.write("**喜愛**")
    
    output_container.write("---")
    
    # 顯示推薦結果表格，並為每行添加愛心按鈕
    for i, (item_id, score) in enumerate(zip(recommended_items, scores)):
        # 正確的ID映射邏輯：將映射後的item_id轉換為原始電影ID
        original_movie_id = reverse_mid_map.get(item_id)
        if original_movie_id is None:
            continue  # 跳過無法映射的電影
        
        movie_info = movies_info.get(original_movie_id, {})
        movie_title = movie_info.get('title', '未知電影')
        movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
        
        col1, col2, col3, col4, col5, col6 = output_container.columns([1, 1, 4, 3, 2, 3])
        
        with col1:
            output_container.write(f"**{i+1}**")
        with col2:
            output_container.write(f"{original_movie_id}")  # 顯示原始電影ID
        with col3:
            output_container.write(f"**{movie_title}**")
        with col4:
            output_container.write(f"{movie_genres}")
        with col5:
            output_container.write(f"{score:.4f}")
        with col6:
            if output_container.button("❤️", key=f"sim_heart_{target_user_id}_{i}", help="加入我的最愛"):
                # 添加到用戶交互記錄，直接使用原始電影ID
                success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                
                if success:
                    output_container.success(f"❤️ 已將《{movie_title}》添加到交互記錄！")
                    # 更新 session_state 中的交互記錄
                    import streamlit as st
                    if 'simulator_recommendations_data' in st.session_state:
                        # 重新獲取更新後的交互記錄
                        updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                        st.session_state.simulator_recommendations_data['user_interactions_df'] = updated_interactions_df
                        st.session_state.simulator_recommendations_data['watched_movie_ids'] = updated_watched_ids
                else:
                    output_container.error("❌ 添加失敗，請稍後再試")
    
    # 顯示用戶歷史交互記錄
    if not user_interactions_df.empty:
        output_container.subheader(f"📚 用戶 {target_user_id} 的歷史交互記錄")
        output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
        
        # 按時間戳降序排列
        sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
        
        # 顯示所有交互記錄
        output_container.dataframe(sorted_interactions, use_container_width=True)
        
        # 顯示統計信息
        output_container.info(f"📊 共 {len(sorted_interactions)} 條交互記錄")
    else:
        output_container.warning("該用戶沒有歷史交互記錄")

    # 添加社群推薦功能
    output_container.markdown("---")
    output_container.subheader("👥 社群推薦 - 看過類似電影的用戶推薦")
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 找到最相似的兩個用戶（使用傳入的用戶嵌入）
        similar_users, similarity_scores = find_similar_users(user_embeddings, target_user_id, num_similar_users=2)
        
        # 為每個相似用戶顯示推薦
        for i, (similar_user_id, similarity_score) in enumerate(zip(similar_users, similarity_scores)):
            # 獲取該用戶看過的高分電影
            watched_movies = get_user_watched_movies(similar_user_id, reverse_uid_map, reverse_mid_map, num_movies=5)
            
            if watched_movies:
                output_container.subheader(f"🎬 用戶 {similar_user_id} 跟你看過類似的電影，所以你也可能喜歡看這些電影")
                output_container.info(f"相似度: {similarity_score:.4f}")
                
                # 創建推薦表格
                col1, col2, col3, col4, col5 = output_container.columns([1, 1, 4, 3, 2])
                with col1:
                    output_container.write("**排名**")
                with col2:
                    output_container.write("**電影ID**")
                with col3:
                    output_container.write("**電影名稱**")
                with col4:
                    output_container.write("**類型**")
                with col5:
                    output_container.write("**用戶評分**")
                
                output_container.write("---")
                
                # 顯示推薦電影
                for j, movie_data in enumerate(watched_movies):
                    movie_id = movie_data['movie_id']
                    rating = movie_data['rating']
                    
                    movie_info = movies_info.get(movie_id, {})
                    movie_title = movie_info.get('title', '未知電影')
                    movie_genres = ' | '.join(movie_info.get('genres', ['未知']))
                    
                    col1, col2, col3, col4, col5 = output_container.columns([1, 1, 4, 3, 2])
                    
                    with col1:
                        output_container.write(f"**{j+1}**")
                    with col2:
                        output_container.write(f"{movie_id}")
                    with col3:
                        output_container.write(f"**{movie_title}**")
                    with col4:
                        output_container.write(f"{movie_genres}")
                    with col5:
                        output_container.write(f"⭐ {rating}")
                
                if i < len(similar_users) - 1:  # 如果不是最後一個用戶，添加分隔線
                    output_container.markdown("---")
            else:
                output_container.warning(f"用戶 {similar_user_id} 沒有足夠的觀看記錄")
        
        # 切換回原目錄
        os.chdir(original_dir)
        
    except Exception as e:
        output_container.error(f"社群推薦功能出錯: {str(e)}")
        # 確保切換回原目錄
        try:
            os.chdir(original_dir)
        except:
            pass

