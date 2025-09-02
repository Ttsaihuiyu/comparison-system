# model_adapters.py
import pandas as pd
import torch
import torch.nn as nn
import os
import pickle
import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# 直接在這裡定義所有需要的類和函數，不依賴外部導入！

# ====== 直接複製 LightGCNConv ======
# 使用條件導入避免 IDE 報錯
import importlib.util

TORCH_GEOMETRIC_AVAILABLE = importlib.util.find_spec("torch_geometric") is not None

if TORCH_GEOMETRIC_AVAILABLE:
    try:
        from torch_geometric.nn import MessagePassing
        
        class LightGCNConv(MessagePassing):
            def __init__(self): 
                super().__init__(aggr='add')
            
            def forward(self, x, edge_index):
                row, col = edge_index
                deg = torch.bincount(row, minlength=x.size(0)).float()
                deg_inv_sqrt = deg.pow(-0.5)
                deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
                norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
                return self.propagate(edge_index, x=x, norm=norm)
            
            def message(self, x_j, norm): 
                return norm.view(-1, 1) * x_j
        
        print("使用 torch_geometric 版本的 LightGCNConv")
        
    except ImportError:
        TORCH_GEOMETRIC_AVAILABLE = False

if not TORCH_GEOMETRIC_AVAILABLE:
    print("torch_geometric 未安裝，使用簡化版本")
    
    class LightGCNConv(nn.Module):
        """簡化版 LightGCNConv，不依賴 torch_geometric"""
        def __init__(self):
            super().__init__()
        
        def forward(self, x, edge_index):
            row, col = edge_index
            deg = torch.bincount(row, minlength=x.size(0)).float()
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
            # 手動實現消息傳播
            out = torch.zeros_like(x)
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                out[dst] += edge_weight[i] * x[src]
            
            return out

# ====== 直接複製 LightGCN ======
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64, n_layers=2):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.embedding = nn.Embedding(num_users + num_items, emb_size)
        nn.init.xavier_uniform_(self.embedding.weight)
        self.convs = nn.ModuleList([LightGCNConv() for _ in range(n_layers)])
    
    def forward(self, edge_index):
        x = self.embedding.weight
        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)
        return torch.stack(embs).mean(0)
    
    def get_user_item(self, edge_index):
        all_emb = self(edge_index)
        return all_emb[:self.num_users], all_emb[self.num_users:]

# ====== 直接複製 build_edge_index ======
def build_edge_index(pairs, num_users):
    """構建雙向邊索引"""
    edges = []
    for u, i in pairs:
        edges.append((u, i + num_users))
        edges.append((i + num_users, u))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()

# ====== 原有的全域變數和函數 ======
RL_RECOMMENDER_PATH = Path(__file__).parent.parent / "RL_recommender"
sys.path.append(str(RL_RECOMMENDER_PATH))

# Global variable to cache model and edge_index
_cached_model = None
_cached_edge_index = None
_cached_n_user = None
_cached_n_item = None

# Additional caches for new heuristic models
_cached_model_heur10 = None
_cached_edge_index_heur10 = None
_cached_model_heur20 = None
_cached_edge_index_heur20 = None

def initialize_model_for_dynamic_updates():
    global _cached_model, _cached_edge_index, _cached_n_user, _cached_n_item
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入映射信息
        with open('mapping/uid_map.pkl', 'rb') as f:
            uid_map = pickle.load(f)
        with open('mapping/mid_map.pkl', 'rb') as f:
            mid_map = pickle.load(f)
        
        n_user = len(uid_map)
        n_item = len(mid_map)
        
        # 載入 embeddings - 使用 heuristic 版本作為默認
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_emb = torch.load("model/RS/user_emb_heuristic.pt", map_location=device, weights_only=True)
        item_emb = torch.load("model/RS/item_emb_heuristic.pt", map_location=device, weights_only=True)
        
        # 現在用本文件中定義的 LightGCN！
        model = LightGCN(n_user, n_item, emb_size=64, n_layers=2).to(device)
        combined_emb = torch.cat([user_emb, item_emb], dim=0)
        model.embedding.weight.data = combined_emb
        
        # 現在用本文件中定義的 build_edge_index！
        train_df = pd.read_csv('data/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        interactions = [(row['user_id'], row['movie_id']) for _, row in train_df.iterrows()]
        edge_index = build_edge_index(interactions, n_user).to(device)
        
        # 儲存到全域變數
        _cached_model = model
        _cached_edge_index = edge_index
        _cached_n_user = n_user
        _cached_n_item = n_item
        
        os.chdir(original_dir)
        print(f"模型初始化完成 - 用戶: {n_user}, 物品: {n_item}, 邊數: {edge_index.shape[1]}")
        return True
        
    except Exception as e:
        print(f"模型初始化失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False

def initialize_heuristic_10epoch_model():
    global _cached_model_heur10, _cached_edge_index_heur10, _cached_n_user, _cached_n_item
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入映射信息
        with open('mapping/uid_map.pkl', 'rb') as f:
            uid_map = pickle.load(f)
        with open('mapping/mid_map.pkl', 'rb') as f:
            mid_map = pickle.load(f)
        
        n_user = len(uid_map)
        n_item = len(mid_map)
        
        # 載入 embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_emb = torch.load("model/RS/user_emb_heuristic.pt", map_location=device, weights_only=True)
        item_emb = torch.load("model/RS/item_emb_heuristic.pt", map_location=device, weights_only=True)
        
        # 創建 heuristic 10 epoch 模型
        model = LightGCN(n_user, n_item, emb_size=64, n_layers=2).to(device)
        combined_emb = torch.cat([user_emb, item_emb], dim=0)
        model.embedding.weight.data = combined_emb
        
        # 建立 edge index
        train_df = pd.read_csv('data/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        interactions = [(row['user_id'], row['movie_id']) for _, row in train_df.iterrows()]
        edge_index = build_edge_index(interactions, n_user).to(device)
        
        # 儲存到全域變數
        _cached_model_heur10 = model
        _cached_edge_index_heur10 = edge_index
        _cached_n_user = n_user
        _cached_n_item = n_item
        
        os.chdir(original_dir)
        print(f"Heuristic 10 Epoch 模型初始化完成 - 用戶: {n_user}, 物品: {n_item}, 邊數: {edge_index.shape[1]}")
        return True
        
    except Exception as e:
        print(f"Heuristic 10 Epoch 模型初始化失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False

def initialize_heuristic_20epoch_model():
    global _cached_model_heur20, _cached_edge_index_heur20, _cached_n_user, _cached_n_item
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入映射信息
        with open('mapping/uid_map.pkl', 'rb') as f:
            uid_map = pickle.load(f)
        with open('mapping/mid_map.pkl', 'rb') as f:
            mid_map = pickle.load(f)
        
        n_user = len(uid_map)
        n_item = len(mid_map)
        
        # 載入 embeddings
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_emb = torch.load("model/RS/user_emb_heuristic_20epoch.pt", map_location=device, weights_only=True)
        item_emb = torch.load("model/RS/item_emb_heuristic_20epoch.pt", map_location=device, weights_only=True)
        
        # 創建 heuristic 20 epoch 模型
        model = LightGCN(n_user, n_item, emb_size=64, n_layers=2).to(device)
        combined_emb = torch.cat([user_emb, item_emb], dim=0)
        model.embedding.weight.data = combined_emb
        
        # 建立 edge index
        train_df = pd.read_csv('data/train.dat', sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
        interactions = [(row['user_id'], row['movie_id']) for _, row in train_df.iterrows()]
        edge_index = build_edge_index(interactions, n_user).to(device)
        
        # 儲存到全域變數
        _cached_model_heur20 = model
        _cached_edge_index_heur20 = edge_index
        _cached_n_user = n_user
        _cached_n_item = n_item
        
        os.chdir(original_dir)
        print(f"Heuristic 20 Epoch 模型初始化完成 - 用戶: {n_user}, 物品: {n_item}, 邊數: {edge_index.shape[1]}")
        return True
        
    except Exception as e:
        print(f"Heuristic 20 Epoch 模型初始化失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        if 'original_dir' in locals():
            os.chdir(original_dir)
        return False

def update_embeddings_after_like(user_id, movie_id):
    """
    超簡化版：點讚後直接更新並重新計算
    """
    global _cached_model, _cached_edge_index, _cached_n_user
    
    if _cached_model is None:
        print("模型未初始化")
        return None, None
    
    try:
        # 添加新邊
        device = _cached_edge_index.device
        new_edges = torch.tensor([
            [user_id, movie_id + _cached_n_user],
            [movie_id + _cached_n_user, user_id]
        ], device=device).t()
        
        # 更新 edge_index
        updated_edge_index = torch.cat([_cached_edge_index, new_edges], dim=1)
        updated_edge_index = torch.unique(updated_edge_index, dim=1)
        
        # 您說的：直接執行這個 function！
        with torch.no_grad():
            new_user_emb, new_item_emb = _cached_model.get_user_item(updated_edge_index)
        
        # 更新快取
        _cached_edge_index = updated_edge_index
        
        print(f"直接執行 get_user_item() 完成！用戶{user_id}喜歡電影{movie_id}")
        return new_user_emb, new_item_emb
        
    except Exception as e:
        print(f"更新失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return None, None

def update_embeddings_after_like_heur10(user_id, movie_id):
    """
    Heuristic 10 Epoch 版本：點讚後直接更新並重新計算
    """
    global _cached_model_heur10, _cached_edge_index_heur10, _cached_n_user
    
    if _cached_model_heur10 is None:
        print("Heuristic 10 Epoch 模型未初始化")
        return None, None
    
    try:
        # 添加新邊
        device = _cached_edge_index_heur10.device
        new_edges = torch.tensor([
            [user_id, movie_id + _cached_n_user],
            [movie_id + _cached_n_user, user_id]
        ], device=device).t()
        
        # 更新 edge_index
        updated_edge_index = torch.cat([_cached_edge_index_heur10, new_edges], dim=1)
        updated_edge_index = torch.unique(updated_edge_index, dim=1)
        
        # 直接執行 get_user_item
        with torch.no_grad():
            new_user_emb, new_item_emb = _cached_model_heur10.get_user_item(updated_edge_index)
        
        # 更新快取
        _cached_edge_index_heur10 = updated_edge_index
        
        print(f"Heuristic 10 Epoch 直接執行 get_user_item() 完成！用戶{user_id}喜歡電影{movie_id}")
        return new_user_emb, new_item_emb
        
    except Exception as e:
        print(f"Heuristic 10 Epoch 更新失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return None, None

def update_embeddings_after_like_heur20(user_id, movie_id):
    """
    Heuristic 20 Epoch 版本：點讚後直接更新並重新計算
    """
    global _cached_model_heur20, _cached_edge_index_heur20, _cached_n_user
    
    if _cached_model_heur20 is None:
        print("Heuristic 20 Epoch 模型未初始化")
        return None, None
    
    try:
        # 添加新邊
        device = _cached_edge_index_heur20.device
        new_edges = torch.tensor([
            [user_id, movie_id + _cached_n_user],
            [movie_id + _cached_n_user, user_id]
        ], device=device).t()
        
        # 更新 edge_index
        updated_edge_index = torch.cat([_cached_edge_index_heur20, new_edges], dim=1)
        updated_edge_index = torch.unique(updated_edge_index, dim=1)
        
        # 直接執行 get_user_item
        with torch.no_grad():
            new_user_emb, new_item_emb = _cached_model_heur20.get_user_item(updated_edge_index)
        
        # 更新快取
        _cached_edge_index_heur20 = updated_edge_index
        
        print(f"Heuristic 20 Epoch 直接執行 get_user_item() 完成！用戶{user_id}喜歡電影{movie_id}")
        return new_user_emb, new_item_emb
        
    except Exception as e:
        print(f"Heuristic 20 Epoch 更新失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return None, None

def get_dynamic_recommendations(user_id, num_recommendations=20, exclude_ids=None):
    """
    使用動態更新的 embeddings 生成推薦
    """
    # 先嘗試獲取更新後的 embeddings
    user_emb, item_emb = update_embeddings_after_like(user_id, 0)  # dummy call to get current embeddings
    
    if user_emb is None:
        return None, None
    
    with torch.no_grad():
        user_vec = user_emb[user_id].unsqueeze(0)
        scores = torch.mm(user_vec, item_emb.t()).squeeze()
        
        if exclude_ids:
            scores[exclude_ids] = -float('inf')
        
        top_scores, top_items = torch.topk(scores, num_recommendations)
        return top_items.cpu().numpy(), top_scores.cpu().numpy()

def get_dynamic_recommendations_heur10(user_id, num_recommendations=20, exclude_ids=None):
    """
    使用 Heuristic 10 Epoch 動態更新的 embeddings 生成推薦
    """
    # 先嘗試獲取更新後的 embeddings
    user_emb, item_emb = update_embeddings_after_like_heur10(user_id, 0)  # dummy call to get current embeddings
    
    if user_emb is None:
        return None, None
    
    with torch.no_grad():
        user_vec = user_emb[user_id].unsqueeze(0)
        scores = torch.mm(user_vec, item_emb.t()).squeeze()
        
        if exclude_ids:
            scores[exclude_ids] = -float('inf')
        
        top_scores, top_items = torch.topk(scores, num_recommendations)
        return top_items.cpu().numpy(), top_scores.cpu().numpy()

def get_dynamic_recommendations_heur20(user_id, num_recommendations=20, exclude_ids=None):
    """
    使用 Heuristic 20 Epoch 動態更新的 embeddings 生成推薦
    """
    # 先嘗試獲取更新後的 embeddings
    user_emb, item_emb = update_embeddings_after_like_heur20(user_id, 0)  # dummy call to get current embeddings
    
    if user_emb is None:
        return None, None
    
    with torch.no_grad():
        user_vec = user_emb[user_id].unsqueeze(0)
        scores = torch.mm(user_vec, item_emb.t()).squeeze()
        
        if exclude_ids:
            scores[exclude_ids] = -float('inf')
        
        top_scores, top_items = torch.topk(scores, num_recommendations)
        return top_items.cpu().numpy(), top_scores.cpu().numpy()

def load_movie_info():
    try:
        movies_df = pd.read_csv(RL_RECOMMENDER_PATH / "raw" / "ml-1m" / "movies.dat", 
                               sep='::', names=['Movie_ID', 'Title', 'Genres'], 
                               engine='python', encoding='latin-1')
        movies_info = {}
        for _, row in movies_df.iterrows():
            genres = row['Genres'].split('|') if pd.notna(row['Genres']) else ['Unknown']
            movies_info[row['Movie_ID']] = {
                'title': row['Title'],
                'genres': genres
            }
        return movies_info
    except Exception as e:
        print(f"Error loading movie info: {e}")
        return {}

def load_user_info():
    try:
        users_df = pd.read_csv(RL_RECOMMENDER_PATH / "raw" / "ml-1m" / "users.dat", 
                              sep='::', names=['User_ID', 'Gender', 'Age', 'Occupation', 'Zip'], 
                              engine='python')
        users_info = {}
        for _, row in users_df.iterrows():
            users_info[row['User_ID']] = {
                'gender': row['Gender'],
                'age': row['Age'],
                'occupation': row['Occupation'],
                'zip': row['Zip']
            }
        return users_info
    except Exception as e:
        print(f"Error loading user info: {e}")
        return {}

def load_mapping_files():
    try:
        uid_map_file = RL_RECOMMENDER_PATH / "mapping" / "uid_map.pkl"
        mid_map_file = RL_RECOMMENDER_PATH / "mapping" / "mid_map.pkl"
        
        with open(uid_map_file, 'rb') as f:
            uid_map = pickle.load(f)
        with open(mid_map_file, 'rb') as f:
            mid_map = pickle.load(f)
        
        reverse_uid_map = {v: k for k, v in uid_map.items()}
        reverse_mid_map = {v: k for k, v in mid_map.items()}
        
        return uid_map, mid_map, reverse_uid_map, reverse_mid_map, True
    except Exception as e:
        print(f"Error loading mapping files: {e}")
        return None, None, None, None, False

def get_user_recommendations(user_embeddings, item_embeddings, user_id, num_recommendations=20, exclude_ids=None):
    """基本推薦功能"""
    with torch.no_grad():
        user_emb = user_embeddings[user_id].unsqueeze(0)
        scores = torch.mm(user_emb, item_embeddings.t()).squeeze()
        
        if exclude_ids and isinstance(exclude_ids, list) and len(exclude_ids) > 0:
            scores[exclude_ids] = -float('inf')
        
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
        user_embeddings = torch.load("model/RS/user_emb_heuristic.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_heuristic.pt", map_location=device, weights_only=True)
        
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
                output_container.subheader(f"用戶 {target_user_id} 的詳細信息")
                # 使用表格形式確保完整顯示
                import pandas as pd
                user_data = pd.DataFrame({
                    '性別': [user_info['gender']],
                    '年齡': [user_info['age']],
                    '職業': [user_info['occupation']],
                    '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
                })
                output_container.dataframe(user_data, use_container_width=True, hide_index=True)

            output_container.subheader(f"為用戶 {target_user_id} 的 Heuristic 推薦結果")
            
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
                    if output_container.button("加入最愛", key=f"heart_{target_user_id}_{i}", help="加入我的最愛"):
                        # 添加到用戶交互記錄，直接使用原始電影ID
                        success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                        
                        if success:
                            output_container.success(f"已將《{movie_title}》添加到交互記錄！")
                            # 更新 session_state 中的交互記錄
                            import streamlit as st
                            if 'recommendations_data' in st.session_state:
                                # 重新獲取更新後的交互記錄
                                updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                                st.session_state.recommendations_data['user_interactions_df'] = updated_interactions_df
                                st.session_state.recommendations_data['watched_movie_ids'] = updated_watched_ids
                        else:
                            output_container.error("添加失敗，請稍後再試")
            
            # 顯示用戶歷史交互記錄
            if not user_interactions_df.empty:
                output_container.subheader(f"用戶 {target_user_id} 的歷史交互記錄")
                output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
                
                # 按時間戳降序排列
                sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
                
                # 顯示所有交互記錄
                output_container.dataframe(sorted_interactions, use_container_width=True)
                
                # 顯示統計信息
                output_container.info(f"共 {len(sorted_interactions)} 條交互記錄")
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
        
        # 檢查所有模型文件（四組完整的嵌入文件）
        user_emb_heur_file = RL_RECOMMENDER_PATH / "model" / "RS" / "user_emb_heuristic.pt"
        item_emb_heur_file = RL_RECOMMENDER_PATH / "model" / "RS" / "item_emb_heuristic.pt"
        user_emb_heur20_file = RL_RECOMMENDER_PATH / "model" / "RS" / "user_emb_heuristic_20epoch.pt"
        item_emb_heur20_file = RL_RECOMMENDER_PATH / "model" / "RS" / "item_emb_heuristic_20epoch.pt"
        user_emb_raw_file = RL_RECOMMENDER_PATH / "model" / "RS" / "user_emb_raw.pt"
        item_emb_raw_file = RL_RECOMMENDER_PATH / "model" / "RS" / "item_emb_raw.pt"
        user_emb_raw20_file = RL_RECOMMENDER_PATH / "model" / "RS" / "user_emb_raw_20epoch.pt"
        item_emb_raw20_file = RL_RECOMMENDER_PATH / "model" / "RS" / "item_emb_raw_20epoch.pt"
        
        missing_files = []
        if not user_emb_heur_file.exists():
            missing_files.append(f"用戶 Heuristic 嵌入文件: {user_emb_heur_file}")
        if not item_emb_heur_file.exists():
            missing_files.append(f"項目 Heuristic 嵌入文件: {item_emb_heur_file}")
        if not user_emb_heur20_file.exists():
            missing_files.append(f"用戶 Heuristic 20 Epoch 嵌入文件: {user_emb_heur20_file}")
        if not item_emb_heur20_file.exists():
            missing_files.append(f"項目 Heuristic 20 Epoch 嵌入文件: {item_emb_heur20_file}")
        if not user_emb_raw_file.exists():
            missing_files.append(f"用戶 Raw 嵌入文件: {user_emb_raw_file}")
        if not item_emb_raw_file.exists():
            missing_files.append(f"項目 Raw 嵌入文件: {item_emb_raw_file}")
        if not user_emb_raw20_file.exists():
            missing_files.append(f"用戶 Raw 20 Epoch 嵌入文件: {user_emb_raw20_file}")
        if not item_emb_raw20_file.exists():
            missing_files.append(f"項目 Raw 20 Epoch 嵌入文件: {item_emb_raw20_file}")
            
        if missing_files:
            return False, f"以下嵌入文件不存在: {'; '.join(missing_files)}"
        
        return True, "所有依賴檢查通過"
        
    except Exception as e:
        return False, f"依賴檢查出錯: {str(e)}"

def get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map):
    try:
        # 1️⃣ 首先嘗試讀取用戶專門的交互記錄文件（向後兼容）
        original_user_id = reverse_uid_map.get(target_user_id, target_user_id)
        interaction_file = RL_RECOMMENDER_PATH / "interaction_collect" / f"user_{original_user_id}_interactions.csv"
        
        if interaction_file.exists():
            user_interactions_df = pd.read_csv(interaction_file)
            watched_movie_ids = []
            for movie_id in user_interactions_df['Movie_ID']:
                mapped_id = None
                for mapped, original in reverse_mid_map.items():
                    if original == movie_id:
                        mapped_id = mapped
                        break
                if mapped_id is not None:
                    watched_movie_ids.append(mapped_id)
            
            return user_interactions_df, watched_movie_ids
        
        # 2️⃣ 直接從 /data 目錄讀取歷史交互資料
        else:
            print(f"從 /data 目錄讀取用戶 {target_user_id} 的歷史交互記錄")
            
            # 讀取所有數據文件
            data_files = ['train.dat', 'val.dat', 'test.dat']
            all_interactions = []
            
            for data_file in data_files:
                file_path = RL_RECOMMENDER_PATH / "data" / data_file
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path, sep=',', names=['user_id', 'movie_id', 'rating', 'timestamp'])
                        user_data = df[df['user_id'] == target_user_id]
                        if not user_data.empty:
                            all_interactions.append(user_data)
                            print(f"從 {data_file} 中找到 {len(user_data)} 條記錄")
                    except Exception as e:
                        print(f"讀取 {data_file} 失敗: {str(e)}")
            
            if not all_interactions:
                print(f"用戶 {target_user_id} 沒有任何歷史交互記錄")
                return pd.DataFrame(columns=['Movie_ID', 'Title', 'Genres', 'Rating', 'Timestamp']), []
            
            # 合併所有交互記錄
            combined_interactions = pd.concat(all_interactions, ignore_index=True)
            
            # 載入電影信息用於創建詳細記錄
            movies_info = load_movie_info()
            
            # 轉換為用戶交互格式
            user_interactions_data = []
            watched_movie_ids = []
            
            for _, row in combined_interactions.iterrows():
                mapped_movie_id = row['movie_id']  # 已經是映射後的ID
                
                # 轉換回原始電影ID用於顯示
                original_movie_id = reverse_mid_map.get(mapped_movie_id)
                if original_movie_id is not None:
                    movie_info = movies_info.get(original_movie_id, {})
                    
                    user_interactions_data.append({
                        'Movie_ID': original_movie_id,
                        'Title': movie_info.get('title', '未知電影'),
                        'Genres': ' | '.join(movie_info.get('genres', ['未知'])),
                        'Rating': row['rating'],
                        'Timestamp': row['timestamp']
                    })
                    
                    # 添加到觀看過的電影列表（使用映射後的ID）
                    watched_movie_ids.append(mapped_movie_id)
            
            user_interactions_df = pd.DataFrame(user_interactions_data)
            print(f"用戶 {target_user_id} 總共有 {len(user_interactions_df)} 條歷史交互記錄")
            
            return user_interactions_df, watched_movie_ids
            
    except Exception as e:
        print(f"獲取用戶交互記錄失敗: {str(e)}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        return pd.DataFrame(columns=['Movie_ID', 'Title', 'Genres', 'Rating', 'Timestamp']), []

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
    try:
        print(f"開始處理: 用戶{user_id}, 原始電影ID{original_movie_id}")
        
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        print(f"切換到目錄: {RL_RECOMMENDER_PATH}")
        
        train_file = "data/train.dat"
        
        mid_map_file = RL_RECOMMENDER_PATH / "mapping" / "mid_map.pkl"
        with open(mid_map_file, 'rb') as f:
            mid_map = pickle.load(f)
        
        mapped_movie_id = mid_map.get(original_movie_id)
        if mapped_movie_id is None:
            print(f"原始電影ID {original_movie_id} 不在映射中")
            os.chdir(original_dir)
            return False
        
        print(f"電影ID映射成功: 原始ID{original_movie_id} -> 映射ID{mapped_movie_id}")
        
        current_timestamp = int(datetime.now().timestamp())
        new_interaction = f"{user_id},{mapped_movie_id},5,{current_timestamp}\n"
        print(f"新交互記錄: {new_interaction.strip()}")
        
        with open(train_file, 'a', encoding='utf-8') as f:
            f.write(new_interaction)
        print(f"已添加到 {train_file}")
        
        # 超簡化版：直接調用 get_user_item()
        try:
           
            new_user_emb, new_item_emb = update_embeddings_after_like(user_id, mapped_movie_id)
            if new_user_emb is not None:
                print(f"原始 Heuristic 直接執行 get_user_item() 成功！")
            
            # 更新 10 epoch 模型
            new_user_emb_10, new_item_emb_10 = update_embeddings_after_like_heur10(user_id, mapped_movie_id)
            if new_user_emb_10 is not None:
                print(f"Heuristic 10 Epoch 直接執行 get_user_item() 成功！")
            
            # 更新 20 epoch 模型
            new_user_emb_20, new_item_emb_20 = update_embeddings_after_like_heur20(user_id, mapped_movie_id)
            if new_user_emb_20 is not None:
                print(f"Heuristic 20 Epoch 直接執行 get_user_item() 成功！")
            
            if new_user_emb is None and new_user_emb_10 is None and new_user_emb_20 is None:
                print(f"所有動態更新失敗，但數據已保存")
        except Exception as embed_error:
            print(f"動態更新遇到問題: {str(embed_error)}")
        
        interaction_file = f"interaction_collect/user_{user_id}_interactions.csv"
        
        new_record = pd.DataFrame({
            'Movie_ID': [original_movie_id],
            'Title': [movie_info.get('title', '未知電影')],
            'Genres': [' | '.join(movie_info.get('genres', ['未知']))],
            'Rating': [5],
            'Timestamp': [current_timestamp]
        })
        
        if os.path.exists(interaction_file):
            existing_df = pd.read_csv(interaction_file)
            if original_movie_id not in existing_df['Movie_ID'].values:
                updated_df = pd.concat([new_record, existing_df], ignore_index=True)
                updated_df.to_csv(interaction_file, index=False)
                print(f"已更新用戶交互文件: {interaction_file}")
            else:
                print(f"電影已存在於用戶交互記錄中")
        else:
            new_record.to_csv(interaction_file, index=False)
            print(f"已創建新的用戶交互文件: {interaction_file}")
        
        os.chdir(original_dir)
        print(f"處理完成，返回成功")
        return True
        
    except Exception as e:
        print(f"添加電影到交互記錄失敗: {str(e)}")
        print(f"錯誤類型: {type(e).__name__}")
        import traceback
        print(f"詳細錯誤: {traceback.format_exc()}")
        os.chdir(original_dir)
        return False

def run_heuristic_10epoch_exposure(output_container=None, target_user_id=None, num_recommendations=20):
    """
    運行 Heuristic 10 Epoch 模型 - 使用 user embedding 和 10 epoch item embedding
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
        item_embeddings = torch.load("model/RS/item_emb_heuristic.pt", map_location=device, weights_only=True)
        
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
            output_container.success("Heuristic 10 Epoch 推薦完成！")
            
            # 顯示用戶信息
            user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
            if user_info:
                output_container.subheader(f"用戶 {target_user_id} 的詳細信息")
                # 使用表格形式確保完整顯示
                import pandas as pd
                user_data = pd.DataFrame({
                    '性別': [user_info['gender']],
                    '年齡': [user_info['age']],
                    '職業': [user_info['occupation']],
                    '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
                })
                output_container.dataframe(user_data, use_container_width=True, hide_index=True)

            output_container.subheader(f"為用戶 {target_user_id} 的 Heuristic 10 Epoch 推薦結果")
            
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
                    if output_container.button("加入最愛", key=f"heur10_heart_{target_user_id}_{i}", help="加入我的最愛"):
                        # 添加到用戶交互記錄，直接使用原始電影ID
                        success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                        
                        if success:
                            output_container.success(f"已將《{movie_title}》添加到交互記錄！")
                            # 更新 session_state 中的交互記錄
                            import streamlit as st
                            if 'heuristic_10epoch_data' in st.session_state:
                                # 重新獲取更新後的交互記錄
                                updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                                st.session_state.heuristic_10epoch_data['user_interactions_df'] = updated_interactions_df
                                st.session_state.heuristic_10epoch_data['watched_movie_ids'] = updated_watched_ids
                        else:
                            output_container.error("添加失敗，請稍後再試")
            
            # 顯示用戶歷史交互記錄
            if not user_interactions_df.empty:
                output_container.subheader(f"用戶 {target_user_id} 的歷史交互記錄")
                output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
                
                # 按時間戳降序排列
                sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
                
                # 顯示所有交互記錄
                output_container.dataframe(sorted_interactions, use_container_width=True)
                
                # 顯示統計信息
                output_container.info(f"共 {len(sorted_interactions)} 條交互記錄")
            else:
                output_container.warning("該用戶沒有歷史交互記錄")
        
        return f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"Heuristic 10 Epoch 推薦執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        os.chdir(original_dir)
        return error_msg

def run_heuristic_20epoch_exposure(output_container=None, target_user_id=None, num_recommendations=20):
    """
    運行 Heuristic 20 Epoch 模型 - 使用 user embedding 和 20 epoch item embedding
    """
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb_heuristic_20epoch.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_heuristic_20epoch.pt", map_location=device, weights_only=True)
        
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
            output_container.success("Heuristic 20 Epoch 推薦完成！")
            
            # 顯示用戶信息
            user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
            if user_info:
                output_container.subheader(f"用戶 {target_user_id} 的詳細信息")
                # 使用表格形式確保完整顯示
                import pandas as pd
                user_data = pd.DataFrame({
                    '性別': [user_info['gender']],
                    '年齡': [user_info['age']],
                    '職業': [user_info['occupation']],
                    '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
                })
                output_container.dataframe(user_data, use_container_width=True, hide_index=True)

            output_container.subheader(f"為用戶 {target_user_id} 的 Heuristic 20 Epoch 推薦結果")
            
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
                    if output_container.button("加入最愛", key=f"heur20_heart_{target_user_id}_{i}", help="加入我的最愛"):
                        # 添加到用戶交互記錄，直接使用原始電影ID
                        success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                        
                        if success:
                            output_container.success(f"已將《{movie_title}》添加到交互記錄！")
                            # 更新 session_state 中的交互記錄
                            import streamlit as st
                            if 'heuristic_20epoch_data' in st.session_state:
                                # 重新獲取更新後的交互記錄
                                updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                                st.session_state.heuristic_20epoch_data['user_interactions_df'] = updated_interactions_df
                                st.session_state.heuristic_20epoch_data['watched_movie_ids'] = updated_watched_ids
                        else:
                            output_container.error("添加失敗，請稍後再試")
            
            # 顯示用戶歷史交互記錄
            if not user_interactions_df.empty:
                output_container.subheader(f"用戶 {target_user_id} 的歷史交互記錄")
                output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
                
                # 按時間戳降序排列
                sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
                
                # 顯示所有交互記錄
                output_container.dataframe(sorted_interactions, use_container_width=True)
                
                # 顯示統計信息
                output_container.info(f"共 {len(sorted_interactions)} 條交互記錄")
            else:
                output_container.warning("該用戶沒有歷史交互記錄")
        
        return f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"Heuristic 20 Epoch 推薦執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        os.chdir(original_dir)
        return error_msg

def get_recommendations_data(target_user_id, num_recommendations=20):
    try:
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb_heuristic.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_heuristic.pt", map_location=device, weights_only=True)
        
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            os.chdir(original_dir)
            return None, "無法載入映射文件"
        
        # 簡化版：初始化模型（隊友建議的功能）
        try:
            if initialize_model_for_dynamic_updates():
                print("模型初始化成功，支援動態更新")
            else:
                print("模型初始化失敗，使用原始方法")
                
        except Exception as cache_error:
            print(f"模型初始化失敗: {str(cache_error)}")
        
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            os.chdir(original_dir)
            return None, error_msg
        
        # 嘗試使用動態更新的 embeddings
        try:
            recommended_items, scores = get_dynamic_recommendations(
                target_user_id, num_recommendations, exclude_ids=watched_movie_ids
            )
            if recommended_items is not None:
                print("使用動態更新的 embeddings 生成推薦")
            else:
                raise Exception("動態推薦失敗")
        except:
            recommended_items, scores = get_user_recommendations(
                user_embeddings, item_embeddings, target_user_id, num_recommendations,
                exclude_ids=watched_movie_ids
            )
            print("使用原始 embeddings 生成推薦")
        
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
            'user_embeddings': user_embeddings
        }
        
        os.chdir(original_dir)
        return recommendations_data, "成功生成推薦"
        
    except Exception as e:
        error_msg = f"推薦生成出錯: {str(e)}"
        os.chdir(original_dir)
        return None, error_msg

def get_heuristic_10epoch_recommendations_data(target_user_id, num_recommendations=20):
    """
    獲取 Heuristic 10 Epoch 模型的推薦數據，但不顯示
    """
    try:
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_heuristic.pt", map_location=device, weights_only=True)
        
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            os.chdir(original_dir)
            return None, "無法載入映射文件"
        
        # 初始化 Heuristic 10 Epoch 模型
        try:
            if initialize_heuristic_10epoch_model():
                print("Heuristic 10 Epoch 模型初始化成功，支援動態更新")
            else:
                print("Heuristic 10 Epoch 模型初始化失敗，使用原始方法")
        except Exception as cache_error:
            print(f"Heuristic 10 Epoch 模型初始化失敗: {str(cache_error)}")
        
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            os.chdir(original_dir)
            return None, error_msg
        
        # 嘗試使用動態更新的 embeddings
        try:
            recommended_items, scores = get_dynamic_recommendations_heur10(
                target_user_id, num_recommendations, exclude_ids=watched_movie_ids
            )
            if recommended_items is not None:
                print("使用 Heuristic 10 Epoch 動態更新的 embeddings 生成推薦")
            else:
                raise Exception("Heuristic 10 Epoch 動態推薦失敗")
        except:
            recommended_items, scores = get_user_recommendations(
                user_embeddings, item_embeddings, target_user_id, num_recommendations,
                exclude_ids=watched_movie_ids
            )
            print("使用 Heuristic 10 Epoch 原始 embeddings 生成推薦")
        
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
            'user_embeddings': user_embeddings
        }
        
        os.chdir(original_dir)
        return recommendations_data, "成功生成推薦"
        
    except Exception as e:
        error_msg = f"Heuristic 10 Epoch 推薦生成出錯: {str(e)}"
        os.chdir(original_dir)
        return None, error_msg

def get_heuristic_20epoch_recommendations_data(target_user_id, num_recommendations=20):
    """
    獲取 Heuristic 20 Epoch 模型的推薦數據，但不顯示
    """
    try:
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb_heuristic_20epoch.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_heuristic_20epoch.pt", map_location=device, weights_only=True)
        
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            os.chdir(original_dir)
            return None, "無法載入映射文件"
        
        # 初始化 Heuristic 20 Epoch 模型
        try:
            if initialize_heuristic_20epoch_model():
                print("Heuristic 20 Epoch 模型初始化成功，支援動態更新")
            else:
                print("Heuristic 20 Epoch 模型初始化失敗，使用原始方法")
        except Exception as cache_error:
            print(f"Heuristic 20 Epoch 模型初始化失敗: {str(cache_error)}")
        
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            os.chdir(original_dir)
            return None, error_msg
        
        # 嘗試使用動態更新的 embeddings
        try:
            recommended_items, scores = get_dynamic_recommendations_heur20(
                target_user_id, num_recommendations, exclude_ids=watched_movie_ids
            )
            if recommended_items is not None:
                print("使用 Heuristic 20 Epoch 動態更新的 embeddings 生成推薦")
            else:
                raise Exception("Heuristic 20 Epoch 動態推薦失敗")
        except:
            recommended_items, scores = get_user_recommendations(
                user_embeddings, item_embeddings, target_user_id, num_recommendations,
                exclude_ids=watched_movie_ids
            )
            print("使用 Heuristic 20 Epoch 原始 embeddings 生成推薦")
        
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
            'user_embeddings': user_embeddings
        }
        
        os.chdir(original_dir)
        return recommendations_data, "成功生成推薦"
        
    except Exception as e:
        error_msg = f"Heuristic 20 Epoch 推薦生成出錯: {str(e)}"
        os.chdir(original_dir)
        return None, error_msg

def run_raw_20epoch_exposure(output_container=None, target_user_id=None, num_recommendations=20):
    """
    運行 Raw 20 Epoch 模型 - 使用 raw 20 epoch embeddings
    """
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 載入用戶和電影信息
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb_raw_20epoch.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_raw_20epoch.pt", map_location=device, weights_only=True)
        
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
            output_container.success("Raw 20 Epoch 推薦完成！")
            
            # 顯示用戶信息
            user_info = users_info.get(target_user_id + 1, {})  # 用戶ID從1開始
            if user_info:
                output_container.subheader(f"用戶 {target_user_id} 的詳細信息")
                # 使用表格形式確保完整顯示
                import pandas as pd
                user_data = pd.DataFrame({
                    '性別': [user_info['gender']],
                    '年齡': [user_info['age']],
                    '職業': [user_info['occupation']],
                    '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
                })
                output_container.dataframe(user_data, use_container_width=True, hide_index=True)

            output_container.subheader(f"為用戶 {target_user_id} 的 Raw 20 Epoch 推薦結果")
            
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
                    if output_container.button("加入最愛", key=f"raw20_heart_{target_user_id}_{i}", help="加入我的最愛"):
                        # 添加到用戶交互記錄，直接使用原始電影ID
                        success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                        
                        if success:
                            output_container.success(f"已將《{movie_title}》添加到交互記錄！")
                            # 更新 session_state 中的交互記錄
                            import streamlit as st
                            if 'raw_20epoch_data' in st.session_state:
                                # 重新獲取更新後的交互記錄
                                updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                                st.session_state.raw_20epoch_data['user_interactions_df'] = updated_interactions_df
                                st.session_state.raw_20epoch_data['watched_movie_ids'] = updated_watched_ids
                        else:
                            output_container.error("添加失敗，請稍後再試")
            
            # 顯示用戶歷史交互記錄
            if not user_interactions_df.empty:
                output_container.subheader(f"用戶 {target_user_id} 的歷史交互記錄")
                output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
                
                # 按時間戳降序排列
                sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
                
                # 顯示所有交互記錄
                output_container.dataframe(sorted_interactions, use_container_width=True)
                
                # 顯示統計信息
                output_container.info(f"共 {len(sorted_interactions)} 條交互記錄")
            else:
                output_container.warning("該用戶沒有歷史交互記錄")
        
        return f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"Raw 20 Epoch 推薦執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        os.chdir(original_dir)
        return error_msg

def get_raw_20epoch_recommendations_data(target_user_id, num_recommendations=20):
    """
    獲取 Raw 20 Epoch 模型的推薦數據，但不顯示
    """
    try:
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        movies_info = load_movie_info()
        users_info = load_user_info()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        user_embeddings = torch.load("model/RS/user_emb_raw_20epoch.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_raw_20epoch.pt", map_location=device, weights_only=True)
        
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            os.chdir(original_dir)
            return None, "無法載入映射文件"
        
        user_interactions_df, watched_movie_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
        
        if target_user_id >= user_embeddings.shape[0] or target_user_id < 0:
            error_msg = f"用戶ID {target_user_id} 超出範圍 (0-{user_embeddings.shape[0]-1})"
            os.chdir(original_dir)
            return None, error_msg
        
        # Raw 模型不需要動態更新，直接使用原始 embeddings
        recommended_items, scores = get_user_recommendations(
            user_embeddings, item_embeddings, target_user_id, num_recommendations,
            exclude_ids=watched_movie_ids
        )
        
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
            'user_embeddings': user_embeddings
        }
        
        os.chdir(original_dir)
        return recommendations_data, "成功生成推薦"
        
    except Exception as e:
        error_msg = f"Raw 20 Epoch 推薦生成出錯: {str(e)}"
        os.chdir(original_dir)
        return None, error_msg

def get_simulator_recommendations_data(target_user_id, num_recommendations=20):
    """
    獲取 Simulator 模型的推薦數據，但不顯示
    """
    data, msg = run_simulator_exposure(
        target_user_id=target_user_id, 
        num_recommendations=num_recommendations
    )
    if data is None:
        return None, msg
    
    movies_info = load_movie_info()
    users_info = load_user_info()
    data['movies_info'] = movies_info
    data['users_info'] = users_info
    
    return data, msg

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
        output_container.subheader(f"用戶 {target_user_id} 的詳細信息")
        
        # 職業代碼到名稱的映射
        occupation_map = {
            0: "其他", 1: "學術/教育", 2: "藝術家", 3: "行政", 4: "大學/研究生",
            5: "客戶服務", 6: "醫生/醫療保健", 7: "高階主管/管理", 8: "農夫", 9: "家庭主婦",
            10: "K-12 學生", 11: "律師", 12: "程式設計師", 13: "退休", 14: "銷售/市場行銷",
            15: "科學家", 16: "自僱人士", 17: "技術員/工程師", 18: "技工/工匠",
            19: "失業", 20: "作家"
        }
        occupation_name = occupation_map.get(user_info['occupation'], "未知")

        # 使用表格形式確保完整顯示
        import pandas as pd
        user_data = pd.DataFrame({
            '性別': [user_info['gender']],
            '年齡': [user_info['age']],
            '職業': [occupation_name],
            '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
        })
        output_container.dataframe(user_data, use_container_width=True, hide_index=True)

    output_container.subheader(f"為用戶 {target_user_id} 的 Heuristic 推薦結果")
    
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
            if output_container.button("加入最愛", key=f"heart_{target_user_id}_{i}", help="加入我的最愛"):
                # 添加到用戶交互記錄，直接使用原始電影ID
                success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                
                if success:
                    output_container.success(f"已將《{movie_title}》添加到交互記錄！")
                    # 更新 session_state 中的交互記錄
                    import streamlit as st
                    if 'recommendations_data' in st.session_state:
                        # 重新獲取更新後的交互記錄
                        updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                        st.session_state.recommendations_data['user_interactions_df'] = updated_interactions_df
                        st.session_state.recommendations_data['watched_movie_ids'] = updated_watched_ids
                else:
                    output_container.error("添加失敗，請稍後再試")
    
    # 顯示用戶歷史交互記錄
    if not user_interactions_df.empty:
        output_container.subheader(f"用戶 {target_user_id} 的歷史交互記錄")
        output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
        
        # 按時間戳降序排列
        sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
        
        # 顯示所有交互記錄
        output_container.dataframe(sorted_interactions, use_container_width=True)
        
        # 顯示統計信息
        output_container.info(f"共 {len(sorted_interactions)} 條交互記錄")
    else:
        output_container.warning("該用戶沒有歷史交互記錄")

    # 添加社群推薦功能
    output_container.markdown("---")
    output_container.subheader("社群推薦 - 看過類似電影的用戶推薦")
    
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
                output_container.subheader(f"用戶 {similar_user_id} 跟你看過類似的電影，所以你也可能喜歡看這些電影")
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
                        output_container.write(f"{rating} 星")
                
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

def run_simulator_exposure(target_user_id=None, num_recommendations=20):
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
        # 使用 RS 目錄中的原始 embeddings
        user_embeddings = torch.load("model/RS/user_emb_raw.pt", map_location=device, weights_only=True)
        item_embeddings = torch.load("model/RS/item_emb_raw.pt", map_location=device, weights_only=True)
        
        # 载入映射文件
        uid_map, mid_map, reverse_uid_map, reverse_mid_map, mapping_success = load_mapping_files()
        
        if not mapping_success:
            error_msg = "無法載入映射文件"
            os.chdir(original_dir)
            return None, error_msg
        
        # 獲取用戶歷史交互記錄，以便過濾推薦
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
        
        # 切換回原目錄
        os.chdir(original_dir)
        
        return {
            'user_id': target_user_id,
            'recommended_items': recommended_items,
            'scores': scores,
            'user_interactions_df': user_interactions_df,
            'watched_movie_ids': watched_movie_ids,
            'reverse_uid_map': reverse_uid_map,
            'reverse_mid_map': reverse_mid_map,
            'user_embeddings': user_embeddings
        }, f"成功為用戶 {target_user_id} 生成了 {len(recommended_items)} 部推薦電影"
        
    except Exception as e:
        error_msg = f"Simulator 推薦執行出錯: {str(e)}"
        # 確保切換回原目錄
        try:
            os.chdir(original_dir)
        except NameError: # original_dir可能尚未定義
            pass
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
        output_container.subheader(f"用戶 {target_user_id} 的詳細信息")
        
        # 職業代碼到名稱的映射
        occupation_map = {
            0: "其他", 1: "學術/教育", 2: "藝術家", 3: "行政", 4: "大學/研究生",
            5: "客戶服務", 6: "醫生/醫療保健", 7: "高階主管/管理", 8: "農夫", 9: "家庭主婦",
            10: "K-12 學生", 11: "律師", 12: "程式設計師", 13: "退休", 14: "銷售/市場行銷",
            15: "科學家", 16: "自僱人士", 17: "技術員/工程師", 18: "技工/工匠",
            19: "失業", 20: "作家"
        }
        occupation_name = occupation_map.get(user_info['occupation'], "未知")

        # 使用表格形式確保完整顯示
        import pandas as pd
        user_data = pd.DataFrame({
            '性別': [user_info['gender']],
            '年齡': [user_info['age']],
            '職業': [occupation_name],
            '歷史交互': [f"{len(watched_movie_ids)} 部電影"]
        })
        output_container.dataframe(user_data, use_container_width=True, hide_index=True)

    output_container.subheader(f"為用戶 {target_user_id} 的 Simulator 推薦結果")
    
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
            if output_container.button("加入最愛", key=f"sim_heart_{target_user_id}_{i}", help="加入我的最愛"):
                # 添加到用戶交互記錄，直接使用原始電影ID
                success = add_movie_to_interactions(target_user_id, original_movie_id, movie_info, reverse_uid_map, reverse_mid_map)
                
                if success:
                    output_container.success(f"已將《{movie_title}》添加到交互記錄！")
                    # 更新 session_state 中的交互記錄
                    import streamlit as st
                    if 'simulator_recommendations_data' in st.session_state:
                        # 重新獲取更新後的交互記錄
                        updated_interactions_df, updated_watched_ids = get_user_interactions(target_user_id, reverse_uid_map, reverse_mid_map)
                        st.session_state.simulator_recommendations_data['user_interactions_df'] = updated_interactions_df
                        st.session_state.simulator_recommendations_data['watched_movie_ids'] = updated_watched_ids
                else:
                    output_container.error("添加失敗，請稍後再試")
    
    # 顯示用戶歷史交互記錄
    if not user_interactions_df.empty:
        output_container.subheader(f"用戶 {target_user_id} 的歷史交互記錄")
        output_container.info(f"數據已保存至: user_{target_user_id}_interactions.csv")
        
        # 按時間戳降序排列
        sorted_interactions = user_interactions_df.sort_values('Timestamp', ascending=False)
        
        # 顯示所有交互記錄
        output_container.dataframe(sorted_interactions, use_container_width=True)
        
        # 顯示統計信息
        output_container.info(f"共 {len(sorted_interactions)} 條交互記錄")
    else:
        output_container.warning("該用戶沒有歷史交互記錄")

    # 添加社群推薦功能
    output_container.markdown("---")
    output_container.subheader("社群推薦 - 看過類似電影的用戶推薦")
    
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
                output_container.subheader(f"用戶 {similar_user_id} 跟你看過類似的電影，所以你也可能喜歡看這些電影")
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
                        output_container.write(f"{rating} 星")
                
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

