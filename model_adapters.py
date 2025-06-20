import os
import sys
import subprocess
import torch
import numpy as np
import networkx as nx
from pathlib import Path

# Add RL_recommender to Python path
RL_RECOMMENDER_PATH = Path("../RL_recommender").resolve()
sys.path.append(str(RL_RECOMMENDER_PATH))

def heuristic_exposure_strategy_wrapper(user_item_graph, rec_item_set, item_emb, output_container=None,
                                       n_selected_communities=2,
                                       n_diverse_users_per_community=10,
                                       n_items_per_user=3):
    """
    修改過的 heuristic exposure 策略函數，帶有 Streamlit 輸出支持
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import community.community_louvain as community_louvain
    from collections import defaultdict, Counter
    
    def calculate_ILS(item_emb, rec_item_set):
        if len(rec_item_set) <= 1:
            return 0.0
        embeddings = item_emb[rec_item_set].cpu().detach().numpy()
        sim_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(sim_matrix, 0)
        k = len(rec_item_set)
        ils = np.sum(sim_matrix) / (k * (k - 1))
        return ils

    # 使用Louvain算法檢測社群
    if output_container:
        output_container.text("正在檢測社群...")

    connected_components = list(nx.connected_components(user_item_graph))
    if output_container:
        output_container.text(f"總共有 {len(connected_components)} 個連通元件")

    all_partition = community_louvain.best_partition(user_item_graph, resolution=2, random_state=42)
    communities = all_partition
    
    if output_container:
        output_container.text(f"總社群數量: {len(set(communities.values()))}")

    # 將節點分為用戶節點和項目節點
    users = [node for node in user_item_graph.nodes() if user_item_graph.nodes[node].get('bipartite') == 0]
    items = [node for node in user_item_graph.nodes() if user_item_graph.nodes[node].get('bipartite') == 1]
    
    if output_container:
        output_container.text(f"用戶節點數: {len(users)}, 電影節點數: {len(items)}")
    
    # 為每個社群分配用戶和項目
    community_users = defaultdict(list)
    community_items = defaultdict(list)
    
    for node, community_id in communities.items():
        if node in users:
            community_users[community_id].append(node)
        elif node in items:
            community_items[community_id].append(node)
    
    # 過濾掉少於3個user或3個item的社群
    valid_communities = [comm_id for comm_id in set(communities.values())
                        if len(community_users[comm_id]) > 2 and len(community_items[comm_id]) > 2]
    
    if output_container:
        output_container.text("Top 社群資訊（前5）:")
        comm_sizes = Counter(communities.values())
        for i, (comm_id, size) in enumerate(comm_sizes.most_common(5)):
            n_users = len(community_users[comm_id])
            n_items = len(community_items[comm_id])
            output_container.text(f"社群 {comm_id}: 總節點={size}, 使用者={n_users}, 電影={n_items}")
    
    if len(valid_communities) < 2:
        if output_container:
            output_container.warning("警告: 有效社群數量不足，無法進行跨社群曝光")
        return []
    
    if output_container:
        output_container.text(f"有效社群數量: {len(valid_communities)}")
    
    # 隨機選擇社群
    if len(valid_communities) < n_selected_communities:
        if output_container:
            output_container.warning(f"警告: 有效社群數量({len(valid_communities)})小於請求數量({n_selected_communities})")
        selected_indices = valid_communities
    else:
        selected_indices = np.random.choice(valid_communities, 
                                            size=n_selected_communities, 
                                            replace=False)

    all_selected_items = []
    # 從被選到的社群創建曝光邊
    for i in range(len(selected_indices)):
        source_idx = selected_indices[i]

        # 計算社群中每個用戶的交互項目集合的多樣性得分(ILS)
        user_diversity_scores = {}
        for user in community_users[source_idx]:
            user = ''.join(filter(str.isdigit, user))
            if int(user) < len(rec_item_set):
                user_items = rec_item_set[int(user)]
                if len(user_items) > 1:
                    user_diversity_scores[user] = calculate_ILS(item_emb, user_items)
        
        if not user_diversity_scores:
            continue
        
        # 選擇ILS最低的用戶(最多樣化的用戶)
        diverse_users = sorted(user_diversity_scores.keys(), 
                                key=lambda u: user_diversity_scores[u])
        diverse_users = diverse_users[:min(n_diverse_users_per_community, len(diverse_users))]

        # 為每個多樣化用戶選擇歷史交互項目
        for user in diverse_users:
            user_items = list(user_item_graph.neighbors(f"u{user}"))
            
            if len(user_items) == 0:
                continue
            
            # 隨機選擇一些項目
            selected_items = np.random.choice(user_items, 
                                            size=min(n_items_per_user, len(user_items)), 
                                            replace=False)
            for item_node_id in selected_items:
                item_node_id = ''.join(filter(str.isdigit, item_node_id))
                all_selected_items.append((int(user), int(item_node_id)))  

    if output_container:
        output_container.text(f"已生成 {len(all_selected_items)} 條曝光邊")
    
    return all_selected_items

def run_heuristic_exposure(output_container=None):
    """
    運行 Heuristic Exposure 模型的包裝函數
    """
    if output_container:
        output_container.text("正在啟動 Heuristic Exposure 模型...")
    
    try:
        # 切換到 RL_recommender 目錄
        original_dir = os.getcwd()
        os.chdir(RL_RECOMMENDER_PATH)
        
        # 添加當前目錄到 Python 路徑
        if str(RL_RECOMMENDER_PATH) not in sys.path:
            sys.path.insert(0, str(RL_RECOMMENDER_PATH))
        
        if output_container:
            output_container.text("正在載入必要模組...")
        
        # 直接導入並執行函數
        try:
            from utility.build_nx_graph import build_nx_graph
            
            if output_container:
                output_container.text("正在載入模型文件...")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            user_embeddings = torch.load("model/simulator/user_emb.pt", map_location=device, weights_only=True)
            item_embeddings = torch.load("model/simulator/item_emb.pt", map_location=device, weights_only=True)
            
            if output_container:
                output_container.text(f"成功載入模型: {user_embeddings.shape[0]} 個用戶, {item_embeddings.shape[0]} 個電影")
            
            # 創建推薦項目集合
            rec_item_set = [np.random.choice(range(item_embeddings.shape[0]), size=20, replace=False).tolist() 
                           for user in range(user_embeddings.shape[0])]
            
            if output_container:
                output_container.text("正在建立用戶-電影網絡...")
            
            user_item_graph = build_nx_graph()
            
            if output_container:
                output_container.text("正在執行 Heuristic Exposure 策略...")
            
            # 執行我們自己的模型函數
            selected_items = heuristic_exposure_strategy_wrapper(
                user_item_graph, rec_item_set, item_embeddings, output_container
            )
            
            # 切換回原目錄
            os.chdir(original_dir)
            
            if output_container:
                output_container.success("Heuristic Exposure 模型執行成功！")
                output_container.subheader("推薦結果摘要")
                output_container.text(f"🎬 總共生成了 {len(selected_items)} 個推薦配對")
                
                if len(selected_items) > 0:
                    unique_users = len(set([item[0] for item in selected_items]))
                    unique_movies = len(set([item[1] for item in selected_items]))
                    output_container.text(f"👥 涉及用戶數量: {unique_users}")
                    output_container.text(f"🎭 涉及電影數量: {unique_movies}")
                    
                    output_container.subheader("推薦範例（前10個）")
                    for i, (user_id, movie_id) in enumerate(selected_items[:10]):
                        output_container.text(f"用戶 {user_id} → 電影 {movie_id}")
                else:
                    output_container.warning("沒有生成任何推薦配對")
                        
            return f"成功生成 {len(selected_items)} 個推薦配對"
            
        except ImportError as e:
            error_msg = f"模組導入失敗: {str(e)}"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
        except Exception as e:
            error_msg = f"執行過程中發生錯誤: {str(e)}"
            if output_container:
                output_container.error(error_msg)
            os.chdir(original_dir)
            return error_msg
            
    except Exception as e:
        error_msg = f"模型執行出錯: {str(e)}"
        if output_container:
            output_container.error(error_msg)
        try:
            os.chdir(original_dir)
        except:
            pass
        return error_msg

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
        output_container.info("注意：RL 模型需要更長的執行時間，請耐心等待...")
    
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
        try:
            os.chdir(original_dir)
        except:
            pass

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
        
        # 檢查模型文件
        user_emb_file = RL_RECOMMENDER_PATH / "model" / "simulator" / "user_emb.pt"
        item_emb_file = RL_RECOMMENDER_PATH / "model" / "simulator" / "item_emb.pt"
        train_data_file = RL_RECOMMENDER_PATH / "data" / "train.dat"
        
        if not user_emb_file.exists():
            return False, f"用戶嵌入文件不存在: {user_emb_file}"
            
        if not item_emb_file.exists():
            return False, f"項目嵌入文件不存在: {item_emb_file}"
            
        if not train_data_file.exists():
            return False, f"訓練數據文件不存在: {train_data_file}"
        
        return True, "所有依賴檢查通過"
        
    except Exception as e:
        return False, f"依賴檢查出錯: {str(e)}" 