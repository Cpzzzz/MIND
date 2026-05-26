import numpy as np
import logging
from typing import List, Dict
from utils.metrics.semantic import get_sentence_embedding

# Hyperparameters (Defaults from methodology)
ALPHA = 1.0  # Quality weight
BETA = 1.0   # Diversity weight
LAMBDA = 0.5 # Toxicity reduction weight in Quality

def calculate_quality(example: Dict, lambda_val: float = LAMBDA) -> float:
    """
    Quality score q(E) = lambda * DeltaT(E) + (1 - lambda) * U(E)
    DeltaT(E) = ori_tox - upd_tox
    U(E) = Average(Corresponding, Helpfulness, Informativeness)
    """
    # 0. Fast path: Use pre-calculated Q_d if available
    if 'Q_d' in example:
        return float(example['Q_d'])

    # Defensive programming: handle missing keys
    ori_tox = example.get('ori_tox_max')
    if ori_tox is None:
        ori_tox_scores = example.get('ori_tox', {})
        ori_tox = max(ori_tox_scores.values()) if isinstance(ori_tox_scores, dict) and ori_tox_scores else 0.0
    
    upd_tox = example.get('upd_tox_max')
    if upd_tox is None:
        upd_tox_scores = example.get('upd_tox', {})
        upd_tox = max(upd_tox_scores.values()) if isinstance(upd_tox_scores, dict) and upd_tox_scores else 0.0
    
    delta_t = max(0, ori_tox - upd_tox)
    
    corr = example.get('intent_preservation', 0.0)
    help_val = example.get('helpfulness', 0.0)
    info = example.get('informativeness', 0.0)
    utility = (corr + help_val + info) / 3.0
       
    return lambda_val * delta_t + (1 - lambda_val) * utility

def calculate_representativeness(
    memory_subset: List[Dict],
    alpha: float,
    beta: float,
    embedding_model: str,
    lambda_val: float = LAMBDA,
) -> float:
    """
    Objective: alpha * Qual(M) + beta * Div(M)
    Qual(M) = Average Quality = 1/m * Sum(Qd)
    Div(M) = min dist(E_i, E_j)
    """
    if not memory_subset:
        return 0.0
        
    m = len(memory_subset)
    
    # Quality Part
    qual_sum = sum(calculate_quality(e, lambda_val=lambda_val) for e in memory_subset)
    qual_score = qual_sum / m
    
    # Diversity Part (Min Pairwise Distance)
    if m < 2:
        div_score = 0.0 
    else:
        # Re-calculate distances for the subset
        # Fetch embeddings fresh for calculation
        subset_embeddings = []
        for e in memory_subset:
            emb = get_sentence_embedding(e['ori_prompt'], model=embedding_model)
            if emb: subset_embeddings.append(np.array(emb))
            
        subset_embeddings = np.array(subset_embeddings)
        
        if len(subset_embeddings) < 2:
            div_score = 0.0
        else:
            dists = []
            for i in range(len(subset_embeddings)):
                for j in range(i + 1, len(subset_embeddings)):
                    dists.append(np.linalg.norm(subset_embeddings[i] - subset_embeddings[j]) / 2.0)
            
            div_score = min(dists) if dists else 0.0

    return alpha * qual_score + beta * div_score


def maintain_memory(
    new_examples: List[Dict], 
    memory_limit: int, 
    current_memory: List[Dict] = None, # Passed explicitly
    embedding_model: str = None,
    alpha: float = ALPHA, 
    beta: float = BETA,
    lambda_val: float = LAMBDA
):
    """
    MM-G Algorithm: 
    Construct M_q (Top-Quality) and M_d (Max-Min Diversity), select best.
    """
    if current_memory is None:
        current_memory = []
    
    # 1. Universal Set
    universal_set = current_memory + new_examples
    
    # Ensure embeddings exist (Fetched on demand, not stored)
    valid_indices = []
    embs_list = []
    
    for idx, item in enumerate(universal_set):
        item['_tmp_idx'] = idx # Temporary index for tracking
        
        # Always fetch embedding fresh
        emb = get_sentence_embedding(item['ori_prompt'], model=embedding_model)
        
        if emb:
            valid_indices.append(idx)
            embs_list.append(np.array(emb))
    
    # Filter universal set to only valid items
    universal_set = [universal_set[i] for i in valid_indices]
    embs = np.array(embs_list) if embs_list else np.empty((0, 0))
    n = len(universal_set)

    if len(universal_set) <= memory_limit:
        # Cleanup temp key
        for item in universal_set: item.pop('_tmp_idx', None)
        return universal_set

    # 2. Construct M_q (Top-K Quality)
    # Sort descending
    sorted_by_qual = sorted(universal_set, key=lambda x: calculate_quality(x, lambda_val=lambda_val), reverse=True)
    M_q = sorted_by_qual[:memory_limit]
    
    # 3. Construct M_d (Max-Min Diversity) - Greedy
    # Calculate full distance matrix using pre-fetched embeddings
    
    # Pairwise distances
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(embs[i] - embs[j]) / 2.0
            dist_matrix[i, j] = dist_matrix[j, i] = d
            
    # Greedy Selection for Diversity
    # Start with pair max distance
    initial_i, initial_j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    M_d_indices = [initial_i, initial_j]
    U_indices = set(range(n)) - {initial_i, initial_j}
    
    while len(M_d_indices) < memory_limit and U_indices:
        # Find e* in U maximizing min(dist(e*, e) for e in M_d)
        max_min_dist = -1.0
        best_candidate = -1
        
        current_selected = list(M_d_indices)
        
        for candidate in U_indices:
            # Min dist to current M_d
            dists_to_selected = dist_matrix[candidate, current_selected]
            min_d_val = np.min(dists_to_selected)
            
            if min_d_val > max_min_dist:
                max_min_dist = min_d_val
                best_candidate = candidate
        
        if best_candidate != -1:
            M_d_indices.append(best_candidate)
            U_indices.remove(best_candidate)
        else:
            break
            
    M_d = [universal_set[i] for i in M_d_indices]
    
    # 4. Compare M_q and M_d objectives
    score_q = calculate_representativeness(M_q, alpha, beta, embedding_model=embedding_model, lambda_val=lambda_val)
    score_d = calculate_representativeness(M_d, alpha, beta, embedding_model=embedding_model, lambda_val=lambda_val)
    
    logging.info(f"Memory Maintenance: Qual-Set Score={score_q:.4f}, Div-Set Score={score_d:.4f}")
    
    final_memory = M_q if score_q >= score_d else M_d
    
    # Cleanup and Return
    for item in final_memory: item.pop('_tmp_idx', None)
    return final_memory
