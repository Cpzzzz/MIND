import numpy as np
import logging
from typing import List, Dict
from utils.metrics.semantic import get_sentence_embedding

# Hyperparameters
ALPHA = 1.0  # Similarity weight
BETA = 1.0   # Quality weight
GAMMA = 1.0  # Diversity weight
LAMBDA = 0.5 # Toxicity reduction weight

def calculate_similarity_score(prompt_emb: np.ndarray, item_emb: np.ndarray) -> float:
    dist = np.linalg.norm(prompt_emb - item_emb) / 2.0
    return 1.0 - dist

def calculate_quality_score(example: Dict, lambda_val: float = LAMBDA) -> float:
    """
    Q_d(x, x') = lambda * DeltaT + (1 - lambda) * U
    DeltaT = T(f(x)) - T(f(x'))
    U = Average(Corresponding, Helpfulness, Informativeness)
    """
    # 0. Fast path: Use pre-calculated Q_d if available
    if 'Q_d' in example:
        return float(example['Q_d'])

    # 1. Toxicity Reduction
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

def retrieve_memory(
    prompt: str, 
    k_examples: int, 
    memory_data: List[Dict] = None, # Passed explicitly
    embedding_model: str = None,
    alpha: float = ALPHA, 
    beta: float = BETA, 
    gamma: float = GAMMA,
    lambda_val: float = LAMBDA,
) -> List[Dict]:
    """
    MR-G Algorithm:
    Select subset E to maximize alpha*Sim + beta*Qual + gamma*Div
    where Div is sum of pairwise distances.
    """
    if memory_data is None:
        memory_data = [] # Should usually be passed in
    if not memory_data:
        return []
        
    # Pre-calculate Query Embedding
    embedding_list = get_sentence_embedding(prompt, model=embedding_model)
    if not embedding_list:
        logging.warning("[Warning] Could not retrieve embedding for prompt. Retrieval skipped.")
        return []
    prompt_emb = np.array(embedding_list)
    
    # Pre-calculate static scores for all candidates
    candidates = []
    for idx, item in enumerate(memory_data):
        # Always retrieve embeddings on demand instead of storing them in memory records.
        item_emb_list = get_sentence_embedding(item['ori_prompt'], model=embedding_model)
        if not item_emb_list:
             # logging.warning(f"[Warning] Could not retrieve embedding for memory item. Skipping item.")
             continue # Skip items where embedding cannot be retrieved

        item_emb = np.array(item_emb_list)
            
        sim = calculate_similarity_score(prompt_emb, item_emb) # Changed signature to accept embedding directly
        qual = calculate_quality_score(item, lambda_val=lambda_val)
        
        candidates.append({
            'item': item,
            'sim': sim,
            'qual': qual,
            'emb': item_emb,
            'idx': idx
        })
        
    sim_weight = alpha / k_examples if k_examples > 0 else 0.0
    qual_weight = beta / k_examples if k_examples > 0 else 0.0
    div_weight = gamma * 2.0 / (k_examples * (k_examples - 1)) if k_examples > 1 else 0.0

    # Universal set U (indices)
    U = set(range(len(candidates)))
    E_indices = []
    
    # Greedy Iterations
    for _ in range(min(k_examples, len(candidates))):
        best_gain = -float('inf')
        best_candidate_idx = -1
        
        # Calculate marginal gain for each candidate e in U
        # Gain(e) = alpha*Sim(e) + beta*Qual(e) + gamma * (Sum dist(e, existing E))
        # Note: Div(E U {e}) - Div(E) = Sum_{e' in E} dist(e, e')
        
        for idx in U:
            cand = candidates[idx]
            
            # Marginal gain under the normalized MR objective in the paper.
            gain = sim_weight * cand['sim'] + qual_weight * cand['qual']
            
            # Diversity gain
            div_gain = 0.0
            if gamma > 0 and E_indices:
                for e_idx in E_indices:
                    e_emb = candidates[e_idx]['emb']
                    dist = np.linalg.norm(cand['emb'] - e_emb) / 2.0
                    div_gain += dist
            
            total_gain = gain + div_weight * div_gain
            
            if total_gain > best_gain:
                best_gain = total_gain
                best_candidate_idx = idx
        
        if best_candidate_idx != -1:
            E_indices.append(best_candidate_idx)
            U.remove(best_candidate_idx)
            
    selected_examples = [candidates[i]['item'] for i in E_indices]
    
    logging.info(f"Memory Retrieval: Selected {len(selected_examples)} examples for prompt '{prompt[:30]}...'")
    return selected_examples
