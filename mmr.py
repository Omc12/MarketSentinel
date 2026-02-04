"""
Maximal Marginal Relevance (MMR) re-ranking for diversity.
"""

import numpy as np
from typing import List, Tuple


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix."""
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normalized = embeddings / norms
    
    # Compute similarity matrix
    return np.dot(normalized, normalized.T)


def mmr_rerank(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_indices: List[int],
    top_k: int = 4,
    lambda_param: float = 0.7
) -> List[int]:
    """
    Apply Maximal Marginal Relevance re-ranking to reduce redundancy.
    
    MMR balances relevance to query with diversity among selected documents.
    
    Args:
        query_embedding: Embedding of the query
        doc_embeddings: Embeddings of candidate documents (shape: [n_docs, embedding_dim])
        doc_indices: Original indices of the documents
        top_k: Number of documents to select
        lambda_param: Trade-off between relevance (1.0) and diversity (0.0)
                     Higher values favor relevance, lower values favor diversity.
                     
    Returns:
        List of selected document indices (from doc_indices)
    """
    if len(doc_indices) == 0:
        return []
    
    if len(doc_indices) <= top_k:
        return doc_indices
    
    n_docs = len(doc_indices)
    
    # Compute relevance scores (similarity to query)
    relevance_scores = np.array([
        cosine_similarity(query_embedding, doc_embeddings[i])
        for i in range(n_docs)
    ])
    
    # Compute document-document similarity matrix
    doc_sim_matrix = cosine_similarity_matrix(doc_embeddings)
    
    # Track selected and remaining documents
    selected_indices = []
    remaining_mask = np.ones(n_docs, dtype=bool)
    
    for _ in range(top_k):
        if not remaining_mask.any():
            break
            
        # Compute MMR scores for remaining documents
        mmr_scores = np.full(n_docs, -np.inf)
        
        for i in range(n_docs):
            if not remaining_mask[i]:
                continue
                
            # Relevance component
            relevance = relevance_scores[i]
            
            # Diversity component (max similarity to already selected docs)
            if selected_indices:
                max_sim_to_selected = max(
                    doc_sim_matrix[i, j] for j in selected_indices
                )
            else:
                max_sim_to_selected = 0.0
            
            # MMR score
            mmr_scores[i] = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
        
        # Select document with highest MMR score
        best_idx = np.argmax(mmr_scores)
        selected_indices.append(best_idx)
        remaining_mask[best_idx] = False
    
    # Return original indices
    return [doc_indices[i] for i in selected_indices]


def mmr_rerank_with_scores(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    doc_indices: List[int],
    initial_scores: List[float],
    top_k: int = 4,
    lambda_param: float = 0.7
) -> List[Tuple[int, float]]:
    """
    MMR re-ranking that also considers initial retrieval scores.
    
    Args:
        query_embedding: Embedding of the query
        doc_embeddings: Embeddings of candidate documents
        doc_indices: Original indices of the documents
        initial_scores: Initial retrieval scores (e.g., from hybrid search)
        top_k: Number of documents to select
        lambda_param: Trade-off between relevance and diversity
        
    Returns:
        List of (doc_index, final_score) tuples
    """
    if len(doc_indices) == 0:
        return []
    
    if len(doc_indices) <= top_k:
        return list(zip(doc_indices, initial_scores))
    
    n_docs = len(doc_indices)
    
    # Normalize initial scores to [0, 1]
    scores_array = np.array(initial_scores)
    if scores_array.max() > scores_array.min():
        normalized_scores = (scores_array - scores_array.min()) / (scores_array.max() - scores_array.min())
    else:
        normalized_scores = np.ones(n_docs)
    
    # Compute document-document similarity matrix
    doc_sim_matrix = cosine_similarity_matrix(doc_embeddings)
    
    # Track selected and remaining documents
    selected = []
    remaining_mask = np.ones(n_docs, dtype=bool)
    
    for _ in range(top_k):
        if not remaining_mask.any():
            break
            
        mmr_scores = np.full(n_docs, -np.inf)
        
        for i in range(n_docs):
            if not remaining_mask[i]:
                continue
            
            # Use normalized initial score as relevance
            relevance = normalized_scores[i]
            
            # Diversity component
            if selected:
                max_sim = max(doc_sim_matrix[i, j] for j, _ in selected)
            else:
                max_sim = 0.0
            
            mmr_scores[i] = lambda_param * relevance - (1 - lambda_param) * max_sim
        
        best_idx = np.argmax(mmr_scores)
        selected.append((best_idx, mmr_scores[best_idx]))
        remaining_mask[best_idx] = False
    
    # Return original indices with scores
    return [(doc_indices[i], score) for i, score in selected]


if __name__ == "__main__":
    # Test MMR
    np.random.seed(42)
    
    # Create sample embeddings
    query_emb = np.random.randn(384)
    doc_embs = np.random.randn(8, 384)
    
    # Make some documents similar to each other
    doc_embs[1] = doc_embs[0] + np.random.randn(384) * 0.1
    doc_embs[3] = doc_embs[2] + np.random.randn(384) * 0.1
    
    indices = list(range(8))
    
    selected = mmr_rerank(query_emb, doc_embs, indices, top_k=4, lambda_param=0.7)
    print(f"Selected indices: {selected}")
