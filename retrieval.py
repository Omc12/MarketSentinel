"""
Hybrid retrieval combining semantic search and BM25.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from fetch_news import NewsArticle
from mmr import mmr_rerank_with_scores


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline."""
    article: NewsArticle
    index: int
    score: float
    retrieval_type: str  # 'semantic', 'bm25', or 'hybrid'


class HybridRetriever:
    """
    Hybrid retriever combining semantic embeddings with BM25 keyword search.
    
    Flow:
    1. Compute semantic similarity scores
    2. Compute BM25 keyword scores
    3. Combine scores with weighted fusion
    4. Apply MMR re-ranking for diversity
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.6,
        bm25_weight: float = 0.4
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            model_name: Sentence transformer model for semantic embeddings
            semantic_weight: Weight for semantic scores in fusion
            bm25_weight: Weight for BM25 scores in fusion
        """
        self.model_name = model_name
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight
        
        # Lazy load the embedding model
        self._embedding_model: Optional[SentenceTransformer] = None
        
        # Document storage
        self.articles: List[NewsArticle] = []
        self.doc_texts: List[str] = []
        self.doc_embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        
    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.model_name)
        return self._embedding_model
    
    def index_articles(self, articles: List[NewsArticle]) -> None:
        """
        Index articles for retrieval.
        
        Args:
            articles: List of news articles to index
        """
        self.articles = articles
        self.doc_texts = [article.full_text for article in articles]
        
        if not articles:
            self.doc_embeddings = None
            self.bm25 = None
            return
        
        # Compute semantic embeddings
        self.doc_embeddings = self.embedding_model.encode(
            self.doc_texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Build BM25 index
        tokenized_docs = [self._tokenize(text) for text in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        # Lowercase and split on whitespace/punctuation
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _semantic_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Perform semantic search.
        
        Returns:
            List of (index, score) tuples
        """
        if self.doc_embeddings is None or len(self.doc_embeddings) == 0:
            return []
        
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Compute cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(self.doc_embeddings, axis=1)
        
        # Avoid division by zero
        doc_norms = np.where(doc_norms == 0, 1, doc_norms)
        if query_norm == 0:
            query_norm = 1
        
        similarities = np.dot(self.doc_embeddings, query_embedding) / (doc_norms * query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def _bm25_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Perform BM25 keyword search.
        
        Returns:
            List of (index, score) tuples
        """
        if self.bm25 is None:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 8
    ) -> List[Tuple[int, float]]:
        """
        Perform hybrid search combining semantic and BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (index, combined_score) tuples
        """
        if not self.articles:
            return []
        
        # Get semantic results
        semantic_results = self._semantic_search(query, top_k=len(self.articles))
        
        # Get BM25 results
        bm25_results = self._bm25_search(query, top_k=len(self.articles))
        
        # Normalize scores to [0, 1]
        semantic_scores = {idx: score for idx, score in semantic_results}
        bm25_scores = {idx: score for idx, score in bm25_results}
        
        # Normalize semantic scores
        if semantic_scores:
            max_sem = max(semantic_scores.values())
            min_sem = min(semantic_scores.values())
            range_sem = max_sem - min_sem if max_sem > min_sem else 1
            semantic_scores = {k: (v - min_sem) / range_sem for k, v in semantic_scores.items()}
        
        # Normalize BM25 scores
        if bm25_scores:
            max_bm25 = max(bm25_scores.values())
            min_bm25 = min(bm25_scores.values())
            range_bm25 = max_bm25 - min_bm25 if max_bm25 > min_bm25 else 1
            bm25_scores = {k: (v - min_bm25) / range_bm25 for k, v in bm25_scores.items()}
        
        # Combine scores
        all_indices = set(semantic_scores.keys()) | set(bm25_scores.keys())
        combined = []
        
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            combined_score = (
                self.semantic_weight * sem_score +
                self.bm25_weight * bm25_score
            )
            combined.append((idx, combined_score))
        
        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)
        
        return combined[:top_k]
    
    def retrieve(
        self,
        query: str,
        initial_top_k: int = 8,
        final_top_k: int = 4,
        mmr_lambda: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Full retrieval pipeline: hybrid search → MMR re-ranking.
        
        Args:
            query: Search query (typically ticker-related)
            initial_top_k: Number of candidates from hybrid search
            final_top_k: Number of final results after MMR
            mmr_lambda: MMR diversity parameter (higher = more relevance)
            
        Returns:
            List of RetrievalResult objects
        """
        if not self.articles:
            return []
        
        # Step 1: Hybrid search for top-k candidates
        hybrid_results = self.hybrid_search(query, top_k=initial_top_k)
        
        if not hybrid_results:
            return []
        
        # Step 2: Get embeddings for MMR
        candidate_indices = [idx for idx, _ in hybrid_results]
        candidate_scores = [score for _, score in hybrid_results]
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        # Get candidate embeddings
        candidate_embeddings = self.doc_embeddings[candidate_indices]
        
        # Step 3: Apply MMR re-ranking
        mmr_results = mmr_rerank_with_scores(
            query_embedding=query_embedding,
            doc_embeddings=candidate_embeddings,
            doc_indices=candidate_indices,
            initial_scores=candidate_scores,
            top_k=final_top_k,
            lambda_param=mmr_lambda
        )
        
        # Step 4: Build results
        results = []
        for idx, score in mmr_results:
            results.append(RetrievalResult(
                article=self.articles[idx],
                index=idx,
                score=score,
                retrieval_type="hybrid+mmr"
            ))
        
        return results


def create_retriever() -> HybridRetriever:
    """Factory function to create a retriever instance."""
    return HybridRetriever(
        model_name="all-MiniLM-L6-v2",
        semantic_weight=0.6,
        bm25_weight=0.4
    )


if __name__ == "__main__":
    # Test the retriever
    from fetch_news import NewsArticle
    
    # Create sample articles
    articles = [
        NewsArticle(
            title="Apple Reports Record Q4 Earnings",
            summary="Apple Inc. announced record-breaking fourth quarter earnings, beating analyst expectations with strong iPhone sales.",
            source="Reuters",
            url="https://example.com/1",
            published_at="20240115T120000"
        ),
        NewsArticle(
            title="Apple Stock Rises on AI Announcements",
            summary="Shares of Apple climbed 5% following announcements about new AI features coming to iOS.",
            source="Bloomberg",
            url="https://example.com/2",
            published_at="20240114T100000"
        ),
        NewsArticle(
            title="Tech Sector Faces Regulatory Pressure",
            summary="Big tech companies including Apple face increased scrutiny from regulators over market dominance.",
            source="WSJ",
            url="https://example.com/3",
            published_at="20240113T090000"
        ),
    ]
    
    retriever = create_retriever()
    retriever.index_articles(articles)
    
    results = retriever.retrieve("Apple stock performance earnings", initial_top_k=3, final_top_k=2)
    
    print(f"Retrieved {len(results)} articles:\n")
    for r in results:
        print(f"Score: {r.score:.3f} | {r.article.title}")
