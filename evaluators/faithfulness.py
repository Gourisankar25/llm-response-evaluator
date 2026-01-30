"""
Faithfulness Scoring - Semantic similarity between response and context

This module measures how semantically similar (faithful) an LLM response
is to the source context using embedding-based similarity.

How it works:
1. Convert response and context into embeddings (numerical vectors)
2. Calculate cosine similarity between the embeddings
3. Higher similarity = response is more faithful to the context

What are embeddings?
- Numerical representations of text that capture semantic meaning
- Similar texts have similar embeddings (close in vector space)
- Example: "happy" and "joyful" have similar embeddings

Cosine Similarity:
- Measures the angle between two vectors
- Range: -1 to 1 (we normalize to 0-1)
- 1.0 = identical meaning
- 0.5 = somewhat similar
- 0.0 = completely different

Model: sentence-transformers/all-MiniLM-L6-v2
- Fast and efficient (22M parameters)
- Good balance of speed and quality
- 384-dimensional embeddings
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaithfulnessScorer:
    """
    Measures semantic faithfulness using embedding similarity.
    
    This scorer evaluates how closely an LLM response aligns with
    the semantic content of the source context.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the faithfulness scorer.
        
        Args:
            model_name: SentenceTransformer model identifier
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto
        """
        logger.info(f"Loading embedding model: {model_name}")
        
        # Load SentenceTransformer model
        # This model converts text into fixed-length embeddings
        self.model = SentenceTransformer(model_name, device=device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(
            f"Model loaded - Embedding dimension: {self.embedding_dim}"
        )
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Convert text into an embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        # The model handles:
        # 1. Tokenization (text → tokens)
        # 2. Encoding (tokens → embeddings)
        # 3. Pooling (multiple token embeddings → single sentence embedding)
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding
    
    def calculate_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Cosine similarity measures the angle between vectors:
        - 1.0 = vectors point in same direction (very similar)
        - 0.0 = vectors are perpendicular (unrelated)
        - -1.0 = vectors point opposite (contradictory)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        # Get embeddings
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Reshape for sklearn (expects 2D arrays)
        emb1 = emb1.reshape(1, -1)
        emb2 = emb2.reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        # Normalize to 0-1 range (cosine can be -1 to 1)
        # Most text similarities are positive, but we normalize anyway
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def score(
        self,
        response: str,
        context: str
    ) -> Dict:
        """
        Score the faithfulness of a response to context.
        
        Args:
            response: LLM response to evaluate
            context: Source context
            
        Returns:
            Dictionary containing:
                - faithfulness_score: float (0-1)
                - similarity: float (raw cosine similarity)
                - response_embedding: np.ndarray
                - context_embedding: np.ndarray
        """
        if not response or not context:
            raise ValueError("Both response and context must be non-empty")
        
        logger.info("Calculating faithfulness score...")
        
        # Get embeddings
        response_emb = self.get_embedding(response)
        context_emb = self.get_embedding(context)
        
        # Calculate similarity
        similarity = self.calculate_similarity(response, context)
        
        # Faithfulness score is the similarity
        faithfulness_score = similarity
        
        logger.info(f"Faithfulness score: {faithfulness_score:.3f}")
        
        return {
            "faithfulness_score": faithfulness_score,
            "similarity": similarity,
            "response_embedding": response_emb,
            "context_embedding": context_emb,
            "embedding_dimension": self.embedding_dim
        }
    
    def score_multi_context(
        self,
        response: str,
        contexts: List[str]
    ) -> Dict:
        """
        Score faithfulness against multiple context sources.
        
        This is useful when you have multiple relevant documents.
        We calculate similarity to each context and return:
        - Max similarity (best match)
        - Average similarity (overall faithfulness)
        
        Args:
            response: LLM response
            contexts: List of context documents
            
        Returns:
            Dictionary with max, average, and per-context scores
        """
        if not contexts:
            raise ValueError("contexts list must not be empty")
        
        logger.info(f"Scoring against {len(contexts)} contexts...")
        
        response_emb = self.get_embedding(response)
        
        similarities = []
        context_scores = []
        
        for i, context in enumerate(contexts):
            context_emb = self.get_embedding(context)
            
            # Calculate similarity
            similarity = float(cosine_similarity(
                response_emb.reshape(1, -1),
                context_emb.reshape(1, -1)
            )[0][0])
            
            # Normalize
            similarity = (similarity + 1) / 2
            
            similarities.append(similarity)
            context_scores.append({
                "context_index": i,
                "similarity": similarity,
                "context_preview": context[:100] + "..." if len(context) > 100 else context
            })
        
        max_similarity = max(similarities)
        avg_similarity = sum(similarities) / len(similarities)
        
        return {
            "max_faithfulness": max_similarity,
            "avg_faithfulness": avg_similarity,
            "num_contexts": len(contexts),
            "context_scores": context_scores,
            "similarities": similarities
        }
    
    def get_most_similar_segments(
        self,
        response: str,
        context: str,
        segment_size: int = 100
    ) -> List[Dict]:
        """
        Find which parts of the context are most similar to the response.
        
        This helps identify which source segments the response is based on.
        
        Args:
            response: LLM response
            context: Source context
            segment_size: Characters per segment
            
        Returns:
            List of segments sorted by similarity (highest first)
        """
        # Split context into segments
        segments = []
        for i in range(0, len(context), segment_size):
            segment = context[i:i + segment_size]
            if len(segment.strip()) > 20:  # Skip very short segments
                segments.append({
                    "text": segment,
                    "start_pos": i,
                    "end_pos": min(i + segment_size, len(context))
                })
        
        # Calculate similarity for each segment
        response_emb = self.get_embedding(response)
        
        for segment in segments:
            segment_emb = self.get_embedding(segment["text"])
            similarity = float(cosine_similarity(
                response_emb.reshape(1, -1),
                segment_emb.reshape(1, -1)
            )[0][0])
            segment["similarity"] = (similarity + 1) / 2
        
        # Sort by similarity (highest first)
        segments.sort(key=lambda x: x["similarity"], reverse=True)
        
        return segments


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("💎 Faithfulness Scoring Demo")
    print("=" * 60)
    
    scorer = FaithfulnessScorer()
    
    # Example 1: High faithfulness
    print("\n📌 Example 1: High Faithfulness")
    print("-" * 60)
    context1 = (
        "Artificial Intelligence is transforming healthcare by enabling "
        "faster diagnosis and personalized treatment plans."
    )
    response1 = (
        "AI is revolutionizing healthcare through improved diagnostics "
        "and customized patient care."
    )
    
    print(f"Context: {context1}")
    print(f"Response: {response1}")
    
    result1 = scorer.score(response1, context1)
    print(f"\n✅ Results:")
    print(f"  - Faithfulness Score: {result1['faithfulness_score']:.3f}")
    print(f"  - Interpretation: {'High' if result1['faithfulness_score'] > 0.7 else 'Medium' if result1['faithfulness_score'] > 0.5 else 'Low'} faithfulness")
    
    # Example 2: Low faithfulness
    print("\n📌 Example 2: Low Faithfulness")
    print("-" * 60)
    context2 = "The stock market closed higher today with tech stocks leading gains."
    response2 = "Quantum computing will revolutionize cryptography in the next decade."
    
    print(f"Context: {context2}")
    print(f"Response: {response2}")
    
    result2 = scorer.score(response2, context2)
    print(f"\n✅ Results:")
    print(f"  - Faithfulness Score: {result2['faithfulness_score']:.3f}")
    print(f"  - Interpretation: Response strays from context topic")
    
    # Example 3: Multiple contexts
    print("\n📌 Example 3: Multiple Context Sources")
    print("-" * 60)
    contexts = [
        "Python is known for its simple syntax and readability.",
        "JavaScript is the language of the web, running in browsers.",
        "Python is widely used in data science and machine learning."
    ]
    response3 = "Python is popular in AI and data science due to its ease of use."
    
    print("Contexts:")
    for i, ctx in enumerate(contexts, 1):
        print(f"  {i}. {ctx}")
    print(f"\nResponse: {response3}")
    
    result3 = scorer.score_multi_context(response3, contexts)
    print(f"\n✅ Results:")
    print(f"  - Max Faithfulness: {result3['max_faithfulness']:.3f}")
    print(f"  - Avg Faithfulness: {result3['avg_faithfulness']:.3f}")
    print(f"\n  📊 Per-Context Scores:")
    for ctx_score in result3['context_scores']:
        print(f"    Context {ctx_score['context_index'] + 1}: {ctx_score['similarity']:.3f}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
