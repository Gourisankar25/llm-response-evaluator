"""
Context Relevance Checker - Evaluates if context is relevant to the question

This module determines how relevant the provided context is for answering
a given question using cross-encoder models.

Why is this important?
Sometimes an LLM is given irrelevant context, leading to poor answers.
This metric helps identify when the context itself is the problem.

How it works:
Uses a cross-encoder model that scores [question, context] pairs directly.
Unlike bi-encoders (which encode separately), cross-encoders see both
texts together, providing more accurate relevance scores.

Bi-Encoder vs Cross-Encoder:
┌─────────────────┬──────────────────┬───────────────────┐
│                 │   Bi-Encoder     │   Cross-Encoder   │
├─────────────────┼──────────────────┼───────────────────┤
│ Encoding        │ Separate         │ Together          │
│ Speed           │ Fast             │ Slower            │
│ Accuracy        │ Good             │ Better            │
│ Use Case        │ Search/Retrieval │ Reranking/Scoring │
└─────────────────┴──────────────────┴───────────────────┘

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO (Microsoft MAchine Reading Comprehension)
- Specifically designed for passage relevance
- Returns a relevance score (higher = more relevant)
"""

from sentence_transformers import CrossEncoder
import numpy as np
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContextRelevanceChecker:
    """
    Evaluates context relevance to questions using cross-encoders.
    
    This checker determines if the provided context is actually useful
    for answering the given question.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the context relevance checker.
        
        Args:
            model_name: Cross-encoder model for relevance scoring
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        
        # Load cross-encoder model
        # Cross-encoders take pairs of texts and output a relevance score
        self.model = CrossEncoder(model_name)
        
        logger.info("Context Relevance Checker ready")
    
    def score_relevance(
        self,
        question: str,
        context: str
    ) -> float:
        """
        Score how relevant the context is for answering the question.
        
        The cross-encoder takes [question, context] as input and outputs
        a relevance score. Higher scores indicate more relevant context.
        
        Args:
            question: The question being asked
            context: The context provided to answer it
            
        Returns:
            Relevance score (typically 0-10, but can vary)
        """
        # Cross-encoder prediction
        # Input format: [[query, passage]]
        score = self.model.predict([[question, context]])[0]
        
        return float(score)
    
    def score(
        self,
        question: str,
        context: str,
        normalize: bool = True
    ) -> Dict:
        """
        Evaluate context relevance with detailed output.
        
        Args:
            question: Question to be answered
            context: Provided context
            normalize: Whether to normalize score to 0-1 range
            
        Returns:
            Dictionary containing:
                - relevance_score: float (0-1 if normalized)
                - raw_score: float (model's raw output)
                - is_relevant: bool (using threshold)
                - interpretation: str (description)
        """
        if not question or not context:
            raise ValueError("Both question and context must be non-empty")
        
        logger.info("Scoring context relevance...")
        
        # Get raw relevance score
        raw_score = self.score_relevance(question, context)
        
        # Normalize to 0-1 range if requested
        # MS-MARCO model typically outputs scores from -10 to +10
        if normalize:
            # Use min-max scaling with empirical bounds
            # Irrelevant pairs score around -5 to 0, relevant pairs 0 to +10
            min_score = -5.0
            max_score = 10.0
            
            # Clip and scale to 0-1 range
            normalized_score = (raw_score - min_score) / (max_score - min_score)
            normalized_score = max(0.0, min(1.0, normalized_score))  # Ensure 0-1 bounds
        else:
            normalized_score = raw_score
        
        # Determine if context is relevant (threshold: 0.5)
        is_relevant = normalized_score > 0.5
        
        # Provide interpretation
        if normalized_score > 0.8:
            interpretation = "Highly relevant - excellent context for the question"
        elif normalized_score > 0.6:
            interpretation = "Relevant - context contains useful information"
        elif normalized_score > 0.4:
            interpretation = "Somewhat relevant - context partially addresses question"
        else:
            interpretation = "Not relevant - context doesn't help answer the question"
        
        logger.info(f"Relevance score: {normalized_score:.3f} - {interpretation}")
        
        return {
            "relevance_score": normalized_score,
            "raw_score": raw_score,
            "is_relevant": is_relevant,
            "interpretation": interpretation
        }
    
    def rank_contexts(
        self,
        question: str,
        contexts: List[str],
        top_k: int = None
    ) -> List[Dict]:
        """
        Rank multiple contexts by relevance to a question.
        
        This is useful when you have multiple potential context sources
        and want to identify the most relevant ones.
        
        Args:
            question: Question to answer
            contexts: List of context documents
            top_k: Return only top K most relevant (None = all)
            
        Returns:
            List of contexts with scores, sorted by relevance (best first)
        """
        if not contexts:
            raise ValueError("contexts list must not be empty")
        
        logger.info(f"Ranking {len(contexts)} contexts...")
        
        # Create question-context pairs
        pairs = [[question, context] for context in contexts]
        
        # Batch prediction (more efficient than one-by-one)
        scores = self.model.predict(pairs)
        
        # Create results with context info
        results = []
        for idx, (context, score) in enumerate(zip(contexts, scores)):
            # Normalize score
            normalized_score = float(1 / (1 + np.exp(-score)))
            
            results.append({
                "context_index": idx,
                "context": context,
                "context_preview": context[:100] + "..." if len(context) > 100 else context,
                "relevance_score": normalized_score,
                "raw_score": float(score)
            })
        
        # Sort by relevance (highest first)
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Return top K if specified
        if top_k is not None:
            results = results[:top_k]
        
        logger.info(f"Top context score: {results[0]['relevance_score']:.3f}")
        
        return results
    
    def check_context_adequacy(
        self,
        question: str,
        context: str,
        min_relevance: float = 0.5
    ) -> Dict:
        """
        Check if the context is adequate for answering the question.
        
        This is useful for identifying when an LLM was given
        insufficient or irrelevant context.
        
        Args:
            question: Question to answer
            context: Provided context
            min_relevance: Minimum acceptable relevance score
            
        Returns:
            Dictionary with adequacy assessment
        """
        result = self.score(question, context)
        
        is_adequate = result["relevance_score"] >= min_relevance
        
        if not is_adequate:
            recommendation = (
                "Context is insufficient. Consider:\n"
                "  1. Retrieving more relevant documents\n"
                "  2. Using a different retrieval strategy\n"
                "  3. Expanding the context window"
            )
        else:
            recommendation = "Context is adequate for answering the question"
        
        return {
            "is_adequate": is_adequate,
            "relevance_score": result["relevance_score"],
            "meets_threshold": is_adequate,
            "threshold": min_relevance,
            "recommendation": recommendation,
            "interpretation": result["interpretation"]
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Context Relevance Demo")
    print("=" * 60)
    
    checker = ContextRelevanceChecker()
    
    # Example 1: Highly relevant context
    print("\n📌 Example 1: Highly Relevant Context")
    print("-" * 60)
    question1 = "What is the capital of France?"
    context1 = "France is a country in Western Europe. Its capital and largest city is Paris, known for the Eiffel Tower."
    
    print(f"Question: {question1}")
    print(f"Context: {context1}")
    
    result1 = checker.score(question1, context1)
    print(f"\n✅ Results:")
    print(f"  - Relevance Score: {result1['relevance_score']:.3f}")
    print(f"  - Is Relevant: {result1['is_relevant']}")
    print(f"  - {result1['interpretation']}")
    
    # Example 2: Irrelevant context
    print("\n📌 Example 2: Irrelevant Context")
    print("-" * 60)
    question2 = "How do you make chocolate chip cookies?"
    context2 = "The stock market experienced volatility today with tech stocks falling 2%."
    
    print(f"Question: {question2}")
    print(f"Context: {context2}")
    
    result2 = checker.score(question2, context2)
    print(f"\n✅ Results:")
    print(f"  - Relevance Score: {result2['relevance_score']:.3f}")
    print(f"  - Is Relevant: {result2['is_relevant']}")
    print(f"  - {result2['interpretation']}")
    
    # Example 3: Ranking multiple contexts
    print("\n📌 Example 3: Ranking Multiple Contexts")
    print("-" * 60)
    question3 = "What are the health benefits of exercise?"
    contexts = [
        "Regular exercise improves cardiovascular health and strengthens muscles.",
        "The latest smartphone features include 5G connectivity and improved cameras.",
        "Exercise also boosts mental health by reducing stress and anxiety.",
        "Climate change is affecting global temperatures and weather patterns."
    ]
    
    print(f"Question: {question3}")
    print("\nContexts to rank:")
    for i, ctx in enumerate(contexts, 1):
        print(f"  {i}. {ctx}")
    
    ranked = checker.rank_contexts(question3, contexts, top_k=3)
    print(f"\n✅ Top 3 Ranked Contexts:")
    for rank, item in enumerate(ranked, 1):
        print(f"\n  #{rank} (Score: {item['relevance_score']:.3f})")
        print(f"     {item['context']}")
    
    # Example 4: Context adequacy check
    print("\n📌 Example 4: Context Adequacy Check")
    print("-" * 60)
    question4 = "What is photosynthesis?"
    context4 = "Plants are living organisms that grow in soil."
    
    print(f"Question: {question4}")
    print(f"Context: {context4}")
    
    adequacy = checker.check_context_adequacy(question4, context4)
    print(f"\n✅ Adequacy Check:")
    print(f"  - Is Adequate: {adequacy['is_adequate']}")
    print(f"  - Score: {adequacy['relevance_score']:.3f}")
    print(f"  - Recommendation: {adequacy['recommendation']}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
