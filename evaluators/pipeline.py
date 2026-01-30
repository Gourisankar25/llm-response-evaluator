"""
Evaluation Pipeline - Comprehensive LLM response assessment

This module combines all individual evaluators into a single pipeline
that provides a complete evaluation of LLM responses.

Metrics evaluated:
1. Hallucination Detection - Contradictions with context
2. Groundedness - % of claims supported by context
3. Faithfulness - Semantic similarity to context
4. Context Relevance - How relevant is context to question
5. Answer Completeness - All question aspects addressed

Overall Score Calculation:
- Weighted average of all metrics
- Configurable weights for each metric
- Final score: 0-100

Quality Tiers:
- 90-100: Excellent
- 75-89: Good
- 60-74: Acceptable
- 40-59: Poor
- 0-39: Very Poor
"""

from typing import Dict, Optional
import logging
from dataclasses import dataclass, asdict
import time

from evaluators.hallucination import HallucinationDetector
from evaluators.groundedness import GroundednessScorer
from evaluators.faithfulness import FaithfulnessScorer
from evaluators.relevance import ContextRelevanceChecker
from evaluators.completeness import CompletenessChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationWeights:
    """
    Configurable weights for each evaluation metric.
    
    Weights should sum to 1.0 for proper normalization.
    Adjust based on your use case priorities.
    """
    hallucination: float = 0.25  # Critical - contradictions are serious
    groundedness: float = 0.20   # Important - claims should be supported
    faithfulness: float = 0.20   # Important - stay close to source
    relevance: float = 0.15      # Moderate - context quality matters
    completeness: float = 0.20   # Important - answer everything
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = (
            self.hallucination + 
            self.groundedness + 
            self.faithfulness + 
            self.relevance + 
            self.completeness
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class LLMEvaluationPipeline:
    """
    Complete evaluation pipeline for LLM responses.
    
    This pipeline runs all evaluation metrics and combines them
    into a comprehensive quality assessment.
    """
    
    def __init__(
        self,
        weights: Optional[EvaluationWeights] = None,
        nli_model: str = "facebook/bart-large-mnli",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the evaluation pipeline.
        
        Args:
            weights: Metric weights (uses defaults if None)
            nli_model: Model for hallucination/groundedness
            embedding_model: Model for faithfulness
            cross_encoder_model: Model for relevance
        """
        logger.info("=" * 60)
        logger.info("Initializing LLM Evaluation Pipeline")
        logger.info("=" * 60)
        
        # Set weights
        self.weights = weights or EvaluationWeights()
        
        # Initialize all evaluators
        logger.info("\n🔧 Loading models...")
        
        self.hallucination_detector = HallucinationDetector(model_name=nli_model)
        logger.info("  ✓ Hallucination detector ready")
        
        self.groundedness_scorer = GroundednessScorer(nli_model=nli_model)
        logger.info("  ✓ Groundedness scorer ready")
        
        self.faithfulness_scorer = FaithfulnessScorer(model_name=embedding_model)
        logger.info("  ✓ Faithfulness scorer ready")
        
        self.relevance_checker = ContextRelevanceChecker(model_name=cross_encoder_model)
        logger.info("  ✓ Context relevance checker ready")
        
        self.completeness_checker = CompletenessChecker(nli_model=nli_model)
        logger.info("  ✓ Completeness checker ready")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Pipeline ready for evaluation!")
        logger.info("=" * 60 + "\n")
    
    def evaluate(
        self,
        question: str,
        context: str,
        llm_response: str,
        hallucination_threshold: float = 0.3,
        groundedness_threshold: float = 0.4,
        relevance_threshold: float = 0.5
    ) -> Dict:
        """
        Run complete evaluation on an LLM response.
        
        Args:
            question: The question that was asked
            context: Source context provided to LLM
            llm_response: LLM's generated response
            hallucination_threshold: Threshold for hallucination detection
            groundedness_threshold: Threshold for claim support
            relevance_threshold: Threshold for context relevance
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        logger.info("\n" + "=" * 60)
        logger.info("🔍 Starting Comprehensive Evaluation")
        logger.info("=" * 60)
        
        start_time = time.time()
        results = {}
        
        # 1. Hallucination Detection
        logger.info("\n1️⃣ Running Hallucination Detection...")
        hallucination_result = self.hallucination_detector.detect(
            context=context,
            response=llm_response,
            threshold=hallucination_threshold
        )
        
        # Convert to 0-1 score (0 = has hallucination, 1 = no hallucination)
        hallucination_score = 1.0 - hallucination_result['contradiction_score']
        
        results['hallucination'] = {
            'score': hallucination_score,
            'has_hallucination': hallucination_result['has_hallucination'],
            'contradiction_score': hallucination_result['contradiction_score'],
            'label': hallucination_result['label'],
            'weight': self.weights.hallucination
        }
        
        # 2. Groundedness Scoring
        logger.info("\n2️⃣ Running Groundedness Assessment...")
        groundedness_result = self.groundedness_scorer.score(
            response=llm_response,
            context=context,
            entailment_threshold=groundedness_threshold
        )
        
        results['groundedness'] = {
            'score': groundedness_result['groundedness_score'],
            'num_claims': groundedness_result['num_claims'],
            'num_supported': groundedness_result['num_supported'],
            'unsupported_claims': groundedness_result['unsupported_claims'],
            'weight': self.weights.groundedness
        }
        
        # 3. Faithfulness Scoring
        logger.info("\n3️⃣ Running Faithfulness Assessment...")
        faithfulness_result = self.faithfulness_scorer.score(
            response=llm_response,
            context=context
        )
        
        results['faithfulness'] = {
            'score': faithfulness_result['faithfulness_score'],
            'similarity': faithfulness_result['similarity'],
            'weight': self.weights.faithfulness
        }
        
        # 4. Context Relevance
        logger.info("\n4️⃣ Running Context Relevance Check...")
        relevance_result = self.relevance_checker.score(
            question=question,
            context=context
        )
        
        results['relevance'] = {
            'score': relevance_result['relevance_score'],
            'is_relevant': relevance_result['is_relevant'],
            'interpretation': relevance_result['interpretation'],
            'weight': self.weights.relevance
        }
        
        # 5. Answer Completeness
        logger.info("\n5️⃣ Running Completeness Check...")
        completeness_result = self.completeness_checker.score(
            question=question,
            response=llm_response
        )
        
        results['completeness'] = {
            'score': completeness_result['completeness_score'],
            'num_aspects': completeness_result['num_aspects'],
            'num_addressed': completeness_result['num_addressed'],
            'missing_aspects': [a.get('description', '') for a in completeness_result['missing_aspects']],
            'weight': self.weights.completeness
        }
        
        # Calculate Overall Score
        overall_score = (
            results['hallucination']['score'] * self.weights.hallucination +
            results['groundedness']['score'] * self.weights.groundedness +
            results['faithfulness']['score'] * self.weights.faithfulness +
            results['relevance']['score'] * self.weights.relevance +
            results['completeness']['score'] * self.weights.completeness
        )
        
        # Convert to 0-100 scale
        overall_score_100 = overall_score * 100
        
        # Determine quality tier
        if overall_score_100 >= 90:
            quality_tier = "Excellent"
            emoji = "🌟"
        elif overall_score_100 >= 75:
            quality_tier = "Good"
            emoji = "✅"
        elif overall_score_100 >= 60:
            quality_tier = "Acceptable"
            emoji = "👍"
        elif overall_score_100 >= 40:
            quality_tier = "Poor"
            emoji = "⚠️"
        else:
            quality_tier = "Very Poor"
            emoji = "❌"
        
        # Identify issues
        issues = []
        if results['hallucination']['has_hallucination']:
            issues.append("Contains hallucinations/contradictions")
        if results['groundedness']['score'] < 0.7:
            issues.append(f"Low groundedness ({results['groundedness']['score']:.0%})")
        if results['faithfulness']['score'] < 0.6:
            issues.append(f"Low faithfulness to context ({results['faithfulness']['score']:.0%})")
        if not results['relevance']['is_relevant']:
            issues.append("Context is not relevant to question")
        if results['completeness']['score'] < 0.8:
            issues.append(f"Incomplete answer ({results['completeness']['score']:.0%})")
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Build final result
        final_result = {
            'overall_score': overall_score,
            'overall_score_100': overall_score_100,
            'quality_tier': quality_tier,
            'emoji': emoji,
            'metrics': results,
            'issues': issues,
            'has_issues': len(issues) > 0,
            'weights': asdict(self.weights),
            'execution_time_seconds': execution_time,
            'input': {
                'question': question,
                'context': context[:200] + "..." if len(context) > 200 else context,
                'response': llm_response[:200] + "..." if len(llm_response) > 200 else llm_response
            }
        }
        
        logger.info("\n" + "=" * 60)
        logger.info(f"{emoji} Evaluation Complete!")
        logger.info("=" * 60)
        logger.info(f"Overall Score: {overall_score_100:.1f}/100 ({quality_tier})")
        logger.info(f"Execution Time: {execution_time:.2f}s")
        
        if issues:
            logger.info(f"\n⚠️ Issues Found ({len(issues)}):")
            for issue in issues:
                logger.info(f"  - {issue}")
        else:
            logger.info("\n✅ No major issues detected!")
        
        logger.info("=" * 60 + "\n")
        
        return final_result
    
    def get_summary(self, evaluation_result: Dict) -> str:
        """
        Generate a human-readable summary of evaluation results.
        
        Args:
            evaluation_result: Result from evaluate() method
            
        Returns:
            Formatted summary string
        """
        r = evaluation_result
        
        summary = f"""
{'=' * 60}
{r['emoji']} LLM Response Evaluation Summary
{'=' * 60}

OVERALL SCORE: {r['overall_score_100']:.1f}/100 ({r['quality_tier']})

METRIC BREAKDOWN:
  1. Hallucination:  {r['metrics']['hallucination']['score']:.2%} (Weight: {r['weights']['hallucination']:.0%})
  2. Groundedness:   {r['metrics']['groundedness']['score']:.2%} (Weight: {r['weights']['groundedness']:.0%})
  3. Faithfulness:   {r['metrics']['faithfulness']['score']:.2%} (Weight: {r['weights']['faithfulness']:.0%})
  4. Relevance:      {r['metrics']['relevance']['score']:.2%} (Weight: {r['weights']['relevance']:.0%})
  5. Completeness:   {r['metrics']['completeness']['score']:.2%} (Weight: {r['weights']['completeness']:.0%})

"""
        if r['has_issues']:
            summary += f"ISSUES DETECTED ({len(r['issues'])}):\n"
            for i, issue in enumerate(r['issues'], 1):
                summary += f"  {i}. {issue}\n"
        else:
            summary += "✅ No major issues detected!\n"
        
        summary += f"\nExecution Time: {r['execution_time_seconds']:.2f}s\n"
        summary += "=" * 60
        
        return summary


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 LLM Evaluation Pipeline Demo")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = LLMEvaluationPipeline()
    
    # Example evaluation
    print("\n📝 Example Evaluation")
    print("-" * 60)
    
    question = "What is Python and when was it created?"
    
    context = """
    Python is a high-level, interpreted programming language created by Guido van Rossum.
    It was first released in 1991. Python emphasizes code readability and simplicity.
    """
    
    llm_response = """
    Python is a programming language developed by Guido van Rossum. It was released in 1991.
    Python is known for its simple and readable syntax, making it popular for beginners.
    """
    
    print(f"Question: {question}")
    print(f"\nContext: {context.strip()}")
    print(f"\nLLM Response: {llm_response.strip()}")
    
    # Run evaluation
    result = pipeline.evaluate(
        question=question,
        context=context,
        llm_response=llm_response
    )
    
    # Print summary
    print(pipeline.get_summary(result))
    
    # Example 2: Poor quality response
    print("\n" + "=" * 60)
    print("📝 Example 2: Poor Quality Response")
    print("=" * 60)
    
    poor_response = "Python is a snake. It was created in 1995 by Dennis Ritchie."
    
    print(f"\nLLM Response: {poor_response}")
    
    result2 = pipeline.evaluate(
        question=question,
        context=context,
        llm_response=poor_response
    )
    
    print(pipeline.get_summary(result2))
