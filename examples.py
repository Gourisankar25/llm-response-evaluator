"""
Example Script: Using the Evaluation Pipeline Directly

This script shows how to use the evaluation pipeline without the API.
Useful for batch processing, notebooks, or integrating into other systems.
"""

from evaluators.pipeline import LLMEvaluationPipeline, EvaluationWeights

def example_1_basic_usage():
    """Example 1: Basic usage with default settings"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = LLMEvaluationPipeline()
    
    # Your data
    question = "What is machine learning?"
    
    context = """
    Machine learning is a subset of artificial intelligence (AI) that
    enables systems to learn and improve from experience without being
    explicitly programmed. It focuses on developing computer programs
    that can access data and use it to learn for themselves.
    """
    
    llm_response = """
    Machine learning is a branch of AI that allows systems to automatically
    learn and improve from experience. It uses algorithms to find patterns
    in data and make predictions or decisions without being explicitly programmed.
    """
    
    # Run evaluation
    result = pipeline.evaluate(question, context, llm_response)
    
    # Print summary
    print(pipeline.get_summary(result))
    
    # Access specific metrics
    print("\n📊 Detailed Metrics:")
    print(f"Hallucination Score: {result['metrics']['hallucination']['score']:.2%}")
    print(f"Groundedness Score: {result['metrics']['groundedness']['score']:.2%}")
    print(f"Faithfulness Score: {result['metrics']['faithfulness']['score']:.2%}")
    print(f"Relevance Score: {result['metrics']['relevance']['score']:.2%}")
    print(f"Completeness Score: {result['metrics']['completeness']['score']:.2%}")


def example_2_custom_weights():
    """Example 2: Using custom metric weights"""
    print("\n" + "=" * 60)
    print("Example 2: Custom Weights")
    print("=" * 60)
    
    # Define custom weights
    # Let's prioritize hallucination detection and groundedness
    custom_weights = EvaluationWeights(
        hallucination=0.35,  # Increased from 0.25
        groundedness=0.30,   # Increased from 0.20
        faithfulness=0.15,   # Decreased from 0.20
        relevance=0.10,      # Decreased from 0.15
        completeness=0.10    # Decreased from 0.20
    )
    
    # Initialize with custom weights
    pipeline = LLMEvaluationPipeline(weights=custom_weights)
    
    # Same example as before
    question = "What is machine learning?"
    context = "Machine learning is a subset of AI that enables systems to learn from data."
    llm_response = "Machine learning is a branch of AI for learning from data."
    
    result = pipeline.evaluate(question, context, llm_response)
    
    print(f"\n✅ Overall Score: {result['overall_score_100']:.1f}/100")
    print(f"Quality Tier: {result['quality_tier']}")
    print(f"\nCustom Weights Used:")
    for metric, weight in result['weights'].items():
        print(f"  - {metric}: {weight:.0%}")


def example_3_detecting_issues():
    """Example 3: Detecting quality issues"""
    print("\n" + "=" * 60)
    print("Example 3: Detecting Issues")
    print("=" * 60)
    
    pipeline = LLMEvaluationPipeline()
    
    # Poor quality response with multiple issues
    question = "What is the capital of France and when was the Eiffel Tower built?"
    
    context = """
    France is a country in Western Europe. Paris is the capital of France.
    The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair.
    """
    
    # Response with: hallucination, unsupported claim, incomplete answer
    llm_response = """
    The capital of France is London. The Eiffel Tower was constructed in 1889
    and is the tallest structure in Europe.
    """
    
    result = pipeline.evaluate(question, context, llm_response)
    
    print(f"\n{result['emoji']} Overall Score: {result['overall_score_100']:.1f}/100")
    print(f"Quality: {result['quality_tier']}")
    
    if result['has_issues']:
        print(f"\n⚠️ Issues Detected ({len(result['issues'])}):")
        for i, issue in enumerate(result['issues'], 1):
            print(f"  {i}. {issue}")
    
    # Check specific problems
    if result['metrics']['hallucination']['has_hallucination']:
        print(f"\n❌ Hallucination: {result['metrics']['hallucination']['label']}")
    
    if result['metrics']['groundedness']['unsupported_claims']:
        print(f"\n⚠️ Unsupported Claims:")
        for claim in result['metrics']['groundedness']['unsupported_claims']:
            print(f"  - {claim}")
    
    if result['metrics']['completeness']['missing_aspects']:
        print(f"\n📋 Missing Information:")
        for aspect in result['metrics']['completeness']['missing_aspects']:
            print(f"  - {aspect}")


def example_4_batch_evaluation():
    """Example 4: Batch evaluation of multiple responses"""
    print("\n" + "=" * 60)
    print("Example 4: Batch Evaluation")
    print("=" * 60)
    
    pipeline = LLMEvaluationPipeline()
    
    # Multiple test cases
    test_cases = [
        {
            "name": "Good Response",
            "question": "What is Python?",
            "context": "Python is a high-level programming language created in 1991.",
            "response": "Python is a high-level programming language."
        },
        {
            "name": "Hallucinated Response",
            "question": "What is Python?",
            "context": "Python is a high-level programming language created in 1991.",
            "response": "Python is a low-level language created in 1985."
        },
        {
            "name": "Incomplete Response",
            "question": "What is Python and who created it?",
            "context": "Python is a programming language created by Guido van Rossum.",
            "response": "Python is a programming language."
        }
    ]
    
    results_summary = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {test['name']}")
        print("-" * 40)
        
        result = pipeline.evaluate(
            test['question'],
            test['context'],
            test['response']
        )
        
        results_summary.append({
            'name': test['name'],
            'score': result['overall_score_100'],
            'tier': result['quality_tier'],
            'has_issues': result['has_issues']
        })
        
        print(f"Score: {result['overall_score_100']:.1f}/100 ({result['quality_tier']})")
        print(f"Issues: {len(result['issues'])}")
    
    # Summary table
    print("\n" + "=" * 60)
    print("📊 Batch Results Summary")
    print("=" * 60)
    print(f"{'Test Case':<25} {'Score':<10} {'Quality':<15} {'Issues'}")
    print("-" * 60)
    for res in results_summary:
        issues_str = "Yes" if res['has_issues'] else "No"
        print(f"{res['name']:<25} {res['score']:<10.1f} {res['tier']:<15} {issues_str}")


def example_5_programmatic_access():
    """Example 5: Accessing detailed metric data programmatically"""
    print("\n" + "=" * 60)
    print("Example 5: Programmatic Access")
    print("=" * 60)
    
    pipeline = LLMEvaluationPipeline()
    
    question = "What causes diabetes?"
    context = "Diabetes is caused by insufficient insulin production or insulin resistance."
    llm_response = "Diabetes occurs when the body cannot produce enough insulin or use it effectively."
    
    result = pipeline.evaluate(question, context, llm_response)
    
    # Extract specific data
    print("\n📊 Extracting Specific Data:\n")
    
    # 1. Overall metrics
    print(f"1. Overall Quality: {result['overall_score_100']:.1f}/100")
    
    # 2. Individual metric scores
    print(f"\n2. Individual Scores:")
    for metric_name in ['hallucination', 'groundedness', 'faithfulness', 'relevance', 'completeness']:
        score = result['metrics'][metric_name]['score']
        print(f"   {metric_name.capitalize()}: {score:.2%}")
    
    # 3. Detailed groundedness info
    print(f"\n3. Groundedness Details:")
    ground = result['metrics']['groundedness']
    print(f"   Total Claims: {ground['num_claims']}")
    print(f"   Supported: {ground['num_supported']}")
    print(f"   Unsupported: {len(ground['unsupported_claims'])}")
    
    # 4. Completeness details
    print(f"\n4. Completeness Details:")
    comp = result['metrics']['completeness']
    print(f"   Question Aspects: {comp['num_aspects']}")
    print(f"   Addressed: {comp['num_addressed']}")
    print(f"   Missing: {len(comp['missing_aspects'])}")
    
    # 5. Performance metrics
    print(f"\n5. Performance:")
    print(f"   Execution Time: {result['execution_time_seconds']:.2f}s")
    
    # 6. Export to dict for storage/further processing
    print(f"\n6. Ready for Export/Storage:")
    print(f"   Keys available: {list(result.keys())}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 LLM Evaluation Pipeline - Example Usage")
    print("=" * 60)
    
    # Run all examples
    example_1_basic_usage()
    example_2_custom_weights()
    example_3_detecting_issues()
    example_4_batch_evaluation()
    example_5_programmatic_access()
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Modify these examples with your own data")
    print("  2. Try different thresholds and weights")
    print("  3. Integrate into your own applications")
    print("  4. Build custom metrics on top of these evaluators")
    print("\nHappy evaluating! 🎉\n")
