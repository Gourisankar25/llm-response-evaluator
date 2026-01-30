"""
Hallucination Detection using Natural Language Inference (NLI)

This module detects hallucinations in LLM responses by checking if claims
contradict the provided source context using NLI models.

How it works:
1. Take the source context as the "premise" (ground truth)
2. Take claims from the LLM response as "hypothesis" (what we're checking)
3. Use an NLI model to classify the relationship:
   - ENTAILMENT: Claim is supported by context ✓
   - NEUTRAL: Can't determine from context (unverifiable)
   - CONTRADICTION: Claim contradicts context (HALLUCINATION!)

Model: facebook/bart-large-mnli
- Trained on Multi-Genre NLI (MNLI) dataset
- 406M parameters, high accuracy
- Returns probabilities for each class
"""

from transformers import pipeline
import torch
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Detects hallucinations in LLM outputs using NLI models.
    
    The detector identifies contradictions between the LLM response
    and the provided source context.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: Optional[str] = None):
        """
        Initialize the hallucination detector.
        
        Args:
            model_name: HuggingFace model identifier for NLI
            device: 'cuda' for GPU, 'cpu' for CPU, None for auto-detect
        """
        logger.info(f"Loading NLI model: {model_name}")
        
        # Auto-detect device if not specified
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        
        # Load the NLI pipeline
        # Pipeline is a high-level HuggingFace API that handles:
        # - Tokenization (converting text to numbers)
        # - Model inference (running the neural network)
        # - Post-processing (converting outputs to readable format)
        self.nli_pipeline = pipeline(
            "text-classification",  # NLI task type
            model=model_name,
            device=device,
            top_k=3  # Return scores for all 3 classes
        )
        
        logger.info(f"Model loaded on device: {'GPU' if device == 0 else 'CPU'}")
    
    def detect(
        self, 
        context: str, 
        response: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Detect hallucinations in an LLM response.
        
        Args:
            context: Source context (ground truth)
            response: LLM-generated response to check
            threshold: Confidence threshold for contradiction (0-1)
                      Higher = stricter detection
        
        Returns:
            Dictionary containing:
                - has_hallucination: bool
                - contradiction_score: float (0-1)
                - entailment_score: float (0-1)
                - neutral_score: float (0-1)
                - label: str ('entailment', 'neutral', or 'contradiction')
                - confidence: float
        """
        if not context or not response:
            raise ValueError("Both context and response must be non-empty strings")
        
        logger.info("Running hallucination detection...")
        
        # Run NLI inference
        # The model checks if the response (hypothesis) follows from context (premise)
        # Format: "premise </s></s> hypothesis" for BART-based NLI models
        nli_input = f"{context} </s></s> {response}"
        result = self.nli_pipeline(nli_input)
        
        # Extract scores for each class
        # Result format: [[{'label': 'LABEL_X', 'score': 0.8}, ...]]
        # For facebook/bart-large-mnli:
        #   LABEL_0 = contradiction
        #   LABEL_1 = neutral
        #   LABEL_2 = entailment
        
        label_mapping = {
            'LABEL_0': 'contradiction',
            'LABEL_1': 'neutral',
            'LABEL_2': 'entailment'
        }
        
        scores = {'contradiction': 0.0, 'neutral': 0.0, 'entailment': 0.0}
        predicted_label = None
        confidence = 0.0
        
        # result[0] gives us the list of label-score dicts
        for item in result[0]:
            mapped_label = label_mapping.get(item['label'], item['label'].lower())
            score = item['score']
            scores[mapped_label] = score
            if score > confidence:
                confidence = score
                predicted_label = mapped_label
        
        # Determine if hallucination detected
        contradiction_score = scores.get('contradiction', 0.0)
        has_hallucination = contradiction_score > threshold
        
        logger.info(
            f"Detection complete - Label: {predicted_label}, "
            f"Confidence: {confidence:.3f}, "
            f"Hallucination: {has_hallucination}"
        )
        
        return {
            "has_hallucination": has_hallucination,
            "contradiction_score": contradiction_score,
            "entailment_score": scores.get('entailment', 0.0),
            "neutral_score": scores.get('neutral', 0.0),
            "label": predicted_label,
            "confidence": confidence,
            "threshold": threshold
        }
    
    def detect_sentence_level(
        self,
        context: str,
        response: str,
        threshold: float = 0.5
    ) -> Dict:
        """
        Detect hallucinations at sentence level for more granular analysis.
        
        This splits the response into sentences and checks each one
        individually against the context.
        
        Args:
            context: Source context
            response: LLM response
            threshold: Contradiction threshold
            
        Returns:
            Dictionary with overall results and per-sentence analysis
        """
        # Split response into sentences
        # Simple split on periods (we'll improve this with spaCy later)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        sentence_results = []
        total_contradiction = 0.0
        hallucinated_sentences = []
        
        logger.info(f"Analyzing {len(sentences)} sentences...")
        
        for idx, sentence in enumerate(sentences):
            # Check each sentence against context
            result = self.detect(context, sentence, threshold)
            
            sentence_results.append({
                "sentence": sentence,
                "index": idx,
                "has_hallucination": result["has_hallucination"],
                "contradiction_score": result["contradiction_score"],
                "label": result["label"]
            })
            
            if result["has_hallucination"]:
                hallucinated_sentences.append(sentence)
                total_contradiction += result["contradiction_score"]
        
        # Calculate average contradiction score
        avg_contradiction = (
            total_contradiction / len(sentences) if sentences else 0.0
        )
        
        return {
            "overall_has_hallucination": len(hallucinated_sentences) > 0,
            "num_sentences": len(sentences),
            "num_hallucinated": len(hallucinated_sentences),
            "hallucination_rate": len(hallucinated_sentences) / len(sentences) if sentences else 0.0,
            "avg_contradiction_score": avg_contradiction,
            "hallucinated_sentences": hallucinated_sentences,
            "sentence_analysis": sentence_results
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🔍 Hallucination Detection Demo")
    print("=" * 60)
    
    # Initialize detector
    detector = HallucinationDetector()
    
    # Example 1: Clear hallucination
    print("\n📌 Example 1: Clear Contradiction")
    print("-" * 60)
    context1 = "Python was created by Guido van Rossum and first released in 1991."
    response1 = "Python was created by Dennis Ritchie in 1985."
    
    print(f"Context: {context1}")
    print(f"Response: {response1}")
    
    result1 = detector.detect(context1, response1)
    print(f"\n✅ Results:")
    print(f"  - Has Hallucination: {result1['has_hallucination']}")
    print(f"  - Contradiction Score: {result1['contradiction_score']:.3f}")
    print(f"  - Label: {result1['label']}")
    
    # Example 2: Correct response
    print("\n📌 Example 2: Correct Information")
    print("-" * 60)
    response2 = "Python was created by Guido van Rossum."
    
    print(f"Context: {context1}")
    print(f"Response: {response2}")
    
    result2 = detector.detect(context1, response2)
    print(f"\n✅ Results:")
    print(f"  - Has Hallucination: {result2['has_hallucination']}")
    print(f"  - Entailment Score: {result2['entailment_score']:.3f}")
    print(f"  - Label: {result2['label']}")
    
    # Example 3: Sentence-level detection
    print("\n📌 Example 3: Sentence-Level Analysis")
    print("-" * 60)
    context3 = "The Eiffel Tower is located in Paris, France. It was completed in 1889."
    response3 = "The Eiffel Tower is in Paris. It was built in 1850. It's made of steel."
    
    print(f"Context: {context3}")
    print(f"Response: {response3}")
    
    result3 = detector.detect_sentence_level(context3, response3)
    print(f"\n✅ Results:")
    print(f"  - Overall Hallucination: {result3['overall_has_hallucination']}")
    print(f"  - Hallucination Rate: {result3['hallucination_rate']:.1%}")
    print(f"  - Hallucinated Sentences: {len(result3['hallucinated_sentences'])}/{result3['num_sentences']}")
    
    if result3['hallucinated_sentences']:
        print(f"\n  ⚠️ Problematic sentences:")
        for sentence in result3['hallucinated_sentences']:
            print(f"    - \"{sentence}\"")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
