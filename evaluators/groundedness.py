"""
Groundedness Scoring - Measures claim support in LLM responses

This module evaluates what percentage of claims in an LLM response
are actually supported by the provided source context.

How it works:
1. Extract atomic claims from the response (individual factual statements)
2. Check each claim against context using NLI
3. Calculate: (Supported claims / Total claims) × 100%

This helps identify when an LLM adds unsupported information,
even if it doesn't directly contradict the context.

Example:
    Context: "Python is a programming language created in 1991."
    Response: "Python is a language from 1991. It's the most popular language worldwide."
    
    Claims: 2
    Supported: 1 ("Python is a language from 1991")
    Unsupported: 1 ("most popular worldwide" - not in context)
    Groundedness: 50%
"""

from typing import List, Dict, Tuple
import re
import logging
from evaluators.hallucination import HallucinationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroundednessScorer:
    """
    Scores the groundedness of LLM responses by checking claim support.
    
    Groundedness measures whether claims are backed by evidence in
    the source context, helping identify unsupported additions.
    """
    
    def __init__(self, nli_model: str = "facebook/bart-large-mnli"):
        """
        Initialize the groundedness scorer.
        
        Args:
            nli_model: NLI model for claim verification
        """
        logger.info("Initializing Groundedness Scorer...")
        
        # Reuse the hallucination detector for NLI inference
        self.detector = HallucinationDetector(model_name=nli_model)
        
        logger.info("Groundedness Scorer ready")
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract atomic claims from text.
        
        An atomic claim is a single factual statement that can be
        independently verified. This is a simplified extraction - 
        in production, you might use more sophisticated NLP.
        
        Current approach:
        - Split on sentence boundaries
        - Split compound sentences on conjunctions
        - Remove very short fragments
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of atomic claims
        """
        if not text:
            return []
        
        # Step 1: Split into sentences
        # Simple approach: split on ., !, ?
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        claims = []
        
        for sentence in sentences:
            # Step 2: Split compound sentences on coordinating conjunctions
            # "X and Y" → ["X", "Y"]
            # "X but Y" → ["X", "Y"]
            sub_claims = re.split(
                r'\s+(?:and|but|however|though|although)\s+', 
                sentence,
                flags=re.IGNORECASE
            )
            
            for claim in sub_claims:
                claim = claim.strip()
                
                # Filter out very short or incomplete claims
                # Must have at least 3 words and 15 characters
                words = claim.split()
                if len(words) >= 3 and len(claim) >= 15:
                    claims.append(claim)
        
        logger.info(f"Extracted {len(claims)} claims from text")
        return claims
    
    def check_claim_support(
        self, 
        claim: str, 
        context: str,
        entailment_threshold: float = 0.3
    ) -> Dict:
        """
        Check if a claim is supported by the context.
        
        Uses NLI to determine if the context entails the claim.
        
        Args:
            claim: Claim to verify
            context: Source context
            entailment_threshold: Minimum entailment score for support
            
        Returns:
            Dictionary with support status and scores
        """
        # Use NLI to check if context entails the claim
        result = self.detector.detect(context, claim)
        
        # A claim is "supported" if:
        # 1. It's entailed by the context (label = 'entailment')
        # 2. Entailment score exceeds threshold (lowered to 0.3 for more leniency)
        is_supported = (
            result['label'] == 'entailment' or 
            result['entailment_score'] > entailment_threshold
        )
        
        return {
            "claim": claim,
            "is_supported": is_supported,
            "entailment_score": result['entailment_score'],
            "neutral_score": result['neutral_score'],
            "contradiction_score": result['contradiction_score'],
            "nli_label": result['label']
        }
    
    def score(
        self,
        response: str,
        context: str,
        entailment_threshold: float = 0.3
    ) -> Dict:
        """
        Calculate the groundedness score for a response.
        
        Args:
            response: LLM response to evaluate
            context: Source context
            entailment_threshold: Threshold for claim support
            
        Returns:
            Dictionary containing:
                - groundedness_score: float (0-1)
                - num_claims: int
                - num_supported: int
                - num_unsupported: int
                - supported_claims: List[str]
                - unsupported_claims: List[str]
                - claim_details: List[Dict]
        """
        if not response or not context:
            raise ValueError("Both response and context must be non-empty")
        
        logger.info("Calculating groundedness score...")
        
        # Extract claims from response
        claims = self.extract_claims(response)
        
        if not claims:
            logger.warning("No claims extracted from response")
            return {
                "groundedness_score": 0.0,
                "num_claims": 0,
                "num_supported": 0,
                "num_unsupported": 0,
                "supported_claims": [],
                "unsupported_claims": [],
                "claim_details": []
            }
        
        # Check each claim
        supported_claims = []
        unsupported_claims = []
        claim_details = []
        
        for claim in claims:
            support_result = self.check_claim_support(
                claim, 
                context, 
                entailment_threshold
            )
            
            claim_details.append(support_result)
            
            if support_result['is_supported']:
                supported_claims.append(claim)
            else:
                unsupported_claims.append(claim)
        
        # Calculate groundedness score
        groundedness_score = len(supported_claims) / len(claims)
        
        logger.info(
            f"Groundedness: {groundedness_score:.2%} "
            f"({len(supported_claims)}/{len(claims)} claims supported)"
        )
        
        return {
            "groundedness_score": groundedness_score,
            "num_claims": len(claims),
            "num_supported": len(supported_claims),
            "num_unsupported": len(unsupported_claims),
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "claim_details": claim_details
        }
    
    def get_grounded_percentage(
        self,
        response: str,
        context: str,
        entailment_threshold: float = 0.5
    ) -> float:
        """
        Quick method to get just the groundedness percentage.
        
        Args:
            response: LLM response
            context: Source context
            entailment_threshold: Threshold for support
            
        Returns:
            Groundedness score as percentage (0-100)
        """
        result = self.score(response, context, entailment_threshold)
        return result['groundedness_score'] * 100


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("🎯 Groundedness Scoring Demo")
    print("=" * 60)
    
    scorer = GroundednessScorer()
    
    # Example 1: Fully grounded response
    print("\n📌 Example 1: Fully Grounded")
    print("-" * 60)
    context1 = (
        "The Python programming language was created by Guido van Rossum. "
        "It was first released in 1991. Python emphasizes code readability."
    )
    response1 = (
        "Python was created by Guido van Rossum. "
        "It was released in 1991."
    )
    
    print(f"Context: {context1}")
    print(f"Response: {response1}")
    
    result1 = scorer.score(response1, context1)
    print(f"\n✅ Results:")
    print(f"  - Groundedness Score: {result1['groundedness_score']:.2%}")
    print(f"  - Supported Claims: {result1['num_supported']}/{result1['num_claims']}")
    
    # Example 2: Partially grounded response
    print("\n📌 Example 2: Partially Grounded")
    print("-" * 60)
    response2 = (
        "Python was created by Guido van Rossum in 1991. "
        "It is the most popular programming language in the world. "
        "Python has 50 million developers globally."
    )
    
    print(f"Context: {context1}")
    print(f"Response: {response2}")
    
    result2 = scorer.score(response2, context1)
    print(f"\n✅ Results:")
    print(f"  - Groundedness Score: {result2['groundedness_score']:.2%}")
    print(f"  - Supported: {result2['num_supported']}")
    print(f"  - Unsupported: {result2['num_unsupported']}")
    
    if result2['unsupported_claims']:
        print(f"\n  ⚠️ Unsupported claims:")
        for claim in result2['unsupported_claims']:
            print(f"    - \"{claim}\"")
    
    # Example 3: Detailed claim analysis
    print("\n📌 Example 3: Detailed Analysis")
    print("-" * 60)
    context3 = (
        "Mars is the fourth planet from the Sun. "
        "It has two moons named Phobos and Deimos. "
        "Mars appears red due to iron oxide on its surface."
    )
    response3 = (
        "Mars is the fourth planet. "
        "It has two small moons. "
        "Mars is red because of rust. "
        "It has liquid water on the surface."
    )
    
    print(f"Context: {context3}")
    print(f"Response: {response3}")
    
    result3 = scorer.score(response3, context3)
    print(f"\n✅ Results:")
    print(f"  - Groundedness: {result3['groundedness_score']:.2%}")
    print(f"\n  📊 Claim-by-Claim Analysis:")
    
    for i, detail in enumerate(result3['claim_details'], 1):
        status = "✓ Supported" if detail['is_supported'] else "✗ Unsupported"
        print(f"    {i}. {status}")
        print(f"       Claim: \"{detail['claim']}\"")
        print(f"       Entailment: {detail['entailment_score']:.3f}")
        print()
    
    print("=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
