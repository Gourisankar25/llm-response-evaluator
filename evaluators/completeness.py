"""
Answer Completeness Checker - Evaluates if answer addresses all question aspects

This module determines whether an LLM response fully addresses all aspects
of a question. A complete answer should cover every part of the query.

How it works:
1. Identify key aspects/components in the question
2. Check if each aspect is addressed in the response
3. Calculate completeness score = (Addressed aspects / Total aspects)

Examples of complete vs incomplete:

Question: "What are the symptoms and treatments for diabetes?"
Complete: "Diabetes symptoms include thirst and fatigue. Treatments involve insulin and diet."
Incomplete: "Diabetes causes increased thirst and hunger." (missing treatments)

Question: "Who invented the telephone and when?"
Complete: "Alexander Graham Bell invented the telephone in 1876."
Incomplete: "Alexander Graham Bell invented it." (missing when)

Techniques used:
- Question decomposition (identify sub-questions)
- NLI-based verification (check if aspect is addressed)
- Keyword/entity matching
"""

import re
from typing import Dict, List, Set
from evaluators.hallucination import HallucinationDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CompletenessChecker:
    """
    Evaluates whether an LLM response completely addresses a question.
    
    This checker identifies question aspects and verifies each is
    sufficiently addressed in the response.
    """
    
    def __init__(self, nli_model: str = "facebook/bart-large-mnli"):
        """
        Initialize the completeness checker.
        
        Args:
            nli_model: NLI model for aspect verification
        """
        logger.info("Initializing Completeness Checker...")
        
        # Use NLI to check if aspects are addressed
        self.detector = HallucinationDetector(model_name=nli_model)
        
        # Question word patterns for aspect identification
        self.question_patterns = {
            'what': ['what', 'which'],
            'when': ['when'],
            'where': ['where'],
            'who': ['who', 'whom'],
            'why': ['why'],
            'how': ['how'],
            'how_many': ['how many', 'how much']
        }
        
        logger.info("Completeness Checker ready")
    
    def identify_question_aspects(self, question: str) -> List[Dict]:
        """
        Identify different aspects/components of a question.
        
        A complex question often asks about multiple things:
        - "What and where?" → 2 aspects
        - "Who, when, and why?" → 3 aspects
        - "How does X work and what are its benefits?" → 2 aspects
        
        Args:
            question: The question to analyze
            
        Returns:
            List of aspect dictionaries with type and text
        """
        aspects = []
        question_lower = question.lower()
        
        # Method 1: Detect question words
        detected_types = set()
        for aspect_type, keywords in self.question_patterns.items():
            for keyword in keywords:
                if keyword in question_lower:
                    detected_types.add(aspect_type)
        
        # Create aspects from detected types
        for aspect_type in detected_types:
            aspects.append({
                'type': aspect_type,
                'description': f"Question asks '{aspect_type}'",
                'keywords': self.question_patterns[aspect_type]
            })
        
        # Method 2: Split on coordinating conjunctions (and, or)
        # "What is X and how does Y work?" → ["What is X", "how does Y work"]
        conjunctions = ['and', 'or', 'also', 'additionally']
        
        sub_questions = [question]
        for conj in conjunctions:
            temp = []
            for q in sub_questions:
                parts = re.split(f'\\s+{conj}\\s+', q, flags=re.IGNORECASE)
                temp.extend(parts)
            sub_questions = temp
        
        # Filter and clean sub-questions
        sub_questions = [q.strip() for q in sub_questions if len(q.strip()) > 10]
        
        # Add sub-questions as aspects if we found multiple
        if len(sub_questions) > 1:
            for i, sub_q in enumerate(sub_questions):
                aspects.append({
                    'type': 'sub_question',
                    'description': sub_q,
                    'text': sub_q
                })
        
        # If no aspects found, treat entire question as one aspect
        if not aspects:
            aspects.append({
                'type': 'general',
                'description': question,
                'text': question
            })
        
        logger.info(f"Identified {len(aspects)} question aspects")
        return aspects
    
    def check_aspect_addressed(
        self,
        aspect: Dict,
        response: str,
        entailment_threshold: float = 0.3
    ) -> Dict:
        """
        Check if a specific aspect is addressed in the response.
        
        Uses multiple heuristics to determine if aspect is addressed:
        1. Simple heuristic for common question types (what, when, etc.)
        2. NLI-based verification for complex cases
        
        Args:
            aspect: Aspect dictionary from identify_question_aspects
            response: LLM response to check
            entailment_threshold: Threshold for "addressed"
            
        Returns:
            Dictionary with addressed status and confidence
        """
        # Simple heuristic checks for common question types
        aspect_type = aspect['type']
        response_lower = response.lower()
        
        # Default: assume addressed unless proven otherwise
        is_addressed = False
        confidence = 0.0
        
        # Simple checks that work well for single-aspect questions
        if aspect_type == 'what' or aspect_type == 'general':
            # If response has substantial content (>20 words), likely answers "what"
            word_count = len(response.split())
            if word_count > 20:
                is_addressed = True
                confidence = 0.9
        elif aspect_type == 'when':
            # Check for time indicators (years, dates, time words)
            time_indicators = ['when', 'year', 'date', 'time', '19', '20', 'century', 'ago', 'before', 'after']
            has_time = any(indicator in response_lower for indicator in time_indicators)
            is_addressed = has_time
            confidence = 0.8 if has_time else 0.2
        elif aspect_type == 'where':
            # Check for location indicators
            location_indicators = ['where', 'in', 'at', 'located', 'place', 'country', 'city']
            has_location = any(indicator in response_lower for indicator in location_indicators)
            is_addressed = has_location
            confidence = 0.8 if has_location else 0.2
        elif aspect_type == 'who':
            # Check for people/entity references (proper nouns, names)
            has_person = bool(re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', response))  # Name pattern
            is_addressed = has_person or 'who' in response_lower
            confidence = 0.8 if has_person else 0.2
        elif aspect_type == 'why' or aspect_type == 'how':
            # Check for explanatory content (conjunctions, causation words)
            explanation_indicators = ['because', 'since', 'due to', 'by', 'through', 'using', 'via']
            has_explanation = any(indicator in response_lower for indicator in explanation_indicators)
            is_addressed = has_explanation or len(response.split()) > 15
            confidence = 0.7 if has_explanation else 0.5
        elif aspect_type == 'sub_question':
            # For sub-questions, use NLI
            aspect_text = aspect.get('text', aspect['description'])
            result = self.detector.detect(
                context=response,
                response=aspect_text,
                threshold=0.5
            )
            is_addressed = result['entailment_score'] > entailment_threshold
            confidence = result['entailment_score']
        else:
            # Default: if response is substantial, assume addressed
            word_count = len(response.split())
            is_addressed = word_count > 15
            confidence = min(0.9, word_count / 20)
        
        return {
            'aspect': aspect,
            'is_addressed': is_addressed,
            'confidence': confidence,
            'entailment_score': confidence
        }
    
    def score(
        self,
        question: str,
        response: str,
        entailment_threshold: float = 0.4
    ) -> Dict:
        """
        Calculate completeness score for a response.
        
        Args:
            question: The original question
            response: LLM response to evaluate
            entailment_threshold: Threshold for aspect addressed
            
        Returns:
            Dictionary containing:
                - completeness_score: float (0-1)
                - num_aspects: int
                - num_addressed: int
                - num_missing: int
                - addressed_aspects: List
                - missing_aspects: List
                - aspect_details: List
        """
        if not question or not response:
            raise ValueError("Both question and response must be non-empty")
        
        logger.info("Calculating completeness score...")
        
        # Identify question aspects
        aspects = self.identify_question_aspects(question)
        
        # Check each aspect
        addressed_aspects = []
        missing_aspects = []
        aspect_details = []
        
        for aspect in aspects:
            check_result = self.check_aspect_addressed(
                aspect,
                response,
                entailment_threshold
            )
            
            aspect_details.append(check_result)
            
            if check_result['is_addressed']:
                addressed_aspects.append(aspect)
            else:
                missing_aspects.append(aspect)
        
        # Calculate completeness score
        completeness_score = len(addressed_aspects) / len(aspects) if aspects else 0.0
        
        logger.info(
            f"Completeness: {completeness_score:.2%} "
            f"({len(addressed_aspects)}/{len(aspects)} aspects addressed)"
        )
        
        return {
            "completeness_score": completeness_score,
            "num_aspects": len(aspects),
            "num_addressed": len(addressed_aspects),
            "num_missing": len(missing_aspects),
            "addressed_aspects": addressed_aspects,
            "missing_aspects": missing_aspects,
            "aspect_details": aspect_details
        }
    
    def get_missing_information(
        self,
        question: str,
        response: str
    ) -> List[str]:
        """
        Get a list of what information is missing from the response.
        
        Args:
            question: Original question
            response: LLM response
            
        Returns:
            List of missing information descriptions
        """
        result = self.score(question, response)
        
        missing = []
        for aspect in result['missing_aspects']:
            if aspect['type'] == 'sub_question':
                missing.append(f"Doesn't answer: {aspect['text']}")
            else:
                missing.append(f"Missing {aspect['type']} information")
        
        return missing
    
    def suggest_improvements(
        self,
        question: str,
        response: str
    ) -> Dict:
        """
        Suggest improvements for an incomplete response.
        
        Args:
            question: Original question
            response: Current response
            
        Returns:
            Dictionary with suggestions and areas for improvement
        """
        result = self.score(question, response)
        
        suggestions = []
        
        if result['completeness_score'] < 1.0:
            suggestions.append(
                f"Response is {result['completeness_score']:.0%} complete. "
                f"Missing {result['num_missing']} aspect(s)."
            )
            
            for aspect in result['missing_aspects']:
                if aspect['type'] == 'sub_question':
                    suggestions.append(f"Add information about: {aspect['text']}")
                else:
                    suggestions.append(f"Include {aspect['type']} information")
        else:
            suggestions.append("Response appears complete!")
        
        return {
            "completeness_score": result['completeness_score'],
            "is_complete": result['completeness_score'] >= 0.9,
            "suggestions": suggestions,
            "missing_aspects": result['missing_aspects']
        }


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("✅ Answer Completeness Demo")
    print("=" * 60)
    
    checker = CompletenessChecker()
    
    # Example 1: Complete answer
    print("\n📌 Example 1: Complete Answer")
    print("-" * 60)
    question1 = "What is Python and when was it created?"
    response1 = "Python is a high-level programming language. It was created by Guido van Rossum in 1991."
    
    print(f"Question: {question1}")
    print(f"Response: {response1}")
    
    result1 = checker.score(question1, response1)
    print(f"\n✅ Results:")
    print(f"  - Completeness: {result1['completeness_score']:.2%}")
    print(f"  - Aspects Addressed: {result1['num_addressed']}/{result1['num_aspects']}")
    
    # Example 2: Incomplete answer
    print("\n📌 Example 2: Incomplete Answer")
    print("-" * 60)
    question2 = "What is Python and when was it created?"
    response2 = "Python is a high-level programming language."
    
    print(f"Question: {question2}")
    print(f"Response: {response2}")
    
    result2 = checker.score(question2, response2)
    print(f"\n✅ Results:")
    print(f"  - Completeness: {result2['completeness_score']:.2%}")
    print(f"  - Missing: {result2['num_missing']} aspect(s)")
    
    missing = checker.get_missing_information(question2, response2)
    if missing:
        print(f"\n  ⚠️ Missing Information:")
        for item in missing:
            print(f"    - {item}")
    
    # Example 3: Complex multi-part question
    print("\n📌 Example 3: Multi-Part Question")
    print("-" * 60)
    question3 = "Who invented the telephone, when was it invented, and how does it work?"
    response3 = "Alexander Graham Bell invented the telephone in 1876."
    
    print(f"Question: {question3}")
    print(f"Response: {response3}")
    
    result3 = checker.score(question3, response3)
    suggestions = checker.suggest_improvements(question3, response3)
    
    print(f"\n✅ Results:")
    print(f"  - Completeness: {result3['completeness_score']:.2%}")
    print(f"\n  💡 Suggestions:")
    for suggestion in suggestions['suggestions']:
        print(f"    - {suggestion}")
    
    # Example 4: Aspect breakdown
    print("\n📌 Example 4: Detailed Aspect Analysis")
    print("-" * 60)
    question4 = "What are the causes and symptoms of diabetes?"
    response4 = "Diabetes is caused by insufficient insulin production or insulin resistance. It can be genetic or lifestyle-related."
    
    print(f"Question: {question4}")
    print(f"Response: {response4}")
    
    result4 = checker.score(question4, response4)
    print(f"\n✅ Aspect Breakdown:")
    for i, detail in enumerate(result4['aspect_details'], 1):
        status = "✓" if detail['is_addressed'] else "✗"
        print(f"\n  {i}. {status} {detail['aspect']['type'].upper()}")
        print(f"     Confidence: {detail['confidence']:.3f}")
        print(f"     Status: {'Addressed' if detail['is_addressed'] else 'Missing'}")
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
