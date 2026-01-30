"""
Tests for the hallucination detection module.

These tests verify that the NLI-based hallucination detector
correctly identifies contradictions, entailments, and neutral cases.
"""

import pytest
from evaluators.hallucination import HallucinationDetector


class TestHallucinationDetector:
    """Test suite for HallucinationDetector"""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing"""
        return HallucinationDetector()
    
    def test_clear_contradiction(self, detector):
        """Test that clear contradictions are detected"""
        context = "The capital of France is Paris."
        response = "The capital of France is London."
        
        result = detector.detect(context, response, threshold=0.5)
        
        assert result["has_hallucination"] is True
        assert result["contradiction_score"] > 0.5
        assert result["label"] == "contradiction"
    
    def test_entailment(self, detector):
        """Test that correct information is recognized"""
        context = "Python was created by Guido van Rossum in 1991."
        response = "Python was created by Guido van Rossum."
        
        result = detector.detect(context, response, threshold=0.5)
        
        assert result["has_hallucination"] is False
        assert result["entailment_score"] > result["contradiction_score"]
        assert result["label"] == "entailment"
    
    def test_neutral_case(self, detector):
        """Test unverifiable claims"""
        context = "The sky is blue during the day."
        response = "The ocean is deep and mysterious."
        
        result = detector.detect(context, response, threshold=0.5)
        
        # Neutral cases should not be flagged as hallucinations
        assert result["has_hallucination"] is False
        # Neutral score should be relatively high
        assert result["neutral_score"] > 0.3
    
    def test_empty_inputs(self, detector):
        """Test error handling for empty inputs"""
        with pytest.raises(ValueError):
            detector.detect("", "Some response")
        
        with pytest.raises(ValueError):
            detector.detect("Some context", "")
    
    def test_threshold_sensitivity(self, detector):
        """Test that threshold affects detection"""
        context = "Water boils at 100 degrees Celsius."
        response = "Water boils at 95 degrees Celsius."
        
        # Stricter threshold (less sensitive)
        result_strict = detector.detect(context, response, threshold=0.8)
        
        # Lenient threshold (more sensitive)
        result_lenient = detector.detect(context, response, threshold=0.3)
        
        # At least one should detect the discrepancy
        assert result_lenient["contradiction_score"] > 0
    
    def test_sentence_level_detection(self, detector):
        """Test sentence-level hallucination analysis"""
        context = "Mars is the fourth planet from the Sun. It has two moons named Phobos and Deimos."
        # First sentence correct, second incorrect
        response = "Mars is the fourth planet from the Sun. It has five moons orbiting it."
        
        result = detector.detect_sentence_level(context, response, threshold=0.5)
        
        assert result["num_sentences"] == 2
        assert result["overall_has_hallucination"] is True
        # Should detect at least the incorrect sentence
        assert result["num_hallucinated"] >= 1
        assert 0 < result["hallucination_rate"] <= 1.0
    
    def test_multiple_sentences_all_correct(self, detector):
        """Test that correct multi-sentence responses pass"""
        context = "The Earth orbits the Sun. This takes about 365 days."
        response = "The Earth orbits the Sun. This process takes approximately one year."
        
        result = detector.detect_sentence_level(context, response, threshold=0.5)
        
        assert result["overall_has_hallucination"] is False
        assert result["num_hallucinated"] == 0
        assert result["hallucination_rate"] == 0.0


# Run tests with: pytest tests/test_evaluators.py -v
