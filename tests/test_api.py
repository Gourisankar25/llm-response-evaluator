"""
Additional tests for API endpoints

These tests verify that the FastAPI endpoints work correctly.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# Create test client
client = TestClient(app)


class TestAPI:
    """Test suite for API endpoints"""
    
    def test_root_endpoint(self):
        """Test the root endpoint returns basic info"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_evaluate_endpoint_valid_request(self):
        """Test evaluation with valid input"""
        payload = {
            "question": "What is Python?",
            "context": "Python is a high-level programming language created by Guido van Rossum in 1991.",
            "llm_response": "Python is a programming language created by Guido van Rossum.",
            "hallucination_threshold": 0.5,
            "groundedness_threshold": 0.5,
            "relevance_threshold": 0.5
        }
        
        response = client.post("/api/evaluate", json=payload, timeout=60)
        
        # Note: This will take a while on first run (model loading)
        # You might want to skip this in CI or mock the pipeline
        assert response.status_code == 200
        
        data = response.json()
        assert "overall_score" in data
        assert "quality_tier" in data
        assert "hallucination" in data
        assert "groundedness" in data
        assert "faithfulness" in data
        assert "relevance" in data
        assert "completeness" in data
    
    def test_evaluate_endpoint_missing_fields(self):
        """Test that missing required fields return 422"""
        payload = {
            "question": "What is Python?"
            # Missing context and llm_response
        }
        
        response = client.post("/api/evaluate", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_evaluate_endpoint_invalid_threshold(self):
        """Test that invalid threshold values return 422"""
        payload = {
            "question": "What is Python?",
            "context": "Python is a programming language.",
            "llm_response": "Python is a language.",
            "hallucination_threshold": 1.5  # Invalid: > 1.0
        }
        
        response = client.post("/api/evaluate", json=payload)
        assert response.status_code == 422
    
    def test_evaluate_endpoint_empty_strings(self):
        """Test that empty strings are rejected"""
        payload = {
            "question": "",
            "context": "Some context",
            "llm_response": "Some response"
        }
        
        response = client.post("/api/evaluate", json=payload)
        assert response.status_code == 422


# Additional integration tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.slow
    def test_full_evaluation_flow(self):
        """Test complete evaluation flow with real models"""
        # This is a slower test that actually runs the models
        payload = {
            "question": "What is the capital of France?",
            "context": "France is a country in Western Europe. Its capital is Paris.",
            "llm_response": "The capital of France is Paris.",
        }
        
        response = client.post("/api/evaluate", json=payload, timeout=60)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check overall score is reasonable
        assert 0 <= data["overall_score"] <= 1
        assert 0 <= data["overall_score_100"] <= 100
        
        # Check all metrics are present
        assert "hallucination" in data
        assert "groundedness" in data
        assert "faithfulness" in data
        assert "relevance" in data
        assert "completeness" in data
        
        # For a correct answer, we expect good scores
        assert data["overall_score_100"] > 50  # Should be decent


# To run only fast tests:
# pytest -m "not slow"

# To run all tests including slow ones:
# pytest

# To run with coverage:
# pytest --cov=app --cov=evaluators
