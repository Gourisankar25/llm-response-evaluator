"""
FastAPI Main Application

This is the entry point for the API server. It sets up the FastAPI app,
configures CORS, and defines the main routes.

What is FastAPI?
- Modern Python web framework for building APIs
- Automatic data validation using Pydantic
- Auto-generated API documentation (Swagger UI)
- High performance (similar to NodeJS/Go)

Key concepts:
- @app.get/post: Route decorators (define URL endpoints)
- async def: Async functions (handle multiple requests efficiently)
- Dependency injection: Reusable components (like pipeline instance)

To run:
    uvicorn app.main:app --reload
    
Then visit:
    - http://localhost:8000/docs (Swagger UI - interactive API docs)
    - http://localhost:8000/redoc (ReDoc - alternative docs)
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import Optional

from app.models import (
    EvaluationRequest,
    EvaluationResponse,
    HealthResponse,
    ErrorResponse,
    HallucinationMetric,
    GroundednessMetric,
    FaithfulnessMetric,
    RelevanceMetric,
    CompletenessMetric
)
from evaluators.pipeline import LLMEvaluationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LLM Response Evaluation API",
    description="""
    A production-ready API for evaluating LLM responses across multiple quality metrics.
    
    ## Features
    * 🔍 Hallucination Detection - Detect contradictions and fabrications
    * 📊 Groundedness Scoring - Verify claims are supported by context
    * 💎 Faithfulness Assessment - Measure semantic similarity
    * 🎯 Context Relevance - Check if context helps answer the question
    * ✅ Completeness Check - Ensure all aspects are addressed
    
    ## Usage
    Send a POST request to `/api/evaluate` with your question, context, and LLM response.
    The API will return comprehensive evaluation metrics and an overall quality score.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows your frontend (like Streamlit) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
# We initialize it once and reuse it (models are heavy to load!)
pipeline: Optional[LLMEvaluationPipeline] = None


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler.
    
    This runs when the server starts. We use it to:
    1. Load ML models into memory
    2. Initialize the evaluation pipeline
    3. Warm up the models (first inference is slow)
    """
    global pipeline
    
    logger.info("=" * 60)
    logger.info("🚀 Starting LLM Evaluation API")
    logger.info("=" * 60)
    
    try:
        logger.info("Loading evaluation pipeline...")
        pipeline = LLMEvaluationPipeline()
        logger.info("✅ Pipeline loaded successfully!")
        
        # Warm up (optional - makes first request faster)
        logger.info("Warming up models...")
        _ = pipeline.evaluate(
            question="Test",
            context="This is a test context.",
            llm_response="This is a test response."
        )
        logger.info("✅ Models warmed up!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
        raise
    
    logger.info("=" * 60)
    logger.info("✅ API is ready to accept requests")
    logger.info("=" * 60)


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - Basic welcome message
    """
    return {
        "message": "LLM Response Evaluation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Use this to verify the API is running and models are loaded.
    """
    return HealthResponse(
        status="healthy",
        message="API is running and ready",
        timestamp=datetime.now(),
        models_loaded=pipeline is not None
    )


@app.post(
    "/api/evaluate",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
    summary="Evaluate an LLM response",
    description="""
    Evaluate an LLM response across 5 key metrics:
    1. Hallucination Detection
    2. Groundedness
    3. Faithfulness  
    4. Context Relevance
    5. Answer Completeness
    
    Returns a comprehensive evaluation with an overall score (0-100).
    """
)
async def evaluate_response(request: EvaluationRequest):
    """
    Main evaluation endpoint.
    
    This is where the magic happens! Takes a question, context, and LLM response,
    runs all evaluations, and returns comprehensive results.
    
    Args:
        request: EvaluationRequest with question, context, and LLM response
        
    Returns:
        EvaluationResponse with all metrics and overall score
    """
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Evaluation pipeline not initialized. Please wait and try again."
        )
    
    try:
        logger.info(f"Received evaluation request for question: {request.question[:50]}...")
        
        # Run evaluation
        result = pipeline.evaluate(
            question=request.question,
            context=request.context,
            llm_response=request.llm_response,
            hallucination_threshold=request.hallucination_threshold,
            groundedness_threshold=request.groundedness_threshold,
            relevance_threshold=request.relevance_threshold
        )
        
        # Convert to response model
        # This ensures the response matches our defined schema
        response = EvaluationResponse(
            overall_score=result['overall_score'],
            overall_score_100=result['overall_score_100'],
            quality_tier=result['quality_tier'],
            emoji=result['emoji'],
            hallucination=HallucinationMetric(**result['metrics']['hallucination']),
            groundedness=GroundednessMetric(**result['metrics']['groundedness']),
            faithfulness=FaithfulnessMetric(**result['metrics']['faithfulness']),
            relevance=RelevanceMetric(**result['metrics']['relevance']),
            completeness=CompletenessMetric(**result['metrics']['completeness']),
            issues=result['issues'],
            has_issues=result['has_issues'],
            weights=result['weights'],
            execution_time_seconds=result['execution_time_seconds']
        )
        
        logger.info(f"Evaluation complete - Score: {result['overall_score_100']:.1f}/100")
        
        return response
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors gracefully"""
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )


# For direct script execution
if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("🚀 Starting LLM Evaluation API Server")
    print("=" * 60)
    print("\n📖 API Documentation will be available at:")
    print("   - http://localhost:8000/docs (Swagger UI)")
    print("   - http://localhost:8000/redoc (ReDoc)")
    print("\n✋ Press CTRL+C to stop the server\n")
    print("=" * 60 + "\n")
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )
