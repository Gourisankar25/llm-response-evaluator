# 🚀 Quick Start Guide

Follow these steps to get the evaluation system running on your machine!

## Step 1: Set Up Environment

Open PowerShell or Command Prompt and navigate to the project folder:

```powershell
cd c:\Users\gourig\Documents\LLM_eval
```

Create and activate a virtual environment:

```powershell
# Create virtual environment
python -m venv venv

# Activate it (Windows PowerShell)
venv\Scripts\activate

# You should see (venv) in your prompt now
```

## Step 2: Install Dependencies

```powershell
# Install all required packages (this will take 5-10 minutes)
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

**Note**: The first time you run the evaluators, models will download (~1-2 GB). Be patient!

## Step 3: Test Individual Components

Let's test each evaluator to make sure everything works:

### Test Hallucination Detector

```powershell
python evaluators/hallucination.py
```

You should see output showing hallucination detection examples. If it works, you're good! ✅

### Test Groundedness Scorer

```powershell
python evaluators/groundedness.py
```

### Test Faithfulness Scorer

```powershell
python evaluators/faithfulness.py
```

### Test Relevance Checker

```powershell
python evaluators/relevance.py
```

### Test Completeness Checker

```powershell
python evaluators/completeness.py
```

### Test Complete Pipeline

```powershell
python evaluators/pipeline.py
```

This runs all evaluators together and shows a complete evaluation!

## Step 4: Start the API Server

In your terminal (with venv activated):

```powershell
python -m uvicorn app.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

**Visit the API docs**: Open your browser and go to:
- http://localhost:8000/docs (Swagger UI - interactive documentation)

You can test the API directly from the browser! Try the `/api/evaluate` endpoint.

## Step 5: Start the Dashboard

Open a **NEW terminal window** (keep the API running in the first one!)

```powershell
# Navigate to project
cd c:\Users\gourig\Documents\LLM_eval

# Activate venv
venv\Scripts\activate

# Start Streamlit
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at: http://localhost:8501

## Step 6: Test the System

### Option A: Use the Dashboard

1. Enter a question, context, and LLM response in the dashboard
2. Click "Evaluate Response"
3. See the results!

### Option B: Use the API Directly

Use PowerShell to test with curl or create a Python script:

```python
# test_api.py
import requests
import json

response = requests.post("http://localhost:8000/api/evaluate", json={
    "question": "What is the capital of France?",
    "context": "France is a country in Europe. Paris is its capital.",
    "llm_response": "The capital of France is Paris."
})

print(json.dumps(response.json(), indent=2))
```

Run it:
```powershell
python test_api.py
```

## Common Issues & Solutions

### Issue: "Module not found" error

**Solution**: Make sure venv is activated and packages are installed:
```powershell
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "CUDA not available" warning

**Solution**: This is fine! Models will run on CPU. It's slower but works perfectly.

### Issue: "Port 8000 already in use"

**Solution**: Another process is using that port. Either:
- Stop the other process
- Or use a different port: `uvicorn app.main:app --port 8001`

### Issue: Models downloading during first run

**Solution**: This is normal! The first run downloads models (~1-2 GB). Subsequent runs will be fast.

### Issue: Evaluation is slow

**Solution**: 
- CPU evaluation takes 10-30 seconds per request
- If you have a GPU (NVIDIA), the models will automatically use it and be much faster
- Consider running evaluations in batches

## What's Next?

Now that everything is working, try:

1. **Experiment**: Test different questions and responses
2. **Adjust thresholds**: See how they affect scoring
3. **Read the code**: Each file has detailed explanations
4. **Add features**: Ideas:
   - Database integration (save evaluation history)
   - Batch evaluation (multiple responses at once)
   - Custom metrics
   - Export results to CSV

## Running Tests

To run the test suite:

```powershell
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=evaluators --cov=app

# Run specific test file
pytest tests/test_evaluators.py
```

## Stopping the Services

To stop the services:

1. **API Server**: Press `Ctrl+C` in the terminal running uvicorn
2. **Dashboard**: Press `Ctrl+C` in the terminal running streamlit

## Daily Workflow

When you come back to work on this:

```powershell
# 1. Navigate to project
cd c:\Users\gourig\Documents\LLM_eval

# 2. Activate venv
venv\Scripts\activate

# 3. Start API (Terminal 1)
python -m uvicorn app.main:app --reload

# 4. Start Dashboard (Terminal 2)
streamlit run dashboard/app.py
```

## Learning Resources

- **FastAPI Tutorial**: https://fastapi.tiangolo.com/tutorial/
- **Streamlit Docs**: https://docs.streamlit.io/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Sentence Transformers**: https://www.sbert.net/

## Questions?

If something doesn't work:
1. Check the error message carefully
2. Make sure venv is activated
3. Try restarting the services
4. Check if all dependencies are installed

Happy evaluating! 🎉
