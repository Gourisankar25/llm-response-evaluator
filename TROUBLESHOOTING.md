# 🔧 Troubleshooting Guide

Common issues and solutions when working with the LLM Evaluation Framework.

---

## Installation Issues

### Issue: "pip install fails"

**Symptoms:**
```
ERROR: Could not build wheels for [package]
```

**Solutions:**
1. Upgrade pip:
   ```powershell
   python -m pip install --upgrade pip
   ```

2. Install Visual C++ Build Tools (Windows):
   - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

3. Try installing problematic packages individually:
   ```powershell
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install transformers
   ```

### Issue: "Virtual environment activation fails"

**Symptoms:**
```
The term 'activate' is not recognized...
```

**Solutions:**
1. Make sure you're in the project directory:
   ```powershell
   cd c:\Users\gourig\Documents\LLM_eval
   ```

2. Use the full path:
   ```powershell
   venv\Scripts\activate
   ```

3. Try PowerShell instead of CMD:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

4. If execution policy error:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

### Issue: "spaCy model download fails"

**Symptoms:**
```
Can't find model 'en_core_web_sm'
```

**Solution:**
```powershell
python -m spacy download en_core_web_sm --user
```

Or download manually:
```powershell
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
```

---

## Model Loading Issues

### Issue: "Model download is very slow"

**Symptoms:**
- First run takes 10-30 minutes
- Models downloading during execution

**Solutions:**
1. **This is normal!** Models are 1-2 GB total.
2. Be patient on first run
3. Subsequent runs will be fast (models cached)
4. Use a stable internet connection

**Verify cache location:**
```powershell
# Check if models are cached
dir $env:USERPROFILE\.cache\huggingface
```

### Issue: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Force CPU usage:
   ```python
   # In evaluators, modify device parameter
   detector = HallucinationDetector(device='cpu')
   ```

2. Or set environment variable:
   ```powershell
   $env:CUDA_VISIBLE_DEVICES = "-1"
   ```

### Issue: "Model loading fails"

**Symptoms:**
```
OSError: Can't load model...
```

**Solutions:**
1. Clear cache:
   ```powershell
   Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface
   ```

2. Reinstall transformers:
   ```powershell
   pip uninstall transformers
   pip install transformers
   ```

3. Check internet connection (models download from HuggingFace)

---

## API Issues

### Issue: "Cannot connect to API"

**Symptoms:**
```
ConnectionError: Cannot connect to http://localhost:8000
```

**Solutions:**
1. Check if API is running:
   - Look for "Uvicorn running on..." message
   - Try accessing http://localhost:8000 in browser

2. Restart API:
   ```powershell
   # Stop with Ctrl+C
   # Restart
   python -m uvicorn app.main:app --reload
   ```

3. Check firewall:
   - Allow Python through Windows Firewall
   - Check antivirus settings

4. Try different port:
   ```powershell
   python -m uvicorn app.main:app --port 8001
   ```

### Issue: "API returns 500 error"

**Symptoms:**
```
{"error": "Internal server error", "status_code": 500}
```

**Solutions:**
1. Check API logs in terminal
2. Look for Python errors in output
3. Verify models loaded successfully:
   ```
   Visit http://localhost:8000/health
   ```

4. Restart API with fresh environment:
   ```powershell
   # Deactivate and reactivate venv
   deactivate
   venv\Scripts\activate
   python -m uvicorn app.main:app --reload
   ```

### Issue: "API very slow"

**Symptoms:**
- Requests take 30+ seconds
- Timeout errors

**Solutions:**
1. **First request is slow** (model loading) - this is normal!
2. Subsequent requests should be faster
3. Use GPU if available (10x faster)
4. Consider smaller models for development:
   ```python
   # In pipeline.py, use distilled models
   nli_model = "typeform/distilbert-base-uncased-mnli"
   ```

---

## Dashboard Issues

### Issue: "Dashboard won't start"

**Symptoms:**
```
command not found: streamlit
```

**Solutions:**
1. Verify Streamlit installed:
   ```powershell
   pip install streamlit
   ```

2. Make sure venv activated:
   ```powershell
   venv\Scripts\activate
   ```

3. Use full path:
   ```powershell
   python -m streamlit run dashboard/app.py
   ```

### Issue: "Dashboard can't connect to API"

**Symptoms:**
- Red "Cannot connect to API" in sidebar
- Evaluate button does nothing

**Solutions:**
1. Start API first:
   ```powershell
   # Terminal 1
   python -m uvicorn app.main:app --reload
   ```

2. Check API_URL in dashboard/app.py:
   ```python
   API_URL = "http://localhost:8000"  # Correct port?
   ```

3. Test API directly in browser:
   - Visit http://localhost:8000/docs

### Issue: "Dashboard looks broken"

**Symptoms:**
- Layout issues
- Missing elements

**Solutions:**
1. Clear browser cache:
   - Press Ctrl+F5 to hard refresh

2. Try different browser:
   - Chrome, Firefox, Edge

3. Update Streamlit:
   ```powershell
   pip install --upgrade streamlit
   ```

---

## Test Issues

### Issue: "Tests fail with import errors"

**Symptoms:**
```
ModuleNotFoundError: No module named 'evaluators'
```

**Solutions:**
1. Run tests from project root:
   ```powershell
   cd c:\Users\gourig\Documents\LLM_eval
   pytest
   ```

2. Install in development mode:
   ```powershell
   pip install -e .
   ```

### Issue: "Tests timeout"

**Symptoms:**
```
Test exceeded timeout
```

**Solutions:**
1. Increase timeout:
   ```powershell
   pytest --timeout=120
   ```

2. Skip slow tests:
   ```powershell
   pytest -m "not slow"
   ```

3. Run specific fast tests:
   ```powershell
   pytest tests/test_evaluators.py::TestHallucinationDetector::test_clear_contradiction
   ```

---

## Performance Issues

### Issue: "Everything is slow"

**Symptoms:**
- Long evaluation times (>60s)
- High CPU usage
- Computer freezing

**Solutions:**
1. **This is expected on CPU**
   - NLP models are computationally intensive
   - 10-30s per evaluation is normal on CPU

2. Use GPU if available:
   - Install CUDA version of PyTorch
   - Models will auto-detect and use GPU

3. Reduce evaluation frequency:
   - Don't evaluate every keystroke
   - Batch evaluations

4. Use smaller models (dev only):
   ```python
   # Faster but less accurate
   pipeline = LLMEvaluationPipeline(
       nli_model="typeform/distilbert-base-uncased-mnli",
       embedding_model="sentence-transformers/all-MiniLM-L6-v2"  # Already small
   )
   ```

### Issue: "High memory usage"

**Symptoms:**
- Computer slowing down
- Out of memory errors

**Solutions:**
1. Close other applications
2. Don't run multiple evaluations simultaneously
3. Restart Python kernel/API between large batches
4. Load models on-demand:
   ```python
   # Instead of loading all models at startup
   # Load only when needed
   ```

---

## Code Issues

### Issue: "Import errors in code"

**Symptoms:**
```
ImportError: cannot import name 'HallucinationDetector'
```

**Solutions:**
1. Check file structure:
   ```
   evaluators/
   ├── __init__.py  # Must exist!
   ├── hallucination.py
   └── ...
   ```

2. Make sure you're importing correctly:
   ```python
   from evaluators.hallucination import HallucinationDetector
   # Not: from hallucination import HallucinationDetector
   ```

### Issue: "Code changes not taking effect"

**Symptoms:**
- Modified code but behavior unchanged
- Old results still appearing

**Solutions:**
1. If using API with --reload:
   - Should auto-reload
   - Check terminal for reload messages

2. Manually restart:
   - Stop with Ctrl+C
   - Restart

3. Clear Python cache:
   ```powershell
   Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item
   Get-ChildItem -Recurse -Filter "__pycache__" | Remove-Item -Recurse
   ```

4. Reimport in Python shell:
   ```python
   import importlib
   import evaluators.hallucination
   importlib.reload(evaluators.hallucination)
   ```

---

## Data Issues

### Issue: "Validation errors"

**Symptoms:**
```
422 Unprocessable Entity
"Field required"
```

**Solutions:**
1. Check all required fields:
   - question (required)
   - context (required)
   - llm_response (required)

2. Verify field lengths:
   - question: 5-1000 chars
   - context: 10-10000 chars
   - llm_response: 10-5000 chars

3. Check thresholds:
   - Must be between 0.0 and 1.0

### Issue: "Unexpected results"

**Symptoms:**
- Scores seem wrong
- Different results each run

**Solutions:**
1. **Small variations are normal**:
   - Models have slight randomness
   - Scores may vary by ±1-2%

2. Check input quality:
   - Context actually relevant?
   - Response well-formed?

3. Adjust thresholds:
   - Default 0.5 may not suit your use case
   - Experiment with different values

4. Verify weights:
   - Check if custom weights sum to 1.0

---

## Environment Issues

### Issue: "Python version mismatch"

**Symptoms:**
```
SyntaxError: invalid syntax
```

**Solutions:**
1. Check Python version:
   ```powershell
   python --version
   # Need 3.10 or higher
   ```

2. Install Python 3.10+:
   - Download from python.org
   - Add to PATH during installation

3. Use py launcher (Windows):
   ```powershell
   py -3.10 -m venv venv
   ```

### Issue: "Package conflicts"

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions:**
1. Fresh virtual environment:
   ```powershell
   Remove-Item -Recurse -Force venv
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Update all packages:
   ```powershell
   pip install --upgrade -r requirements.txt
   ```

---

## Getting More Help

### Check Logs
Always check the terminal output for error messages.

### API Docs
Visit http://localhost:8000/docs for interactive API testing.

### Model Documentation
- [HuggingFace Models](https://huggingface.co/models)
- [Sentence Transformers](https://www.sbert.net/)

### Community Resources
- [FastAPI Discord](https://discord.gg/fastapi)
- [HuggingFace Forum](https://discuss.huggingface.co/)
- [Stack Overflow](https://stackoverflow.com/)

### Debug Mode
Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Still Stuck?

1. **Read error messages carefully** - they usually tell you exactly what's wrong
2. **Google the error** - someone else probably had the same issue
3. **Check you followed all setup steps** - miss one, and things break
4. **Try restarting everything** - fresh start often helps
5. **Verify all requirements installed** - `pip list` shows what's installed

Remember: Every developer faces these issues. Debugging is part of learning! 🐛🔍
