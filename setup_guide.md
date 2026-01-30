# 📚 Setup Guide for Beginners

This guide will help you set up the project step-by-step, with explanations for each component.

## 🎯 What You'll Learn

- How to set up a Python virtual environment
- How to install ML libraries
- How NLP models work
- How to run a FastAPI server
- How to use Streamlit for visualization

## 📋 Prerequisites

- Python 3.10 or higher installed
- Git installed (you mentioned you know this!)
- A code editor (VS Code recommended)

## 🚀 Step-by-Step Setup

### Step 1: Create a Virtual Environment

**What is a virtual environment?**
A virtual environment is an isolated Python environment that keeps your project dependencies separate from other projects.

```bash
# Navigate to the project folder
cd c:\Users\gourig\Documents\LLM_eval

# Create virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate

# You should see (venv) in your terminal now
```

### Step 2: Install Dependencies

**What are we installing?**
- `fastapi` - Web framework for building APIs (you know the basics!)
- `transformers` - HuggingFace library for NLP models
- `sentence-transformers` - For creating text embeddings
- `sqlalchemy` - Database ORM (Object-Relational Mapping)
- `streamlit` - For building the dashboard UI
- `pytest` - For testing

```bash
# Install all packages
pip install -r requirements.txt

# Download spaCy language model (for text processing)
python -m spacy download en_core_web_sm

# This might take 5-10 minutes depending on your internet
```

### Step 3: Understand the Project Structure

```
LLM_eval/
├── evaluators/          # Core evaluation logic (we'll build this first!)
│   ├── hallucination.py    # Detects if LLM made stuff up
│   ├── groundedness.py     # Checks if claims are backed by evidence
│   ├── faithfulness.py     # Measures semantic similarity
│   ├── relevance.py        # Checks context relevance
│   └── pipeline.py         # Combines all evaluators
│
├── app/                 # FastAPI backend (wraps evaluators in API)
│   ├── main.py             # Main API application
│   ├── models.py           # Data models (request/response formats)
│   └── database.py         # Database connection
│
├── dashboard/           # Streamlit UI (for visualization)
│   └── app.py
│
└── tests/              # Testing (we'll write tests as we go!)
    ├── test_evaluators.py
    └── test_api.py
```

### Step 4: Run Your First Test

Once we build the first component (hallucination detector), you can test it:

```bash
# Run a specific test
pytest tests/test_evaluators.py -v

# Run all tests
pytest
```

### Step 5: Start the API

```bash
# Start FastAPI server
python -m uvicorn app.main:app --reload

# --reload means it auto-restarts when you change code
```

Open your browser: http://localhost:8000/docs
You'll see **automatic API documentation**!

### Step 6: Start the Dashboard

In a **new terminal** (keep the API running):

```bash
# Activate venv again
venv\Scripts\activate

# Start Streamlit
streamlit run dashboard/app.py
```

Open: http://localhost:8501

## 🧠 Key Concepts We'll Learn

### 1. Natural Language Inference (NLI)
**What**: Models that determine if a hypothesis follows from a premise
**Use**: Detect hallucinations by checking if response contradicts context

Example:
- Premise: "Paris is the capital of France"
- Hypothesis: "Paris is the capital of Germany"
- Result: **Contradiction** → Hallucination detected!

### 2. Embeddings
**What**: Numbers that represent text meaning
**Use**: Compare semantic similarity between response and context

Example:
- "The dog is happy" → [0.2, 0.8, 0.1, ...]
- "The puppy is joyful" → [0.21, 0.79, 0.12, ...]
- These are similar → High faithfulness score!

### 3. Cross-Encoders vs Bi-Encoders
**Bi-Encoder**: Encodes text separately, fast but less accurate
**Cross-Encoder**: Encodes texts together, slower but more accurate
**Use**: Cross-encoders for relevance scoring (more precise)

### 4. FastAPI Concepts
**Endpoint**: A URL that accepts requests (e.g., `/api/evaluate`)
**Pydantic Model**: Data validation (ensures correct input format)
**Async**: Non-blocking operations (multiple requests at once)

### 5. SQLAlchemy ORM
**ORM**: Object-Relational Mapping (work with database using Python objects)
**Why**: Instead of writing SQL, you work with Python classes

Example:
```python
# Instead of: SELECT * FROM evaluations WHERE score > 0.8
# You write:
db.query(Evaluation).filter(Evaluation.score > 0.8).all()
```

## 🐛 Troubleshooting

### "Module not found"
```bash
# Make sure venv is activated
venv\Scripts\activate

# Reinstall packages
pip install -r requirements.txt
```

### "CUDA not available" (GPU warning)
This is fine! The models will run on CPU. It's slower but works perfectly for learning.

### Models downloading during first run
**Normal!** The first time you run the code, it downloads models (~1-2 GB). Subsequent runs will be fast.

## 🎯 Next Steps

1. ✅ Setup complete? Great!
2. 📖 Read the concept explanations as we build each component
3. 🧪 Run tests after each module
4. 🎨 Experiment with the dashboard
5. 🚀 Try evaluating different LLM responses

## 💡 Tips for Learning

- **Run the code** after each module we build
- **Modify parameters** and see how results change
- **Ask questions** if something is unclear
- **Break when needed** - we're building incrementally!

Ready to start coding? Let's build the first evaluator! 🚀
