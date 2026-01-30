# 🧠 LLM Response Evaluation Framework

A production-ready system for evaluating LLM responses across 5 key metrics: hallucination detection, groundedness, faithfulness, context relevance, and answer completeness.

## 📊 Evaluation Metrics

1. **Hallucination Detection** - Detects contradictions and fabricated information using NLI models
2. **Groundedness Score** - Measures % of claims supported by source context
3. **Faithfulness Score** - Calculates semantic similarity between response and context
4. **Context Relevance** - Scores how relevant the provided context is to the question
5. **Answer Completeness** - Checks if the answer addresses all aspects of the question

## 🛠️ Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Database**: SQLite (can upgrade to SQL Server)
- **ML/NLP**: HuggingFace Transformers, SentenceTransformers
- **Testing**: pytest
- **CI/CD**: GitHub Actions

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Set Up Environment

Create a `.env` file:
```
DATABASE_URL=sqlite:///./llm_eval.db
MODEL_CACHE_DIR=./models
LOG_LEVEL=INFO
```

### 3. Run the API

```bash
python -m uvicorn app.main:app --reload
```

API will be available at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 4. Run the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard will open at: http://localhost:8501

## 📁 Project Structure

```
LLM_eval/
├── app/
│   ├── main.py              # FastAPI app
│   ├── models.py            # Pydantic models
│   ├── database.py          # Database setup
│   └── routers/
│       └── evaluation.py    # Evaluation endpoints
├── evaluators/
│   ├── __init__.py
│   ├── hallucination.py     # Hallucination detector
│   ├── groundedness.py      # Groundedness scorer
│   ├── faithfulness.py      # Faithfulness scorer
│   ├── relevance.py         # Context relevance checker
│   ├── completeness.py      # Answer completeness checker
│   └── pipeline.py          # Main evaluation pipeline
├── dashboard/
│   └── app.py               # Streamlit dashboard
├── tests/
│   ├── test_evaluators.py
│   └── test_api.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=evaluators --cov=app
```

## 📚 How It Works

### Hallucination Detection (NLI)
Uses Natural Language Inference models to detect contradictions:
- **Premise**: Source context
- **Hypothesis**: LLM response claim
- **Output**: Entailment/Neutral/Contradiction

### Groundedness Scoring
1. Extracts atomic claims from the response
2. Checks each claim against source context
3. Calculates % of grounded claims

### Faithfulness Scoring
1. Generates embeddings for response and context
2. Calculates cosine similarity
3. Higher score = more faithful to source

### Context Relevance
Uses cross-encoder models to score how relevant the context is to answering the question.

### Answer Completeness
1. Identifies key aspects in the question
2. Checks if each aspect is addressed in the answer
3. Calculates completeness percentage

## 🔄 API Usage

```python
import requests

response = requests.post("http://localhost:8000/api/evaluate", json={
    "question": "What is the capital of France?",
    "context": "France is a country in Europe. Paris is its capital city.",
    "llm_response": "The capital of France is Paris."
})

print(response.json())
```

## 📈 Roadmap

- [x] Phase 1: Core evaluation engine
- [x] Phase 2: FastAPI integration
- [x] Phase 3: SQLite database
- [x] Phase 4: Streamlit dashboard
- [ ] Phase 5: Advanced caching
- [ ] Phase 6: SQL Server migration
- [ ] Phase 7: Docker containerization

## 🤝 Contributing

This is a learning project! Feel free to experiment and improve.

## 📝 License

MIT License
