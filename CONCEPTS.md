# 🧠 Core Concepts Explained

This document explains the key concepts used in the LLM Evaluation Framework.

---

## 1. Natural Language Inference (NLI)

### What is it?
NLI is a task where a model determines the relationship between two texts:
- **Premise**: The ground truth (your source)
- **Hypothesis**: The claim being checked (LLM response)

### Three possible outcomes:
- **Entailment**: Hypothesis logically follows from premise ✅
- **Neutral**: Cannot determine from premise ⚠️
- **Contradiction**: Hypothesis contradicts premise ❌

### Example:
```
Premise: "Paris is the capital of France"
Hypothesis: "Paris is France's capital city"
Result: ENTAILMENT (same meaning)

Premise: "Paris is the capital of France"
Hypothesis: "Paris is the capital of Germany"
Result: CONTRADICTION (hallucination!)
```

### How we use it:
- **Hallucination Detection**: Check if response contradicts context
- **Groundedness**: Check if each claim is entailed by context
- **Completeness**: Check if question aspects are addressed

### Model: facebook/bart-large-mnli
- 406M parameters
- Trained on Multi-Genre NLI dataset
- Very accurate for contradiction detection

---

## 2. Embeddings & Semantic Similarity

### What are Embeddings?
Embeddings are numerical representations of text that capture meaning.

Think of it like coordinates on a map:
- Similar meanings → close together in space
- Different meanings → far apart

### Example:
```
"The dog is happy" → [0.2, 0.8, 0.1, 0.5, ...]
"The puppy is joyful" → [0.21, 0.79, 0.12, 0.48, ...]
                          ↑ These vectors are similar!

"The car is fast" → [0.9, 0.1, 0.8, 0.2, ...]
                     ↑ This vector is very different
```

### Cosine Similarity
Measures the angle between two vectors:
- 1.0 = Same direction (very similar)
- 0.5 = 45-degree angle (somewhat similar)
- 0.0 = Perpendicular (unrelated)

### How we use it:
- **Faithfulness Scoring**: Compare response embedding to context embedding
- Higher similarity = more faithful to source

### Model: sentence-transformers/all-MiniLM-L6-v2
- 22M parameters
- 384-dimensional embeddings
- Fast and efficient

---

## 3. Cross-Encoders vs Bi-Encoders

### Bi-Encoders (what we use for embeddings)
```
Text 1 → Encoder → Embedding 1 ┐
                                 ├→ Compare → Similarity
Text 2 → Encoder → Embedding 2 ┘
```

**Pros:**
- Very fast (encode once, compare many times)
- Great for search and retrieval
- Can pre-compute embeddings

**Cons:**
- Less accurate for pairwise comparison
- Texts encoded independently

### Cross-Encoders (what we use for relevance)
```
[Text 1, Text 2] → Encoder → Relevance Score
```

**Pros:**
- More accurate (sees both texts together)
- Better for ranking and relevance scoring
- Captures interactions between texts

**Cons:**
- Slower (must encode both texts together)
- Cannot pre-compute

### When to use which?
- **Bi-Encoder**: Search, retrieval, large-scale comparison
- **Cross-Encoder**: Reranking, relevance scoring, quality assessment

### How we use it:
- **Context Relevance**: Cross-encoder scores how relevant context is to question

### Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking task
- Optimized for relevance scoring

---

## 4. Claim Extraction & Groundedness

### What is Groundedness?
Groundedness measures what percentage of claims in a response are actually supported by the source context.

### The Process:

#### Step 1: Extract Atomic Claims
Break response into individual factual statements:

```
Response: "Python was created by Guido van Rossum in 1991 and is popular for AI."

Atomic Claims:
1. "Python was created by Guido van Rossum"
2. "Python was created in 1991"
3. "Python is popular for AI"
```

#### Step 2: Check Each Claim
Use NLI to verify each claim against context:

```
Context: "Python is a language created by Guido van Rossum in 1991."

Claim 1: "Python was created by Guido van Rossum" → ENTAILED ✅
Claim 2: "Python was created in 1991" → ENTAILED ✅
Claim 3: "Python is popular for AI" → NEUTRAL ⚠️ (not mentioned in context)
```

#### Step 3: Calculate Score
```
Groundedness = Supported Claims / Total Claims
             = 2 / 3
             = 67%
```

### Why is this important?
Even if a response doesn't contradict the context, it might add unsupported information. Groundedness catches this!

---

## 5. Question Decomposition & Completeness

### What is Completeness?
Completeness measures whether a response addresses ALL aspects of a question.

### Complex Questions Have Multiple Aspects:

#### Example 1:
```
Question: "What is Python and when was it created?"

Aspects:
1. What is Python? (definition)
2. When was it created? (time)

Complete Answer: Must address BOTH aspects
Incomplete Answer: Answers only one aspect
```

#### Example 2:
```
Question: "Who invented the telephone, when, and how does it work?"

Aspects:
1. Who invented it?
2. When was it invented?
3. How does it work?

Complete Answer: Must address ALL THREE
```

### How We Detect Aspects:

1. **Question Words**: what, when, where, who, why, how
2. **Conjunctions**: Split on "and", "or", "also"
3. **NLI Verification**: Check if each aspect is addressed in response

### Completeness Score:
```
Completeness = Addressed Aspects / Total Aspects
```

---

## 6. Overall Quality Score

### Weighted Average
We combine all metrics using configurable weights:

```python
Overall Score = (
    hallucination_score × 0.25 +  # 25% weight
    groundedness_score × 0.20 +    # 20% weight
    faithfulness_score × 0.20 +    # 20% weight
    relevance_score × 0.15 +       # 15% weight
    completeness_score × 0.20      # 20% weight
) × 100
```

### Quality Tiers:
- **90-100**: Excellent 🌟
- **75-89**: Good ✅
- **60-74**: Acceptable 👍
- **40-59**: Poor ⚠️
- **0-39**: Very Poor ❌

### Why These Weights?
- **Hallucination** (25%): Most critical - contradictions are serious
- **Groundedness** (20%): Very important - claims should be backed
- **Faithfulness** (20%): Very important - stay close to source
- **Relevance** (15%): Important - but LLM can't control context quality
- **Completeness** (20%): Very important - answer everything

**You can adjust these based on your use case!**

---

## 7. FastAPI Concepts (for beginners)

### What is FastAPI?
A modern Python framework for building APIs (web services).

### Key Concepts:

#### Endpoints
URLs that accept requests:
```python
@app.get("/")  # GET request to root
async def root():
    return {"message": "Hello"}

@app.post("/api/evaluate")  # POST request with data
async def evaluate(request: EvaluationRequest):
    # Process evaluation
    return result
```

#### Pydantic Models
Data validation classes:
```python
class EvaluationRequest(BaseModel):
    question: str  # Must be a string
    context: str
    llm_response: str
```

If you send wrong data types, FastAPI automatically rejects it!

#### Automatic Documentation
FastAPI generates interactive API docs at `/docs`. You can:
- See all endpoints
- Try them directly in the browser
- View request/response formats

#### Async/Await
```python
async def my_function():
    # Can handle multiple requests at once
    result = await some_operation()
    return result
```

Benefits:
- Non-blocking operations
- Better performance
- Can handle many concurrent requests

---

## 8. Streamlit Concepts (for beginners)

### What is Streamlit?
A Python library for creating web apps without HTML/CSS/JavaScript.

### Key Components:

#### Input Widgets
```python
# Text input
question = st.text_input("Question")

# Text area (multi-line)
context = st.text_area("Context", height=200)

# Slider
threshold = st.slider("Threshold", 0.0, 1.0, 0.5)

# Button
if st.button("Evaluate"):
    # Do something
```

#### Display Components
```python
# Title
st.title("My App")

# Metric
st.metric("Score", "85/100")

# Progress bar
st.progress(0.85)

# JSON viewer
st.json({"key": "value"})
```

#### Layout
```python
# Columns
col1, col2 = st.columns(2)
with col1:
    st.write("Left side")
with col2:
    st.write("Right side")

# Sidebar
with st.sidebar:
    st.header("Settings")
```

### Reactive Updates
Streamlit automatically reruns your script when:
- User interacts with a widget
- You click a button
- Data changes

---

## 9. Testing with Pytest

### What is Pytest?
A Python testing framework for writing and running tests.

### Test Structure:
```python
def test_something():
    # Arrange: Set up test data
    detector = HallucinationDetector()
    context = "Paris is the capital of France"
    response = "Paris is France's capital"
    
    # Act: Run the code
    result = detector.detect(context, response)
    
    # Assert: Check the result
    assert result['has_hallucination'] is False
```

### Running Tests:
```bash
pytest                    # Run all tests
pytest -v                 # Verbose output
pytest test_file.py       # Run specific file
pytest -k "hallucination" # Run tests matching name
```

### Why Test?
- Catch bugs early
- Ensure code works as expected
- Safe refactoring (change code without breaking it)
- Document expected behavior

---

## 10. Virtual Environments

### What is a Virtual Environment?
An isolated Python environment for your project.

### Why Use One?
- **Isolation**: Each project has its own packages
- **No conflicts**: ProjectA uses package v1, ProjectB uses v2
- **Reproducibility**: Same environment on any machine
- **Clean**: Doesn't pollute global Python installation

### How It Works:
```
Global Python: Python 3.10, pip, basic packages
    │
    ├── Project1 venv: fastapi 0.109, transformers 4.36
    │
    └── Project2 venv: django 4.2, pandas 2.0
```

Each project is independent!

### Commands:
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Deactivate
deactivate

# Install packages (only in this venv)
pip install package_name
```

---

## Summary: How Everything Fits Together

```
User Input (Question + Context + LLM Response)
    ↓
FastAPI Endpoint (/api/evaluate)
    ↓
Evaluation Pipeline
    ├→ Hallucination Detector (NLI)
    ├→ Groundedness Scorer (NLI + Claim Extraction)
    ├→ Faithfulness Scorer (Embeddings + Cosine Similarity)
    ├→ Relevance Checker (Cross-Encoder)
    └→ Completeness Checker (Question Decomposition + NLI)
    ↓
Weighted Score Calculation
    ↓
Results (JSON Response)
    ↓
Streamlit Dashboard (Visual Display)
```

Each component uses specialized NLP models and techniques to evaluate different aspects of quality!

---

## Next Steps for Learning

1. **Experiment**: Change thresholds and see how results change
2. **Read Code**: Each file has detailed comments
3. **Try Different Models**: Swap out models to see differences
4. **Add Features**: Extend the system with your own ideas
5. **Dive Deeper**: Learn more about transformers, NLP, and ML

Happy learning! 🚀
