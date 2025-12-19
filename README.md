# 🤖 Professeur IA - AI Tutor with Chain of Thought Reasoning

**An intelligent AI tutoring system that helps students learn from their course materials using Retrieval-Augmented Generation (RAG) and Chain of Thought (CoT) reasoning.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit->=1.28-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Production Ready](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)](#)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow & Process](#workflow--process)
- [Tools Documentation](#tools-documentation)
- [Chain of Thought (CoT)](#chain-of-thought-cot)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [API Keys & Secrets](#api-keys--secrets)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**Professeur IA** is an intelligent tutoring assistant that combines:

1. **Retrieval-Augmented Generation (RAG)**: Searches your course documents for relevant information
2. **Chain of Thought (CoT) Reasoning**: Shows explicit reasoning steps before providing answers
3. **Multi-Tool Agent**: Uses 5 specialized tools for different learning tasks
4. **Semantic Search**: FAISS-based vector indexing for accurate document retrieval
5. **Fast Inference**: Powered by Groq's Llama 3.3 70B model

### What Problems Does It Solve?

- ❌ **Generic AI tutors**: Professeur IA uses YOUR course materials
- ❌ **Black-box AI**: Shows explicit reasoning (CoT) so you understand HOW it thinks
- ❌ **Slow responses**: Uses Groq for ultra-fast inference
- ❌ **Limited functionality**: 5 specialized tools for different learning needs
- ❌ **Poor user experience**: Beautiful, responsive Streamlit UI with custom styling

---

## ✨ Key Features

### 1. **Upload & Index Your Courses**
- Upload multiple PDF files (lectures, notes, textbooks)
- Automatic semantic indexing using FAISS
- Real-time document preview and status tracking

### 2. **Intelligent Search**
- Search course materials using natural language
- Find relevant content instantly using semantic similarity
- See which document each result came from

### 3. **Interactive Quizzes**
- Auto-generate quiz questions from course material
- Multiple-choice format with immediate feedback
- Learns from your answers to provide explanations

### 4. **Study Planning**
- Generate personalized revision schedules
- Multi-day study plans with learning objectives
- Organized by topic and time allocation

### 5. **Code Execution**
- Execute Python code for math problems
- Data analysis and visualization
- Auto-saves plots for easy viewing

### 6. **Wikipedia Fallback**
- Searches Wikipedia for general knowledge
- Supplements course material when needed
- Always tries course materials first

### 7. **Chain of Thought Reasoning**
- Shows 4-step reasoning: Pensée → Action → Observation → Réponse
- Transparent decision-making process
- Learn how the AI thinks, not just what it answers

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PROFESSEUR IA                             │
│                  (Streamlit Web App)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
   ┌────▼────────┐                    ┌────────────▼──────┐
   │   Document  │                    │   Chat Interface  │
   │  Processing │                    │   & User Input    │
   └────┬────────┘                    └────────────┬──────┘
        │                                          │
        ├─ PDF Loading (PyPDFLoader)               │
        ├─ Text Chunking (RecursiveCharacterTextSplitter)
        ├─ Embedding Generation (HuggingFace)      │
        └─ Vector Indexing (FAISS)                 │
                                                   │
                    ┌──────────────────────────────┘
                    │
            ┌───────▼────────┐
            │  LangChain     │
            │  Agent System  │
            └───────┬────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
   ┌────▼────┐ ┌───▼────┐ ┌───▼──────┐
   │   Tool  │ │  Tool  │ │   Tool   │
   │ Factory │ │ Chain  │ │ Executor │
   └────┬────┘ └───┬────┘ └───┬──────┘
        │          │          │
   5 Specialized Tools:
   1. search_course
   2. generate_quiz_context
   3. create_study_plan
   4. search_wikipedia
   5. python_interpreter
                    │
            ┌───────▼────────┐
            │  Groq LLM      │
            │ (Llama 3.3 70B)│
            └────────────────┘
```

### Data Flow

```
User Upload PDF
    ↓
Process & Chunk Documents
    ↓
Generate Embeddings
    ↓
Index in FAISS Vector Store
    ↓
User Asks Question
    ↓
Agent Analyzes Query
    ↓
Select Appropriate Tool(s)
    ↓
Execute Tool(s) → Get Results
    ↓
LLM Generates Answer with CoT
    ↓
Display Response with Reasoning
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- ~2GB free disk space (for embeddings model)
- API key for Groq (free tier available)

### Step 1: Clone the Repository

```bash
git clone https://github.com/CocoCoffre/ProjetGenAI_CourseRAG.git
cd ProjetGenAI_CourseRAG
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Secrets

Create a `.streamlit/secrets.toml` file in your project:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

**Get your Groq API key:**
1. Visit [console.groq.com](https://console.groq.com)
2. Sign up for a free account
3. Create an API key
4. Copy it to `secrets.toml`

### Step 5: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## ⚡ Quick Start

### 1. Upload Your Course Materials

1. Click "Upload course PDFs" in the sidebar
2. Select one or more PDF files (lectures, notes, textbooks)
3. Click "🔄 Process Documents"
4. Wait for indexing to complete (you'll see "✅ Documents ready")

### 2. Ask a Question

```
User: "What is an LSTM?"

Professeur IA will:
1. Search course materials
2. Show reasoning (CoT)
3. Provide answer with citations
```

### 3. Request a Quiz

```
User: "Quiz me on neural networks"

Professeur IA will:
1. Extract relevant content
2. Create a question
3. Wait for your answer
4. Provide feedback
```

### 4. Get a Study Plan

```
User: "Create a 3-day revision schedule"

Professeur IA will:
1. Analyze your materials
2. Generate daily topics
3. Set learning objectives
4. Create a Markdown table
```

---

## 🔄 Workflow & Process

### Complete User Interaction Workflow

```
┌─────────────────────────────────┐
│  User uploads PDFs to sidebar   │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Click "Process Documents"      │
│  - Load PDFs                    │
│  - Split into chunks            │
│  - Generate embeddings          │
│  - Index in FAISS               │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Status: ✅ Documents Ready    │
│  Shows: X vectors indexed       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  User Types Question/Request    │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Agent Receives Query           │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│  CHAIN OF THOUGHT (4 Steps)            │
│                                        │
│  🧠 Pensée (Thought)                  │
│     Analyze: What's being asked?      │
│                                        │
│  → Action                              │
│     Decide: Which tool(s) to use?     │
│                                        │
│  → Observation                         │
│     Show: What tool(s) found?         │
│                                        │
│  ✅ Réponse (Answer)                  │
│     Provide: Final answer with facts  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  Display Response in Chat       │
│  - Formatted nicely             │
│  - Plots if generated           │
│  - Tools used (debug section)   │
└─────────────────────────────────┘
```

### Tool Selection Logic

The agent automatically chooses the right tool:

```
Question about course content?
├─ YES → search_course
│        ├─ Found results?
│        │  ├─ YES → Use them
│        │  └─ NO → search_wikipedia
│        └─ Return formatted answer
└─ NO
   ├─ Quiz/Test request?
   │  ├─ YES → generate_quiz_context
   │  │        Create multiple-choice Q
   │  └─ NO
   │     ├─ Study planning request?
   │     │  ├─ YES → create_study_plan
   │     │  │        Generate table
   │     │  └─ NO
   │     │     ├─ Math/Code problem?
   │     │     │  ├─ YES → python_interpreter
   │     │     │  │        Execute code
   │     │     │  └─ NO → search_wikipedia
   │     │     │          General knowledge
```

---

## 🛠️ Tools Documentation

### Tool 1: `search_course` 🔍

**Purpose**: Search your uploaded course materials

**How it works**:
1. Takes user's natural language question
2. Converts to embedding using HuggingFace model
3. Searches FAISS index for 4 most similar chunks
4. Returns matching content with source citations

**When it's used**:
- User asks about course content
- Agent needs information from materials
- Fallback for "I don't know" questions

**Example**:
```
User: "What is backpropagation?"
→ search_course searches materials
→ Finds 4 chunks mentioning backpropagation
→ Returns formatted results with sources
```

---

### Tool 2: `generate_quiz_context` 📝

**Purpose**: Extract course content for quiz generation

**How it works**:
1. Uses MMR (Maximal Marginal Relevance) retrieval
2. Gets relevant AND diverse content
3. Returns up to 6 chunks for question creation

**When it's used**:
- User requests "Quiz me", "Test me", etc.
- Agent needs to create educational questions
- Want diverse content (not repetitive)

**Example**:
```
User: "Quiz me on machine learning"
→ generate_quiz_context retrieves material
→ Agent creates MCQ from content
→ Shows question, waits for answer
→ Provides feedback
```

---

### Tool 3: `create_study_plan` 📅

**Purpose**: Generate personalized revision schedules

**How it works**:
1. Analyzes available documents
2. Creates daily breakdown over N days
3. Generates Markdown table with:
   - Day number
   - Topics to review
   - Learning objectives

**When it's used**:
- User asks for study plan, revision schedule
- User wants to organize learning
- Need structured approach to materials

**Example**:
```
User: "Make a 5-day study plan"
→ Analyzes course materials
→ Creates 5-day schedule
→ Markdown table output:
   | Day | Topics | Objectives |
```

---

### Tool 4: `search_wikipedia` 🌐

**Purpose**: Fallback general knowledge search

**How it works**:
1. Only used if course materials don't have answer
2. Queries Wikipedia for relevant articles
3. Returns up to 2000 characters of content
4. Helpful for broader context

**When it's used**:
- Course search found nothing
- User asks about concepts not in materials
- Need general knowledge supplement

**Example**:
```
User: "What's artificial intelligence?"
→ search_course finds nothing
→ search_wikipedia fills the gap
→ Returns Wikipedia article excerpt
```

---

### Tool 5: `python_interpreter` 🐍

**Purpose**: Execute Python code for calculations and visualizations

**Capabilities**:
- Run arbitrary Python code
- Auto-imports numpy, pandas if needed
- Captures print() output
- Saves matplotlib plots to `plot.png`
- Error handling and reporting

**When it's used**:
- Math problems
- Data analysis
- Plotting/visualization
- Any code execution request

**Example**:
```
User: "Plot sin(x) from 0 to 2π"
→ Executes Python code
→ Generates plot
→ Saves as plot.png
→ Displays in chat
```

---

## 🧠 Chain of Thought (CoT)

Chain of Thought makes AI reasoning explicit and transparent.

### What is CoT?

Instead of just answering directly, the AI shows its thinking:

```
❌ Without CoT:
Q: What is an LSTM?
A: An LSTM is a type of neural network...
   (User doesn't see how the answer was derived)

✅ With CoT:
Q: What is an LSTM?

🧠 Pensée (Thought):
The user is asking about a specific neural network architecture.
I should search the course materials first.

→ Action:
I will use search_course to find information about LSTMs in the materials.

→ Observation:
Found 3 relevant sections:
- Lecture_05.pdf: Definition of LSTM cells
- Lecture_06.pdf: LSTM architecture diagram
- Notes.pdf: LSTM vs GRU comparison

✅ Réponse (Answer):
An LSTM (Long Short-Term Memory) is... [detailed answer with citations]
```

### Why CoT Matters

✅ **Transparency**: You see exactly how the AI thinks  
✅ **Learning**: Understand the reasoning process  
✅ **Trust**: Know where information comes from  
✅ **Accuracy**: Explicit steps catch errors  
✅ **Debugging**: Easy to spot wrong reasoning  

### CoT Format

All responses follow this 4-step structure:

```
🧠 **Pensée (Thought)**
   ↓ [Analyze what's being asked]
   ↓
→ **Action**
   ↓ [Decide which tool(s) to use and explain why]
   ↓
→ **Observation**
   ↓ [Show what the tool(s) found]
   ↓
✅ **Réponse (Answer)**
   [Final answer with citations and explanations]
```

---

## ⚙️ Configuration

### Environment Variables

Create `.streamlit/secrets.toml`:

```toml
# Required: Groq API Key (free at console.groq.com)
GROQ_API_KEY = "gsk_..."
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[client]
maxMessageSize = 2000

[logger]
level = "info"
```

### Document Processing Parameters

Edit in `app.py` line ~120:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Size of each chunk (characters)
    chunk_overlap=100,   # Overlap between chunks
    separators=["\n\n", "\n", ".", " "]
)
```

**Recommendations**:
- **chunk_size**: 300-1000 chars (500 is optimal)
- **chunk_overlap**: 50-200 chars (100 is good)
- **overlap %**: 20% of chunk_size

---

## 📁 Project Structure

```
ProjetGenAI_CourseRAG/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── CODE_REFACTORING_SUMMARY.md     # Code documentation
├── .streamlit/
│   ├── config.toml                 # Streamlit config
│   └── secrets.toml                # API keys (git-ignored)
├── .gitignore                      # Git ignore rules
└── plot.png                        # Generated plots (temp)
```

### Key Files

**`app.py`** (796 lines)
- Complete Streamlit application
- 3 main sections:
  1. Document Processing
  2. Tool Factory
  3. UI & Chat Interface
- Fully documented with docstrings

**`requirements.txt`**
```
streamlit>=1.28.0
langchain>=0.1.0
langchain-groq>=0.0.1
langchain-community>=0.0.10
langchain-huggingface>=0.0.1
faiss-cpu>=1.7.4
pypdf>=3.17.0
matplotlib>=3.8.0
numpy>=1.24.0
pandas>=2.0.0
python-dotenv>=1.0.0
```

---

## 🔐 API Keys & Secrets

### Getting Your Groq API Key

1. **Visit**: [console.groq.com](https://console.groq.com)
2. **Sign up**: Free account (no credit card needed)
3. **Create API Key**: Go to "Keys" section
4. **Copy**: Your API key
5. **Store**: In `.streamlit/secrets.toml`

### Security Best Practices

✅ **DO**:
- Store secrets in `.streamlit/secrets.toml`
- Add `secrets.toml` to `.gitignore`
- Rotate keys regularly
- Use environment variables in production

❌ **DON'T**:
- Commit secrets to GitHub
- Share your API key
- Use secrets in code
- Store secrets in version control

### Production Deployment

For Streamlit Cloud:

1. Go to your app settings
2. Click "Secrets"
3. Paste your `secrets.toml` content
4. Save

---

## 🧪 Testing

### Test Scenarios

#### Test 1: Course Question
```
Input: "What is the difference between supervised and unsupervised learning?"
Expected: Searches materials, shows CoT, cites sources
```

#### Test 2: Quiz Request
```
Input: "Quiz me on neural networks"
Expected: Creates MCQ, waits for answer, provides feedback
```

#### Test 3: Study Plan
```
Input: "Create a 3-day study plan"
Expected: Generates Markdown table with daily breakdown
```

#### Test 4: Math Problem
```
Input: "Calculate the derivative of x^3 + 2x"
Expected: Executes Python, shows work, displays result
```

#### Test 5: Visualization
```
Input: "Plot y = sin(x) from 0 to 2π"
Expected: Generates and displays plot image
```

### Performance Benchmarks

| Task | Expected Time |
|------|---------------|
| Document indexing (10 pages) | 5-10 seconds |
| Simple question | 2-5 seconds |
| Quiz generation | 3-7 seconds |
| Code execution | 1-3 seconds |
| Study plan | 5-10 seconds |

---

## 🐛 Troubleshooting

### Issue: "No documents loaded yet"
**Solution**:
1. Click "Upload course PDFs"
2. Select PDF files
3. Click "🔄 Process Documents"
4. Wait for "✅ Documents ready" message

### Issue: "API key not configured"
**Solution**:
1. Create `.streamlit/secrets.toml`
2. Add `GROQ_API_KEY = "your_key"`
3. Restart Streamlit (`Ctrl+C`, then `streamlit run app.py`)

### Issue: Slow document processing
**Solution**:
- For large PDFs, try smaller files first
- Check internet (downloading embeddings model)
- Reduce chunk_size if memory issues

### Issue: Irrelevant search results
**Solution**:
1. Check document quality (readable PDFs)
2. Use more specific queries
3. Adjust chunk_size (try 300 or 800)
4. Re-process documents after changes

### Issue: "ModuleNotFoundError"
**Solution**:
```bash
# Make sure virtual environment is active
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes
4. **Test** thoroughly
5. **Commit** with clear messages (`git commit -m 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Areas for Contribution

- 🐛 Bug fixes
- 📈 Performance improvements
- 📚 Documentation enhancements
- 🎨 UI/UX improvements
- 🧪 Additional test cases
- 🌍 Language translations

---

## 📄 License

MIT License - feel free to use this project for personal or commercial purposes.

---

## 👨‍💻 Author

**Course RAG Project Team** - December 2025

Built with ❤️ for students who want to learn smarter.

---

## 📞 Support

- 📖 **Documentation**: See [CODE_REFACTORING_SUMMARY.md](CODE_REFACTORING_SUMMARY.md)
- 🐛 **Issues**: Report on GitHub Issues
- 💬 **Discussions**: Use GitHub Discussions
- 📧 **Email**: Contact via GitHub profile

---

## 🎓 Learning Resources

### About RAG (Retrieval-Augmented Generation)
- [LangChain Documentation](https://python.langchain.com/)
- [FAISS Vector Store Guide](https://github.com/facebookresearch/faiss)
- [Semantic Search Explained](https://www.sbert.net/)

### About Chain of Thought
- [CoT Paper: Wei et al. (2022)](https://arxiv.org/abs/2201.11903)
- [LLM Reasoning Guide](https://platform.openai.com/docs/guides/reasoning)

### Technologies Used
- [Streamlit](https://streamlit.io/) - Web interface
- [LangChain](https://www.langchain.com/) - LLM framework
- [Groq](https://groq.com/) - Fast LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) - Vector search
- [HuggingFace](https://huggingface.co/) - Embeddings

---

## 🌟 Acknowledgments

Thanks to:
- **Groq** for fast LLM inference
- **Meta** for FAISS vector database
- **HuggingFace** for embeddings model
- **Streamlit** for the web framework
- **LangChain** for LLM orchestration

---

## 📊 Project Stats

- **Lines of Code**: 796 (app.py)
- **Docstrings**: 50+
- **Comments**: 100+
- **Tools**: 5 specialized agents
- **Features**: 7 core features
- **Languages**: Python 3.9+
- **UI Framework**: Streamlit
- **LLM**: Groq Llama 3.3 70B

---

**Last Updated**: December 19, 2025  
**Status**: ✅ Production Ready

---

<div align="center">

### Made with ❤️ for better learning

[⬆ Back to Top](#-professeur-ia---ai-tutor-with-chain-of-thought-reasoning)

</div>
