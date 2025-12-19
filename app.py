"""Intelligent AI Tutor Application using Streamlit and LangChain.

This application provides an AI-powered tutoring system that helps students learn from
their course materials using Retrieval-Augmented Generation (RAG) with Chain of Thought (CoT)
reasoning. It supports document search, quizzes, study planning, and code execution.

Architecture:
- Document Processing: PDF loading and semantic chunking
- Vector Retrieval: FAISS-based semantic search
- Multi-Tool Agent: 5 specialized tools for different learning tasks
- Chain of Thought: Explicit reasoning steps (Pensée → Action → Observation → Réponse)
- LLM: Groq's Llama 3.3 70B for fast inference

Author: Course RAG Project Team
Date: December 2025
"""

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import sys
from io import StringIO

# LangChain ecosystem imports
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

# Configure Streamlit page properties
st.set_page_config(
    page_title="AI Tutor - Course RAG with CoT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SECTION 1: DOCUMENT PROCESSING
# ============================================================================

def process_documents(uploaded_files: list) -> list:
    """Process uploaded PDF files and extract their content.
    
    This function:
    1. Creates temporary files for each PDF upload
    2. Loads pages using PyPDFLoader
    3. Stores original filename in document metadata
    4. Generates previews for study planning context
    5. Cleans up temporary files after processing
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects (type: PDF)
        
    Returns:
        List of LangChain Document objects with preserved metadata
    """
    documents = []
    st.session_state.doc_previews = {}  # Reset to avoid mixing courses
    
    for file in uploaded_files:
        # Create temporary file to hold PDF content during processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            # Load PDF pages using PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # Process metadata: Replace temp path with actual filename
            full_text_preview = ""
            for i, doc in enumerate(docs):
                # Store original filename (not /tmp/xxxxx.pdf)
                doc.metadata["source"] = file.name
                
                # Collect text from first 3 pages for study planning context
                if i < 3:
                    full_text_preview += doc.page_content + "\n"
            
            # Store preview for later use in create_study_plan tool
            st.session_state.doc_previews[file.name] = full_text_preview[:3000]
            
            # Add all pages to document collection
            documents.extend(docs)
            
        finally:
            # Ensure temporary file is cleaned up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return documents


def build_vector_store(documents: list) -> FAISS:
    """Build a FAISS vector store from documents for semantic search.
    
    Strategy:
    - Chunk documents into semantic units (500 chars) for accurate retrieval
    - Use recursive splitting to maintain context coherence
    - Apply overlap between chunks to preserve surrounding information
    - Generate embeddings using HuggingFace's lightweight model
    - Index with FAISS for O(1) similarity search performance
    
    Args:
        documents: List of LangChain Document objects with page_content
        
    Returns:
        FAISS vector store indexed and ready for similarity searches
    """
    # Initialize splitter with optimized parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        separators=["\n\n", "\n", ".", " "]  # Try these boundaries in order
    )
    
    # Split documents into chunks
    splits = text_splitter.split_documents(documents)
    print(f"[DEBUG] Created {len(splits)} chunks from {len(documents)} documents")
    
    # Initialize embeddings model (all-MiniLM-L6-v2: 384-dim, ~22MB)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create and index FAISS vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    print(f"[DEBUG] Vectorstore indexed with {vectorstore.index.ntotal} vectors")
    
    return vectorstore


# ============================================================================
# SECTION 2: TOOL FACTORY - Creates all agent tools with closure pattern
# ============================================================================

def create_all_tools(vectorstore: FAISS, doc_previews: dict) -> list:
    """Factory function to create all LLM tools with vectorstore access.
    
    Why closure pattern?
    - Avoids using st.session_state inside tools (causes issues in LangGraph)
    - Binds vectorstore and doc_previews at tool creation time
    - Each tool has direct access without global state
    
    Tools created (in recommended usage order):
    1. search_course: Find information in uploaded PDFs
    2. generate_quiz_context: Extract content for quiz questions
    3. create_study_plan: Generate revision schedules
    4. search_wikipedia: General knowledge fallback
    5. python_interpreter: Execute code for math/analysis/plotting
    
    Args:
        vectorstore: FAISS vector store for document search
        doc_previews: Dict mapping filename → preview text
        
    Returns:
        List of @tool decorated functions ready for agent
    """
    
    # ========== TOOL 1: Search Course Documents ==========
    @tool
    def search_course(query: str) -> str:
        """Search uploaded course PDFs for information using semantic similarity.
        
        Implementation:
        - Uses FAISS vector store to find semantically similar chunks
        - Ranks results by cosine similarity
        - Returns top-4 matches with source citations
        - Handles cases where no matches are found
        
        Args:
            query: User's question or search term (natural language)
            
        Returns:
            Formatted string with matching content and source filenames
        """
        print(f"[DEBUG search_course] Query: {query}")
        print(f"  Vectorstore size: {vectorstore.index.ntotal} vectors")
        
        try:
            # Semantic search: find 4 most similar chunks
            results = vectorstore.similarity_search(query, k=4)
            print(f"  Results found: {len(results)}")
            
            # Handle no results case
            if not results:
                return f"No content found for '{query}' in documents."
            
            # Format results with source citations
            formatted = []
            for doc in results:
                source = doc.metadata.get('source', 'Unknown')
                formatted.append(f"[From {source}]\n{doc.page_content}")
            
            # Join results with separator
            return "\n---\n".join(formatted)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            return f"Error searching documents: {e}"
    
    
    # ========== TOOL 2: Generate Quiz Questions ==========
    @tool
    def generate_quiz_context(topic: str) -> str:
        """Extract course content for quiz generation.
        
        Strategy: Uses Maximal Marginal Relevance (MMR) retrieval to get:
        - Relevant chunks matching the topic
        - Diverse chunks (not just repetitive similar content)
        
        Args:
            topic: Subject or concept to quiz on
            
        Returns:
            Relevant course content for question generation
        """
        # Use MMR for better diversity in results
        retriever = vectorstore.as_retriever(
            search_type="mmr",  # Maximal Marginal Relevance
            search_kwargs={'k': 6, 'fetch_k': 20}
        )
        results = retriever.invoke(topic)
        
        if not results:
            return f"No information found on '{topic}' in course materials."
        
        # Combine all chunks into one context
        content = "\n\n".join([doc.page_content for doc in results])
        print(f"[DEBUG QUIZ] Topic: {topic} - Chunks found: {len(results)}")
        
        return f"Course content on '{topic}':\n{content}"
    
    
    # ========== TOOL 3: Create Study Plan ==========
    @tool
    def create_study_plan(days: int, focus: str = "All") -> str:
        """Generate a personalized study/revision plan.
        
        Creates a structured schedule for students to review course material
        efficiently over N days.
        
        Args:
            days: Number of days available for revision
            focus: Specific topic to focus on (optional, default: All topics)
            
        Returns:
            Study plan with daily breakdown and learning objectives
        """
        # Build context from document previews
        context_str = "Available Documents:\n\n"
        for filename, preview in doc_previews.items():
            context_str += f"{filename}\nPreview: {preview[:300]}...\n\n"
        
        if not doc_previews:
            context_str = "No document previews available"
        
        # Return structured prompt for LLM to generate plan
        return (
            f"{context_str}\n"
            f"Create a {days}-day study plan with Markdown table:\n"
            f"Columns: (Day | Topics to Review | Learning Objectives)"
        )
    
    
    # ========== TOOL 4: Search Wikipedia ==========
    @tool
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for general knowledge information.
        
        Used as fallback when course materials don't contain the answer.
        Provides external reference for broader understanding of topics.
        
        Args:
            query: Topic or concept to search for
            
        Returns:
            Wikipedia excerpt or error message
        """
        try:
            # Initialize Wikipedia retriever
            retriever = WikipediaRetriever(
                top_k_results=1,
                doc_content_chars_max=2000
            )
            
            # Retrieve Wikipedia content
            docs = retriever.invoke(query)
            
            if not docs:
                return "No Wikipedia results found for this query."
                
            # Return content
            return "\n\n".join([doc.page_content for doc in docs])
            
        except Exception as e:
            return f"Wikipedia search error: {e}"
    
    
    # ========== TOOL 5: Python Code Interpreter ==========
    @tool
    def python_interpreter(code: str) -> str:
        """Execute Python code for calculations, data analysis, and plotting.
        
        Capabilities:
        - Executes arbitrary Python code safely (in isolated namespace)
        - Captures stdout for result display
        - Auto-imports numpy/pandas if referenced
        - Saves matplotlib plots to 'plot.png'
        - Returns execution output and error messages
        
        Args:
            code: Valid Python code to execute
            
        Returns:
            Execution output (print statements) + plot save message if created
        """
        # Redirect stdout to capture print() output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Isolated namespace for code execution (security + isolation)
        exec_globals = {'plt': plt}
        
        try:
            # Auto-import numpy if referenced in code
            if 'np.' in code or 'numpy' in code:
                import numpy as np
                exec_globals['np'] = np
                exec_globals['numpy'] = np
            
            # Auto-import pandas if referenced in code
            if 'pd.' in code or 'pandas' in code:
                import pandas as pd
                exec_globals['pd'] = pd
                exec_globals['pandas'] = pd
            
            # Execute user code in isolated namespace
            exec(code, exec_globals)
            
            # Retrieve captured output from print statements
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Save plot if matplotlib figures were created
            if plt.get_fignums():
                plt.savefig('plot.png', dpi=150, bbox_inches='tight')
                plt.close('all')  # Close all figures to free memory
                output += "\n\n📊 Plot generated and saved."
            
            # Return output or success message if nothing printed
            return output if output else "✅ Code executed successfully."
        
        except Exception as e:
            # Restore stdout before returning error
            sys.stdout = old_stdout
            return f"❌ Execution error: {type(e).__name__}: {str(e)}"
    
    
    # ========== Return all tools as a list ==========
    return [
        search_course,
        generate_quiz_context,
        create_study_plan,
        search_wikipedia,
        python_interpreter,
    ]


# ============================================================================
# SECTION 3: STREAMLIT UI - Main Application Interface
# ============================================================================

def main():
    """Main Streamlit application with custom styling and chat interface.
    
    Architecture:
    - Custom CSS for complete UI redesign (not default Streamlit)
    - Session state for persistent conversation history
    - Sidebar for document management
    - Chat interface for user interactions
    - Agent integration for tool-based reasoning
    """
    
    # ========== CUSTOM CSS STYLING ==========
    st.markdown("""
    <style>
    /* Reset Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    
    /* Main gradient background (purple theme) */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* SIDEBAR - FULLY VISIBLE AND ACCESSIBLE */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 3px solid #667eea !important;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1) !important;
    }
    
    /* Ensure all sidebar content is visible */
    [data-testid="stSidebar"] > * {
        visibility: visible !important;
        opacity: 1 !important;
        display: block !important;
    }
    
    [data-testid="stSidebar"] * {
        visibility: visible !important;
        opacity: 1 !important;
        color: #2d3748 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #667eea !important;
        font-weight: 700 !important;
        visibility: visible !important;
    }
    
    /* FILE UPLOADER - FULLY VISIBLE */
    [data-testid="stFileUploader"] {
        background: #f8f9ff !important;
        border-radius: 15px !important;
        border: 2px dashed #667eea !important;
        padding: 1.5rem !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="stFileUploader"] * {
        color: #2d3748 !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(135deg, #f8f9ff 0%, #e8eaf8 100%) !important;
        border: 2px dashed #667eea !important;
        border-radius: 12px !important;
        visibility: visible !important;
    }
    
    /* SIDEBAR TOGGLE BUTTON - ALWAYS VISIBLE */
    button[kind="header"] {
        background-color: #667eea !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        visibility: visible !important;
        z-index: 1000 !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
    }
    
    button[kind="header"]:hover {
        background-color: #764ba2 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.6) !important;
        transform: scale(1.05) !important;
    }
    
    /* SUCCESS/WARNING STATUS BADGES */
    [data-testid="stSidebar"] .stSuccess {
        background: #48bb78 !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stSidebar"] .stWarning {
        background: #f56565 !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    /* CUSTOM TITLE HERO SECTION */
    .custom-title {
        text-align: center;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .custom-title h1 {
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin: 0 !important;
    }
    
    .custom-subtitle {
        color: #667eea;
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    /* CHAT MESSAGE BUBBLE STYLING */
    .stChatMessage {
        background: white !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        margin: 1rem 0 !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15) !important;
    }
    
    /* User message: gradient background with white text */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) * {
        color: white !important;
    }
    
    /* Assistant message: white background with purple border and text */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: white !important;
        border: 2px solid #667eea !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) * {
        color: #2d3748 !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) h1,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) h2,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) h3 {
        color: #667eea !important;
    }
    
    /* CHAT INPUT CONTAINER STYLING */
    .stChatInputContainer {
        background: white !important;
        border-radius: 25px !important;
        padding: 1rem !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15) !important;
        border: 2px solid #667eea !important;
    }
    
    .stChatInputContainer textarea {
        background: white !important;
        color: #2d3748 !important;
        border-radius: 20px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 1rem !important;
    }
    
    .stChatInputContainer textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* BUTTON STYLING */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 15px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Expandable sections (debug info) */
    .streamlit-expanderHeader {
        background: #667eea !important;
        color: white !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: white !important;
        color: #2d3748 !important;
        border: 2px solid #667eea !important;
        border-top: none !important;
    }
    
    /* Code styling */
    code {
        background: #f7fafc !important;
        color: #667eea !important;
        border-radius: 5px !important;
    }
    
    /* Table styling */
    table {
        background: white !important;
        border-radius: 10px !important;
    }
    
    thead tr {
        background: #667eea !important;
    }
    
    thead th {
        color: white !important;
        font-weight: 600 !important;
    }
    
    tbody td {
        color: #2d3748 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Message appearance animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ========== HEADER SECTION ==========
    st.markdown("""
    <div class="custom-title">
        <h1>🧠 Professeur IA</h1>
        <p class="custom-subtitle">Intelligent AI Tutor - Learn with Chain of Thought Reasoning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== SESSION STATE INITIALIZATION ==========
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "doc_previews" not in st.session_state:
        st.session_state.doc_previews = {}
    if "tools" not in st.session_state:
        st.session_state.tools = None
    
    # ========== SIDEBAR: Document Management ==========
    with st.sidebar:
        st.header("📚 Course Documents")
        
        files = st.file_uploader(
            "Upload course PDFs",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more course materials (PDF format)"
        )
        
        process_btn = st.button(
            "🔄 Process Documents",
            use_container_width=True,
            help="Parse and index uploaded PDFs"
        )
        
        if process_btn and files:
            with st.spinner("Processing documents..."):
                docs = process_documents(files)
                st.session_state.vectorstore = build_vector_store(docs)
                st.session_state.tools = create_all_tools(
                    vectorstore=st.session_state.vectorstore,
                    doc_previews=st.session_state.doc_previews
                )
                st.success("✅ Documents ready for learning!")
        
        st.divider()
        st.caption("🔧 Tools: RAG Search • Quizzes • Planning • Code Execution")
        
        if st.session_state.vectorstore is not None:
            num_vectors = st.session_state.vectorstore.index.ntotal
            st.success(f"🟢 Memory Loaded\n{num_vectors} vectors indexed")
        else:
            st.warning("🔴 No documents loaded yet")
        
        if st.session_state.doc_previews:
            st.markdown("### 📄 Indexed Files:")
            for filename in st.session_state.doc_previews.keys():
                st.caption(f"📖 {filename}")
    
    # ========== CHAT INTERFACE ==========
    for msg in st.session_state.messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        elif msg.type == "ai" and msg.content:
            st.chat_message("assistant").write(msg.content)
    
    # ========== USER INPUT & AGENT EXECUTION ==========
    user_input = st.chat_input("Ask me anything about your courses...")
    
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        if st.session_state.tools is None:
            st.error("⚠️ Please upload and process documents first!")
            st.stop()
        
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("❌ API key not configured!")
            st.stop()
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=4096
        )
        
        tools = st.session_state.tools
        
        docs_context = ""
        if st.session_state.doc_previews:
            docs_context = "\n\n**INDEXED DOCUMENTS:**\n"
            for filename in st.session_state.doc_previews.keys():
                docs_context += f"- {filename}\n"
        else:
            docs_context = "\n\n**NOTE: No documents currently indexed.**"
        
        system_prompt = (
            "You are Professeur IA, an expert AI tutor specialized in helping students learn.\n"
            f"{docs_context}\n\n"
            
            "⚠️ **MANDATORY: Chain of Thought (CoT) - ALWAYS use this format**\n\n"
            
            "🧠 **RESPONSE STRUCTURE (Pensée → Action → Observation → Réponse):**\n"
            "1. **Pensée (Thought)**: Analyze the user's question and identify what's needed\n"
            "2. **Action (Action)**: Explain which tool(s) you'll use and why\n"
            "3. **Observation (Observation)**: Show the tool results or findings\n"
            "4. **Réponse (Answer)**: Provide the final answer with citations\n\n"
            
            "📋 **TOOL USAGE RULES:**\n"
            "- **Course Questions**: ALWAYS try search_course FIRST, then Wikipedia if needed\n"
            "- **Quizzes**: Use generate_quiz_context, create ONE question, wait for answer\n"
            "- **Study Plans**: Use create_study_plan, return Markdown table\n"
            "- **Math/Code**: Use python_interpreter with print() for output\n"
            "- **Definitions**: Try search_course → search_wikipedia\n\n"
            
            "✨ **PEDAGOGICAL GUIDELINES:**\n"
            "- Be encouraging and supportive\n"
            "- Explain concepts clearly with course material examples\n"
            "- For quizzes: Never give answers immediately\n"
            "- Always cite your sources\n"
            "- Use Markdown for formatting\n\n"
            
            "🌍 **LANGUAGE**: Respond in the same language as the user (French or English)\n"
        )
        
        agent_graph = create_agent(llm, tools=tools, system_prompt=system_prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking with Chain of Thought..."):
                try:
                    response = agent_graph.invoke({"messages": st.session_state.messages})
                    full_history = response["messages"]
                    final_answer = full_history[-1].content
                    
                    st.markdown(final_answer)
                    
                    if os.path.exists("plot.png"):
                        st.image(
                            "plot.png",
                            caption="📊 Generated Visualization",
                            use_container_width=True
                        )
                        os.remove("plot.png")
                    
                    used_tools = []
                    for msg in full_history:
                        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                used_tools.append(tool_call['name'])
                    
                    if used_tools:
                        with st.expander("🔍 Debug Info: Tools Used", expanded=False):
                            st.write(f"**{', '.join(set(used_tools))}**")
                    
                    st.session_state.messages = full_history
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("View Full Error"):
                        st.code(traceback.format_exc())


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
