import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt

# --- IMPORTS STRICTS (Basés sur ta doc LangGraph) ---
from langchain.tools import tool
from langchain_groq import ChatGroq
# C'est LA fonction dont parle ta doc (qui construit un graph):
from langchain.agents import create_agent 

from langchain_community.retrievers import WikipediaRetriever
from langchain_experimental.tools import PythonREPLTool

# --- Imports standards RAG ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_community.vectorstores import FAISS

# --- Config ---
st.set_page_config(page_title="Agent Étudiant (LangGraph)", page_icon="🤖")

# --- 1. BACKEND (PDF) ---

def process_documents(uploaded_files):
    """Lecture des PDF avec extraction des noms de fichiers réels."""
    documents = []
    
    # On vide les previews précédents pour ne pas mélanger les cours
    st.session_state.doc_previews = {}
    
    for file in uploaded_files:
        # Création fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
            
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            
            # --- CORRECTION CRITIQUE : GESTION DES METADONNÉES ---
            # PyPDFLoader met le chemin temporaire dans 'source'. On remet le vrai nom du fichier.
            # On en profite pour capturer le début du texte pour le planning.
            full_text_preview = ""
            for i, doc in enumerate(docs):
                doc.metadata["source"] = file.name # Remplace '/tmp/x.pdf' par 'Lecture 01.pdf'
                if i < 3: # On prend les 3 premières pages pour l'aperçu structurel
                    full_text_preview += doc.page_content + "\n"
            
            # On stocke cet aperçu dans la session
            st.session_state.doc_previews[file.name] = full_text_preview[:3000] # Limite à 2000 car
            
            documents.extend(docs)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    return documents

def build_vector_store(documents):
    """Indexation with optimized chunking."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # Smaller chunks for better retrieval
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    splits = text_splitter.split_documents(documents)
    print(f"DEBUG: Created {len(splits)} chunks from {len(documents)} documents")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    print(f"DEBUG: Vectorstore indexed with {vectorstore.index.ntotal} vectors")
    
    return vectorstore

def check_vectorstore_quality(vectorstore, test_queries):
    """Check if vectorstore is working by running test queries."""
    print("\n[VECTORSTORE QUALITY CHECK]")
    print(f"Total vectors indexed: {vectorstore.index.ntotal}")
    
    for query in test_queries:
        results = vectorstore.similarity_search(query, k=2)
        print(f"\nQuery: '{query}'")
        print(f"  Matches: {len(results)}")
        if results:
            print(f"  Top match: {results[0].page_content[:100]}...")
            
# --- 2. DÉFINITION DES SCHÉMAS  ---


# --- 3. DÉFINITION DE L'OUTIL AVEC @tool (TA DEMANDE) ---

def create_all_tools(vectorstore, doc_previews: dict):
    """
    Factory function that creates all tools with access to vectorstore and previews.
    Returns a list of tool objects.
    """
    
    # Tool 1: search_course
    @tool
    def search_course(query: str) -> str:
        """Search uploaded PDF course documents for information about the course content."""
        print(f"\n[DEBUG search_course] Query: {query}")
        print(f"  Vectorstore size: {vectorstore.index.ntotal}")
        
        try:
            results = vectorstore.similarity_search(query, k=4)
            print(f"  Results found: {len(results)}")
            
            if not results:
                return f"No content found for '{query}' in documents."
            
            formatted = []
            for doc in results:
                source = doc.metadata.get('source', 'Unknown')
                formatted.append(f"[From {source}]\n{doc.page_content}")
            
            return "\n---\n".join(formatted)
        except Exception as e:
            print(f"  ERROR: {e}")
            return f"Error searching: {e}"
    
    # Tool 2: generate_quiz_context
    @tool
    def generate_quiz_context(topic: str) -> str:
        """Extract course content to prepare a quiz. Use ONLY when user asks for quiz/test/exercise."""
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 6, 'fetch_k': 20}
        )
        results = retriever.invoke(topic)
        
        if not results:
            return f"No information found on '{topic}'."
        
        content = "\n\n".join([doc.page_content for doc in results])
        print(f"DEBUG QUIZ - Topic: {topic} - Chunks found: {len(results)}")
        return f"Course content on '{topic}':\n{content}"
    
    # Tool 3: create_study_plan
    @tool
    def create_study_plan(days: int, focus: str = "All") -> str:
        """Create revision schedule based on uploaded documents. Use when user asks for planning/schedule/revision."""
        context_str = "Documents available:\n\n"
        for filename, preview in doc_previews.items():
            context_str += f"{filename}\n{preview[:300]}...\n\n"
        
        if not doc_previews:
            context_str = "No preview available - use documents from system prompt"
        
        return (
            f"{context_str}\n"
            f"Create a {days}-day study plan with table: (Jour | Sujets | Objectifs)"
        )
    
    # Tool 4: search_wikipedia (doesn't need vectorstore)
    @tool
    def search_wikipedia(query: str) -> str:
        """Search Wikipedia for general knowledge. Use ONLY if search_course finds nothing."""
        try:
            from langchain_community.retrievers import WikipediaRetriever
            retriever = WikipediaRetriever(top_k_results=1, doc_content_chars_max=2000)
            docs = retriever.invoke(query)
            if not docs:
                return "No Wikipedia results found."
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Wikipedia search error: {e}"
    @tool
    def python_interpreter(code: str) -> str:
        """
        Execute Python code for calculations, data analysis, or plotting.
        Supports matplotlib plots - save as 'plot.png'.
        
        Args:
            code: Python code to execute
        """
        import sys
        from io import StringIO
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        exec_globals = {'plt': plt}
        
        try:
            # Auto-import numpy/pandas si nécessaire
            if 'np.' in code or 'numpy' in code:
                import numpy as np
                exec_globals['np'] = np
                exec_globals['numpy'] = np
            
            if 'pd.' in code or 'pandas' in code:
                import pandas as pd
                exec_globals['pd'] = pd
                exec_globals['pandas'] = pd
            
            # Exécuter le code
            exec(code, exec_globals)
            
            # Récupérer output
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # Sauvegarder plot si créé
            if plt.get_fignums():
                plt.savefig('plot.png', dpi=150, bbox_inches='tight')
                plt.close('all')
                output += "\n\n📊 Graphique créé et sauvegardé."
            
            return output if output else "✅ Code exécuté avec succès."
        
        except Exception as e:
            sys.stdout = old_stdout
            return f"❌ Erreur: {type(e).__name__}: {str(e)}"
    # Return list of tool objects
    return [
        search_course,
        generate_quiz_context,
        create_study_plan,
        search_wikipedia,
        python_interpreter, 
    ]


# --- 4. APPLICATION ---

def main():
    # ✨ CUSTOM CSS - Design lisible et moderne
    st.markdown("""
    <style>
    /* ===== RESET STREAMLIT ===== */
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
    
    /* ===== FOND PRINCIPAL ===== */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* ===== SIDEBAR - Fond BLANC ===== */
    [data-testid="stSidebar"] {
        background: white !important;
        border-right: 3px solid #667eea;
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }
    
    [data-testid="stSidebar"] * {
        color: #2d3748 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #667eea !important;
        font-weight: 700 !important;
    }
    
    /* Success badge dans sidebar */
    [data-testid="stSidebar"] .stSuccess {
        background: #48bb78 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
    }
    
    [data-testid="stSidebar"] .stWarning {
        background: #f56565 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* ===== TITRE PRINCIPAL ===== */
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
    
    /* ===== MESSAGES CHAT ===== */
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
    
    /* Message UTILISATEUR - Fond gradient, texte BLANC */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) * {
        color: white !important;
    }
    
    /* Message ASSISTANT - Fond blanc, texte NOIR */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) {
        background: white !important;
        border: 2px solid #667eea !important;
    }
    
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) * {
        color: #2d3748 !important;
    }
    
    /* Emojis dans les messages assistant */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) h1,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) h2,
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) h3 {
        color: #667eea !important;
    }
    
    /* ===== INPUT CHAT ===== */
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
        font-size: 1rem !important;
    }
    
    .stChatInputContainer textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Placeholder text */
    .stChatInputContainer textarea::placeholder {
        color: #a0aec0 !important;
    }
    
    /* ===== BOUTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: white !important;
        border-radius: 15px !important;
        padding: 1.5rem !important;
        border: 2px dashed #667eea !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #667eea !important;
        font-weight: 600 !important;
    }
    
    /* ===== EXPANDER (DEBUG) ===== */
    .streamlit-expanderHeader {
        background: #667eea !important;
        color: white !important;
        border-radius: 15px !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    .streamlit-expanderContent {
        background: white !important;
        color: #2d3748 !important;
        border-radius: 0 0 15px 15px !important;
        border: 2px solid #667eea !important;
        border-top: none !important;
        padding: 1rem !important;
    }
    
    /* ===== CODE BLOCKS ===== */
    code {
        background: #f7fafc !important;
        color: #667eea !important;
        padding: 0.2rem 0.5rem !important;
        border-radius: 5px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    pre {
        background: #f7fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        padding: 1rem !important;
    }
    
    pre code {
        color: #2d3748 !important;
    }
    
    /* ===== TABLES ===== */
    table {
        background: white !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }
    
    thead tr {
        background: #667eea !important;
    }
    
    thead th {
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem !important;
    }
    
    tbody tr {
        border-bottom: 1px solid #e2e8f0 !important;
    }
    
    tbody td {
        color: #2d3748 !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .stChatMessage {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* ===== MARKDOWN HEADINGS ===== */
    .stMarkdown h2 {
        color: #667eea !important;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
    }
    
    .stMarkdown h3 {
        color: #764ba2 !important;
        font-weight: 600 !important;
    }
    
    /* ===== LISTE À PUCES ===== */
    .stMarkdown ul {
        color: #2d3748 !important;
    }
    
    .stMarkdown li {
        margin: 0.5rem 0 !important;
    }
    
    /* ===== DIVIDER ===== */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #667eea, transparent) !important;
        margin: 2rem 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # ✨ TITRE CUSTOM
    st.markdown("""
    <div class="custom-title">
        <h1>🤖 Professeur IA</h1>
        <p class="custom-subtitle">Votre assistant intelligent pour réviser vos cours</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "doc_previews" not in st.session_state:
        st.session_state.doc_previews = {}
    if "tools" not in st.session_state:  # ← NEW: Store tools here
        st.session_state.tools = None
    
    # Sidebar
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader("PDF", type="pdf", accept_multiple_files=True)
        
        process_btn = st.button("Traiter les documents")
        
        if process_btn and files:
            with st.spinner("Analyse..."):
                # Process documents
                docs = process_documents(files)
                st.session_state.vectorstore = build_vector_store(docs)
                
                st.session_state.tools = create_all_tools(
                    vectorstore=st.session_state.vectorstore,
                    doc_previews=st.session_state.doc_previews
                )
                
                st.success("Prêt !")
    
    # ... rest of sidebar UI ...
        if st.session_state.vectorstore is not None:
            st.success("🟢 Mémoire chargée")
        else:
            st.warning("🔴 Mémoire vide")
        
        if st.session_state.doc_previews:
            st.markdown("### Fichiers en mémoire :")
            for f_name in st.session_state.doc_previews.keys():
                st.caption(f"📄 {f_name}")
    # Display chat history
    for msg in st.session_state.messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        elif msg.type == "ai" and msg.content:
            st.chat_message("assistant").write(msg.content)
            
    # Chat input
    user_input = st.chat_input("Votre question...")
    
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Check if tools are ready
        if st.session_state.tools is None:
            st.error("Veuillez d'abord uploader et traiter des documents!")
            st.stop()
        
        # Get API key
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Pas de clé API !")
            st.stop()
        
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="openai/gpt-oss-120b",
            temperature=0
        )
        
        # ✅ Use the tools from session state
        from langchain_experimental.tools import PythonREPLTool
        python_repl = PythonREPLTool()
        python_repl.name = "python_interpreter"
        python_repl.description = "Execute Python code for calculations."
        
        tools = st.session_state.tools 
        
        # Build system prompt
        docs_context = ""
        if st.session_state.doc_previews:
            docs_context = "\n\n**DOCUMENTS CHARGÉS:**\n"
            for filename in st.session_state.doc_previews.keys():
                docs_context += f"- {filename}\n"
        
        system_prompt = (
            "You are Professeur IA, an intelligent tutor helping students learn from their course documents.\n"
    f"{docs_context}\n\n"
    
    " **CRITICAL: YOU MUST FOLLOW THIS FORMAT FOR EVERY RESPONSE**\n\n"
    
    " **CHAIN OF THOUGHT (CoT) - MANDATORY FORMAT**\n"
    "EVERY response MUST show these 4 steps explicitly. NO EXCEPTIONS:\n\n"
    
    "**Step 1 - Pensée (Thought):**\n"
    "Analyze what the user is asking. What is the core question? What information do I need?\n\n"
    
    "**Step 2 - Action:**\n"
    "Decide which tool(s) to use and explain why. Be explicit about your tool choice.\n\n"
    
    "**Step 3 - Observation:**\n"
    "Show what you found after using the tool. Display the actual results or data.\n\n"
    
    "**Step 4 - Réponse:**\n"
    "Give your final answer based on the observation. Cite sources.\n\n"
    
    "---\n\n"
    
    " **MANDATORY TOOL USAGE RULES:**\n\n"
    
    "**FOR QUESTIONS ABOUT COURSE CONTENT:**\n"
    "1. ALWAYS call search_course() FIRST\n"
    "2. If search_course returns results → Use them in your answer\n"
    "3. If search_course returns nothing → Then try search_wikipedia\n"
    "4. NEVER skip directly to Wikipedia without trying search_course first\n\n"
    
    "**FOR DEFINITIONS/CONCEPTS:**\n"
    "Order: search_course → search_wikipedia (if needed)\n"
    "Always try the course documents before external sources.\n\n"
    
    "**FOR QUIZ REQUESTS (user says 'quiz me', 'test me', 'ask me about'):**\n"
    "1. Use generate_quiz_context to extract relevant content\n"
    "2. Create ONE multiple-choice question with 3-4 options\n"
    "3. DO NOT give the answer immediately\n"
    "4. Wait for user's response, then explain if correct/incorrect\n\n"
    
    "**FOR STUDY PLANNING (user asks for 'planning', 'schedule', 'révision'):**\n"
    "1. Use create_study_plan tool\n"
    "2. Return a Markdown table with columns:\n"
    "   - Jour (Day)\n"
    "   - Sujets à réviser (Topics)\n"
    "   - Objectifs d'apprentissage (Learning objectives)\n\n"
    
    " **FOR MATH/CALCULATIONS/PLOTTING:**\n"
    "Use python_interpreter tool with this pattern:\n"
    "- For calculations: use print() to show results\n"
    "- For plots: use plt.savefig('plot.png') at the end\n"
    "- Always include print statements for output\n\n"
    
        )
        
        # Create agent with tools
        agent_graph = create_agent(llm, tools=tools, system_prompt=system_prompt)
        
        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                try:
                    response = agent_graph.invoke({"messages": st.session_state.messages})
                    full_history = response["messages"]
                    final_answer = full_history[-1].content
            
                    # Afficher la réponse
                    st.markdown(final_answer)
            
                    # ✅ AFFICHER LE PLOT SI CRÉÉ
                    if os.path.exists("plot.png"):
                        st.image("plot.png", caption="📊 Résultat graphique", use_container_width=True)
                        os.remove("plot.png")  # Nettoyer après affichage
            
                    # Debug tools
                    used_tools = []
                    for msg in full_history:
                        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                used_tools.append(tool_call['name'])
            
                    if used_tools:
                        with st.expander("🔍 Debug: Outils utilisés", expanded=True):
                            st.write(f"**{', '.join(set(used_tools))}**")
            
                    st.session_state.messages = full_history
                except Exception as e:
                    st.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
