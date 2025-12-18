import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- IMPORTS STRICTS (Basés sur ta doc LangGraph) ---
from langchain.tools import tool, Tool
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

def create_all_tools(vectorstore: FAISS, doc_previews: dict):
    """
    Factory function that creates all tools with access to vectorstore and previews.
    Call this AFTER documents are processed.
    """
    
    # Tool 1: search_course
    def _search_course(query: str) -> str:
        """Searches the course PDFs."""
        print(f"\n[DEBUG search_course] Query: {query}")
        results = vectorstore.similarity_search(query, k=4)
        print(f"  Results found: {len(results)}")
        
        if not results:
            return f"No content found for '{query}' in documents."
        
        formatted = []
        for doc in results:
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"[From {source}]\n{doc.page_content}")
        
        return "\n---\n".join(formatted)
    
    # Tool 2: generate_quiz_context
    def _generate_quiz(topic: str) -> str:
        """Extracts course content for quiz generation."""
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={'k': 6, 'fetch_k': 20}
        )
        results = retriever.invoke(topic)
        
        if not results:
            return f"No information found on '{topic}'."
        
        content = "\n\n".join([doc.page_content for doc in results])
        return f"Course content on '{topic}':\n{content}"
    
    # Tool 3: create_study_plan
    def _create_plan(days: int, focus: str = "All") -> str:
        """Creates study plan based on documents."""
        context_str = "Documents available:\n\n"
        for filename, preview in doc_previews.items():
            context_str += f"{filename}\n{preview[:300]}...\n\n"
        
        if not context_str or context_str == "Documents available:\n\n":
            context_str = "No preview available"
        
        return (
            f"{context_str}\n"
            f"Create a {days}-day study plan with table: (Jour | Sujets | Objectifs)"
        )
    
    # Tool 4: search_wikipedia (doesn't need vectorstore)
    def _search_wikipedia(query: str) -> str:
        """Searches Wikipedia for general knowledge."""
        try:
            from langchain_community.retrievers import WikipediaRetriever
            retriever = WikipediaRetriever(top_k_results=1, doc_content_chars_max=2000)
            docs = retriever.invoke(query)
            if not docs:
                return "No Wikipedia results found."
            return "\n\n".join([doc.page_content for doc in docs])
        except Exception as e:
            return f"Wikipedia search error: {e}"
    
    # Return all tools as a list
    return [
        Tool(
            name="search_course",
            description="Search uploaded PDF course documents for information.",
            func=_search_course,
        ),
        Tool(
            name="generate_quiz_context",
            description="Extract course content to prepare a quiz. Use when user asks for quiz/test.",
            func=_generate_quiz,
        ),
        Tool(
            name="create_study_plan",
            description="Create revision schedule based on documents. Use when user asks for planning/schedule.",
            func=_create_plan,
        ),
        Tool(
            name="search_wikipedia",
            description="Search Wikipedia for general knowledge. Use only if search_course finds nothing.",
            func=_search_wikipedia,
        ),
        # Add python_repl_tool separately (it's already a tool object)
    ]

# --- 3. LE DATA SCIENTIST (PYTHON REPL) ---
# On instancie l'outil officiel
python_repl_tool = PythonREPLTool()
python_repl_tool.name = "python_interpreter"
python_repl_tool.description = (
    "A Python shell. Use this to execute python commands. "
    "Input should be a valid python script. "
    "Use this to solve math problems, calculating statistics, or plotting data. "
    "If you want to see the output of a value, you should print it with `print(...)`."
    "If you generate a plot, save it as 'plot.png'."
)

# --- 4. APPLICATION ---

def main():
    st.title("🤖 Agent Étudiant")
    
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
                
                # ✅ CREATE TOOLS HERE (after vectorstore is ready)
                st.session_state.tools = create_all_tools(
                    vectorstore=st.session_state.vectorstore,
                    doc_previews=st.session_state.doc_previews
                )
                
                st.success("Prêt !")
    
    # ... rest of sidebar UI ...
    
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
            model_name="qwen/qwen3-32b",
            temperature=0
        )
        
        # ✅ Use the tools from session state
        from langchain_experimental.tools import PythonREPLTool
        python_repl = PythonREPLTool()
        python_repl.name = "python_interpreter"
        python_repl.description = "Execute Python code for calculations."
        
        tools = st.session_state.tools + [python_repl]
        
        # Build system prompt
        docs_context = ""
        if st.session_state.doc_previews:
            docs_context = "\n\n**DOCUMENTS CHARGÉS:**\n"
            for filename in st.session_state.doc_previews.keys():
                docs_context += f"- {filename}\n"
        
        system_prompt = (
            "You are Professeur IA, a helpful tutor.\n"
            f"{docs_context}\n\n"
            "MANDATORY RULES:\n"
            "1. Always use search_course FIRST before search_wikipedia\n"
            "2. Show Chain of Thought: Pensée → Action → Observation → Réponse\n"
            "3. Never skip search_course!\n"
        )
        
        # Create agent with tools
        agent_graph = create_agent(llm, tools=tools, system_prompt=system_prompt)
        
        # Run agent
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                try:
                    response = agent_graph.invoke({"messages": st.session_state.messages})
                    final_answer = response["messages"][-1].content
                    st.write(final_answer)
                    st.session_state.messages = response["messages"]
                except Exception as e:
                    st.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
