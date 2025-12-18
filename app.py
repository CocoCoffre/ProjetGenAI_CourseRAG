import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

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
    """Indexation."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

# --- 2. DÉFINITION DES SCHÉMAS  ---


# --- 3. DÉFINITION DE L'OUTIL AVEC @tool (TA DEMANDE) ---

@tool
def search_course(query: str) -> str:
    """
    Searches for information strictly within the uploaded PDF course documents.
    Use this tool to answer questions about the specific course content.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "Aucun document n'est chargé."
    
    # Recherche
    results = st.session_state.vectorstore.similarity_search(query, k=4)
    return "\n\n".join([doc.page_content for doc in results])
    
@tool
def search_wikipedia(query: str) -> str:
    """
    Searches for a general definition or historical fact on Wikipedia.
    Use this tool ONLY when asked about general knowledge concepts that you cannot found in the PDF.
    Do NOT used this tool if asked for a quizz.
    1. DO NOT COPY the raw text.
    2. Summarize the answer in 3-4 clear sentences.
    3. CLEANES mathematical formulas (removes LaTeX tags like 'displaystyle').

    Args:
        query: The exact subject or term to search for in JSON format (ex "Vanishing gradient", "Victor Hugo").
    """
    try:
        # Initialisation du retriever
        retriever = WikipediaRetriever(
            top_k_results=1,
            doc_content_chars_max=2000
        ) 
        # Invocation
        docs = retriever.invoke(query)
        
        # Vérification si on a des résultats
        if not docs:
            return "Aucun résultat trouvé sur Wikipedia pour cette recherche."
            
        return "\n\n".join([doc.page_content for doc in docs])
        
    except Exception as e:
        return f"Erreur lors de la recherche Wikipedia : {e}"

@tool
def generate_quiz_context(topic: str) -> str:
    """
    Extracts content from the course to prepare a quiz.
    Use this tool ONLY when the user explicitly asks for a quiz, a test, or an exercise or with a sentence like "Ask me about..."


    Args:
        topic: The specific keyword (e.g. "Vanishing gradient", "LSTM"). Avoid sentences, use keywords.
        
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "Impossible de faire un quiz : aucun cours chargé."
    
    retriever = st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 6, 'fetch_k': 20}
    )
    
    results = retriever.invoke(topic)
    
    if not results:
        return f"Aucune information trouvée dans le cours sur le sujet '{topic}'."

    content = "\n\n".join([doc.page_content for doc in results])
    
    # Petit debug (visible dans les logs Streamlit si besoin)
    print(f"DEBUG QUIZ - Topic: {topic} - Chunks found: {len(results)}")
    
    return f"Contenu du cours sur '{topic}' :\n{content}"

@tool
def create_study_plan(days: int, focus: str = "All") -> str:
    """
    Generates a revision schedule based on the uploaded documents.
    The LLM already knows which documents are available from the system prompt.
    
    Args:
        days: The number of days the user has to study.
        focus: Specific focus or 'All' to cover all uploaded documents.
    """
    
    # Try to access from session state as fallback
    previews = st.session_state.get("doc_previews", {})
    
    if not previews:
        return (
            "❌ Aucun document détecté. Vérifiez que vous avez uploadé et traité les PDFs.\n"
            "L'agent a reçu la liste des fichiers disponibles dans le contexte système, "
            "mais les données de prévisualisation ne sont pas accessibles."
        )
    
    context_str = "📚 **Documents disponibles pour la révision :**\n\n"
    for filename, preview in previews.items():
        # Show first 400 chars of each document
        context_str += f"**📖 {filename}**\n``````\n\n"
    
    return (
        f"{context_str}\n"
        f"**INSTRUCTION POUR L'AGENT :**\n"
        f"Crée un planning de révision détaillé sur **{days} jour(s)** en citant les thèmes principaux "
        f"de chaque document. Format obligatoire : tableau Markdown avec colonnes "
        f"(Jour | Sujets à réviser | Objectifs d'apprentissage)."
    )

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
    
    # --- INITIALISATION ROBUSTE ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "doc_previews" not in st.session_state:
        st.session_state.doc_previews = {}
        
    # Sidebar
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader("PDF", type="pdf", accept_multiple_files=True)
        
        process_btn = st.button("Traiter les documents")
        
        if process_btn and files:
            with st.spinner("Analyse..."):
                docs = process_documents(files)
                st.session_state.vectorstore = build_vector_store(docs)
                st.success("Prêt !")
        
        st.markdown("---")
        st.caption("Outils : RAG, Wiki, Quiz, Python, Planning")
        
        if st.session_state.vectorstore is not None:
            st.success("🟢 Mémoire chargée : L'agent a accès aux cours.")
        else:
            st.warning("🔴 Mémoire vide : L'agent ne connait pas le cours.")
            st.caption("👉 Veuillez cliquer sur 'Traiter les documents' ci-dessus.")
            
        if "doc_previews" in st.session_state and st.session_state.doc_previews:
            st.markdown("### Fichiers en mémoire :")
            for f_name in st.session_state.doc_previews.keys():
                st.caption(f"📄 {f_name}")
    
    # Affichage de l'historique
    for msg in st.session_state.messages:
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        elif msg.type == "ai" and msg.content:
            st.chat_message("assistant").write(msg.content)

    # Chat
    user_input = st.chat_input("Votre question...")

    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Pas de clé API !")
            st.stop()

        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="qwen/qwen3-32b",
            temperature=0
        )
        
        # ✅ KEY FIX: Extract doc_previews NOW, before creating agent
        current_doc_previews = st.session_state.get("doc_previews", {})
        
        tools = [
            search_course, 
            search_wikipedia, 
            generate_quiz_context, 
            python_repl_tool, 
            create_study_plan  # This will use the current_doc_previews from closure
        ]

        # ✅ Include current documents in the system prompt
        docs_context = ""
        if current_doc_previews:
            docs_context = "\n\n**DOCUMENTS CHARGÉS:**\n"
            for filename in current_doc_previews.keys():
                docs_context += f"- {filename}\n"
        else:
            docs_context = "\n\n**IMPORTANT: Aucun document n'est chargé pour le moment.**"
        
        system_prompt = (
         f"You are an intelligent and helpful Private Tutor named 'Professeur IA'.\n"
         f"Your goal is to help students learn based on their course documents.{docs_context}\n\n"
         "RULES:\n"
         "1. LANGUAGE: ALWAYS answer in the same language as the user's question.\n"
         "2. COURSE QUESTIONS: Use 'search_course' to find answers in the PDF.\n"
         "3. DEFINITIONS: Use 'search_wikipedia' for general definitions.\n"
         "4. QUIZZES: Use 'generate_quiz_context' with specific keywords.\n"
         "5. MATHS/LOGIC: Use 'python_interpreter'.\n"
         "6. PLANNING: Use 'create_study_plan' when user asks for a schedule.\n"
         "7. Be pedagogical, encouraging, and clear.")
        
        agent_graph = create_agent(llm, tools=tools, system_prompt=system_prompt)

        inputs = {"messages": st.session_state.messages}
        
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                try:
                    response = agent_graph.invoke(inputs)
                    full_history = response["messages"]
                    final_answer = full_history[-1].content
                    
                    st.write(final_answer)
    
                    if os.path.exists("plot.png"):
                        st.image("plot.png", caption="Graphique généré par l'Agent")
                        os.remove("plot.png")
                    
                    # Debug
                    last_human_index = -1
                    for i, msg in enumerate(full_history):
                        if isinstance(msg, HumanMessage):
                            last_human_index = i
                    
                    used_tools = []
                    if last_human_index != -1:
                        for msg in full_history[last_human_index:]:
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    used_tools.append(tool_call['name'])
                    
                    if used_tools:
                        unique_tools = list(set(used_tools))
                        with st.expander("🕵️‍♂️ Debug : Voir les outils utilisés", expanded=True):
                            st.write(f"Pour répondre, l'agent a utilisé : **{', '.join(unique_tools)}**")
                    else:
                        with st.expander("🕵️‍♂️ Debug", expanded=False):
                            st.write("L'agent a répondu directement (sans outils).")
                            
                    st.session_state.messages = full_history
                    
                except Exception as e:
                    st.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
