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
    Generates a revision schedule based on the ACTUAL filenames and content of uploaded documents.
    """
    # Add defensive check and better error handling
    if "doc_previews" not in st.session_state:
        st.session_state.doc_previews = {}
    
    previews = st.session_state.doc_previews
    
    if not previews:
        return "❌ Aucun document en mémoire. Veuillez d'abord uploader et traiter les PDFs."
    
    context_str = "📚 Voici les documents chargés :\n\n"
    for filename, preview in previews.items():
        context_str += f"**{filename}**\n``````\n\n"
    
    return (
        f"{context_str}\n"
        f"**INSTRUCTION** : Crée un planning détaillé sur {days} jours en citant ces fichiers exacts, "
        f"avec une table Markdown (Jour | Sujets à réviser | Objectifs d'apprentissage)."
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
    
    # --- INITIALISATION ROBUSTE (Tout en haut) ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    
        
    # Sidebar
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader("PDF", type="pdf", accept_multiple_files=True)
        
        # Bouton de traitement
        process_btn = st.button("Traiter les documents")
        
        if process_btn and files:
            with st.spinner("Analyse..."):
                docs = process_documents(files)
                st.session_state.vectorstore = build_vector_store(docs)
                st.success("Prêt !")
        st.markdown("---")
        st.caption("Outils : RAG, Wiki, Quiz, Python, Planning")
        
        # --- INDICATEUR D'ÉTAT (DEBUG VISUEL) ---
        if st.session_state.vectorstore is not None:
            st.success("🟢 Mémoire chargée : L'agent a accès aux cours.")
            # Petit bonus : Afficher le nombre de "chunks" (morceaux) en mémoire si possible
            # st.caption(f"{st.session_state.vectorstore.index.ntotal} fragments mémorisés.")
        else:
            st.warning("🔴 Mémoire vide : L'agent ne connait pas le cours.")
            st.caption("👉 Veuillez cliquer sur 'Traiter les documents' ci-dessus.")
            
        # Affichage des fichiers détectés
        if "doc_previews" in st.session_state and st.session_state.doc_previews:
            st.markdown("### Fichiers en mémoire :")
            for f_name in st.session_state.doc_previews.keys():
                st.caption(f"📄 {f_name}")
    # Affichage de l'historique
    for msg in st.session_state.messages:
        # LangGraph utilise des formats de messages spécifiques, on adapte l'affichage
        if msg.type == "human":
            st.chat_message("user").write(msg.content)
        elif msg.type == "ai" and msg.content: # On n'affiche que les réponses AI textuelles
            st.chat_message("assistant").write(msg.content)

    # Chat
    user_input = st.chat_input("Votre question...")

    if user_input:
        # 1. Affichage User
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        if "doc_previews" not in st.session_state:
            st.session_state.doc_previews = {} # Initialisation garantie
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Pas de clé API !")
            st.stop()

        # 2. Configuration Agent
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="qwen/qwen3-32b",
            temperature=0
        )
        
        tools = [search_course, search_wikipedia, generate_quiz_context, python_repl_tool, create_study_plan]

        # 3. Création de l'Agent (Syntaxe exacte create_agent)
        # Note: Dans la doc, checkpointer=None par défaut pour un agent stateless
        system_prompt = (
         "You are an intelligent and helpful Private Tutor named 'Professeur IA'.\n"
         "Your goal is to help students learn based on their course documents.\n\n"
         "RULES:\n"
         "1. LANGUAGE: ALWAYS answer in the same language as the user's question, regardless of the internal reasoning.\n"
         "2. COURSE QUESTIONS: Use 'search_course' to find answers in the PDF. Cite the context if possible.\n"
         "3. DEFINITIONS: Use 'search_wikipedia' for general definitions if you cannot find the answer in the PDF files.\n"
         "   - EXTRACT the main topic/keyword (e.g. user says 'Quiz on vanishing gradient problem', you extract 'Vanishing gradient').\n"
         "   - Use 'generate_quiz_context' with that specific keyword.\n"
         "   - Generate ONE multiple-choice question (A, B, C) based strictly on the retrieved context.\n"
         "   - DO NOT give the answer immediately. Wait for the user to reply.\n"
         "   - Once the user replies, correct them and explain why.\n"
         "4. 'generate_quiz_context' for quizzes ('ask me about...').\n"
         "5. 'python_interpreter' for MATHS, LOGIC, or PLOTTING.\n"
         "   - If asked to plot/draw: Write python code using matplotlib.\n"
         "   - Save the figure using `plt.savefig('plot.png')`.\n"
         "   - Do not try to show it with plt.show().\n"
         "6. 'create_study_plan': when user asks for a PLANNING or SCHEDULE.\n"
         "   - Extract the number of days (default to 3 if not specified).\n"
         "   - Output a clean MARKDOWN TABLE (Jour | Sujets | Objectifs).\n"
         "7. BEHAVIOR: Be pedagogical, encouraging, and clear.")
        
        agent_graph = create_agent(llm, tools=tools, system_prompt=system_prompt)

        # 4. Exécution (Syntaxe LangGraph)
        # On doit passer l'état actuel (les messages)
        inputs = {"messages": st.session_state.messages + [("human", user_input)]}
        
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                try:
                    # 2. INVOQUER L'AGENT AVEC L'HISTORIQUE COMPLET
                    # LangGraph va gérer les appels d'outils INTERNES ici
                    inputs = {"messages": st.session_state.messages}
                    response = agent_graph.invoke(inputs)
                    
                    # 3. RECUPERER LA REPONSE FINALE
                    # response["messages"] contient tout l'historique mis à jour (User + ToolCalls + ToolOutputs + AI Final)
                    full_history = response["messages"]
                    final_answer = full_history[-1].content
                    
                    # 4. AFFICHER LA REPONSE
                    st.write(final_answer)
    
                    if os.path.exists("plot.png"):
                        st.image("plot.png", caption="Graphique généré par l'Agent")
                        os.remove("plot.png") # Nettoyage
                    # 5. DEBUGGING
                    last_human_index = -1
                    for i, msg in enumerate(full_history):
                        if isinstance(msg, HumanMessage):
                            last_human_index = i
                    
                    # 2. On scanne les messages suivants à la recherche d'appels d'outils
                    used_tools = []
                    if last_human_index != -1:
                        for msg in full_history[last_human_index:]:
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    used_tools.append(tool_call['name'])
                    
                    # 3. Affichage visuel
                    if used_tools:
                        # set() pour éviter les doublons si l'outil est appelé 2 fois
                        unique_tools = list(set(used_tools))
                        with st.expander("🕵️‍♂️ Debug : Voir les outils utilisés", expanded=True):
                            st.write(f"Pour répondre, l'agent a utilisé : **{', '.join(unique_tools)}**")
                    else:
                        with st.expander("🕵️‍♂️ Debug", expanded=False):
                            st.write("L'agent a répondu directement (sans outils).")
                            
                    # 6. METTRE A JOUR LA MEMOIRE DE SESSION
                    # On remplace l'historique par celui retourné par l'agent (qui contient les traces des outils)
                    st.session_state.messages = full_history
                    
                except Exception as e:
                    st.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
