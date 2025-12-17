import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- IMPORTS STRICTS (Basés sur ta doc LangGraph) ---
from langchain.tools import tool
from langchain_groq import ChatGroq
# C'est LA fonction dont parle ta doc (qui construit un graph):
from langgraph.prebuilt import create_react_agent as create_agent 

from langchain_community.retrievers import WikipediaRetriever

# --- Imports standards RAG ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Config ---
st.set_page_config(page_title="Agent Étudiant (LangGraph)", page_icon="🤖")

# --- 1. BACKEND (PDF) ---

def process_documents(uploaded_files):
    """Lecture des PDF."""
    documents = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
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

# --- 2. DÉFINITION DE L'OUTIL AVEC @tool (TA DEMANDE) ---

@tool
def search_course(query: str) -> str:
    """
    Recherche des informations dans le cours PDF.
    Utilise cet outil pour répondre aux questions sur le contenu des documents.
    """
    if "vectorstore" not in st.session_state or st.session_state.vectorstore is None:
        return "Aucun document n'est chargé."
    
    # Recherche
    results = st.session_state.vectorstore.similarity_search(query, k=4)
    return "\n\n".join([doc.page_content for doc in results])
    
@tool
def search_wikipedia(query: str) -> str:
    """
    Cherche des définitions ou des faits historiques sur Wikipedia.
    Utilise cet outil pour les concepts généraux (ex: 'Qui est Victor Hugo ?', 'Définition de la mitose').
    """
    try:
        # On utilise exactement ton import
        retriever = WikipediaRetriever() 
        # On limite à 2 documents pour ne pas saturer le LLM
        retriever.top_k_results = 2
        
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Erreur Wikipedia : {e}"
        
# --- 3. APPLICATION ---

def main():
    st.title("🤖 Agent Étudiant")
    
    # Initialisation de la mémoire
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar
    with st.sidebar:
        st.header("Documents")
        files = st.file_uploader("PDF", type="pdf", accept_multiple_files=True)
        if st.button("Traiter") and files:
            with st.spinner("Analyse..."):
                docs = process_documents(files)
                st.session_state.vectorstore = build_vector_store(docs)
                st.success("Prêt !")

    # Affichage de l'historique
    for msg in st.session_state.messages:
        # LangGraph utilise des formats de messages spécifiques, on adapte l'affichage
        role = "user" if msg.type == "human" else "assistant"
        st.chat_message(role).write(msg.content)

    # Chat
    user_input = st.chat_input("Votre question...")

    if user_input:
        # 1. Affichage User
        st.chat_message("user").write(user_input)
        
        groq_api_key = st.secrets.get("GROQ_API_KEY")
        if not groq_api_key:
            st.error("Pas de clé API !")
            st.stop()

        # 2. Configuration Agent
        llm = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name="llama-3.3-70b-versatile",
            temperature=0
        )
        
        tools = [search_course, search_wikipedia]

        # 3. Création de l'Agent (Syntaxe exacte create_agent)
        # Note: Dans la doc, checkpointer=None par défaut pour un agent stateless
        agent_graph = create_agent(llm, tools=tools)

        # 4. Exécution (Syntaxe LangGraph)
        # On doit passer l'état actuel (les messages)
        inputs = {"messages": st.session_state.messages + [("human", user_input)]}
        
        with st.chat_message("assistant"):
            with st.spinner("Réflexion..."):
                try:
                    # Invoke renvoie le nouvel état final
                    response = agent_graph.invoke(inputs)
                    
                    # La réponse finale est le dernier message de l'agent
                    final_message = response["messages"][-1]
                    st.write(final_message.content)
                    
                    # Mise à jour de l'historique (On garde tout l'historique renvoyé par le graph)
                    st.session_state.messages = response["messages"]
                    
                except Exception as e:
                    st.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
