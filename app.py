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
    Perform a search on Wikipedia when asked specifically about a definition.
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
        st.markdown("---")
        st.caption("Outils disponibles : Cours, Wikipedia")

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
        system_prompt = (
            "You are a student assistant."
            "If you cannot answer using a tool, say: 'I cannot answer, this is beyond my capabilities'."
            "Choose the correct tool to answer. If it's in the course, use search_course. If it's a definition, use search_wikipedia."
        )
        
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
                    
                    # 5. METTRE A JOUR LA MEMOIRE DE SESSION
                    # On remplace l'historique par celui retourné par l'agent (qui contient les traces des outils)
                    st.session_state.messages = full_history
                    
                except Exception as e:
                    st.error(f"Erreur: {e}")

if __name__ == "__main__":
    main()
