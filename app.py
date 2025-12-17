import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Nouveaux Imports pour l'Agent ---
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --- Imports Standards ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- Config ---
st.set_page_config(page_title="Agent Étudiant ReAct", page_icon="🤖")

def process_documents(uploaded_files):
    """Charge et lit les PDF."""
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
    """Indexe les documents."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_agent_executor(vectorstore):
    """
    Crée l'agent ReAct capable d'utiliser le cours comme un outil.
    """
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Clé API manquante.")
        st.stop()

    # 1. Le LLM (Cerveau)
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama-3.3-70b-versatile", 
        temperature=0
    )

    # 2. Création de l'outil de recherche (Step 1)
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        name="recherche_cours_pdf",
        description="Utilise cet outil pour trouver des informations dans les documents de cours PDF fournis par l'utilisateur. Cherche toujours ici en premier si la question porte sur le cours."
    )
    
    tools = [retriever_tool]

    # 3. Le Prompt de l'Agent (Instruction système)
    # On définit comment l'agent doit se comporter
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un assistant étudiant intelligent. Tu as accès à des documents de cours. "
                   "Utilise tes outils pour répondre aux questions. "
                   "Si l'information est dans le cours, cite le contexte. "
                   "Si la question est hors sujet (ex: météo), dis que tu ne peux répondre qu'au cours."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # IMPORTANT: Là où l'agent 'réfléchit'
    ])

    # 4. Construction de l'Agent (Step 2)
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 5. L'Exécuteur (Celui qui fait tourner la boucle)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True # Affiche le raisonnement dans la console (logs)
    )
    
    return agent_executor

def main():
    st.title("🤖 Agent Étudiant (Mode ReAct)")
    st.markdown("Cet agent peut **décider** d'utiliser vos cours pour répondre.")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar : Chargement
    with st.sidebar:
        st.header("Base de Connaissances")
        files = st.file_uploader("Ajouter des cours (PDF)", type="pdf", accept_multiple_files=True)
        if st.button("Analyser et Indexer") and files:
            with st.spinner("Lecture et indexation..."):
                docs = process_documents(files)
                st.session_state.vectorstore = build_vector_store(docs)
                st.success(f"{len(docs)} pages ingérées dans la mémoire.")

    # Chat Interface
    question = st.chat_input("Pose ta question à l'agent...")
    
    if question:
        # Affichage message utilisateur
        st.chat_message("user").write(question)

        if st.session_state.vectorstore is None:
            st.warning("Attention : Aucun cours chargé. L'agent ne pourra compter que sur ses connaissances générales.")
            # On pourrait empêcher l'exécution, mais un Agent peut aussi répondre sans outils !
            # Pour l'exercice, on va quand même demander les docs
        else:
            with st.chat_message("assistant"):
                with st.spinner("L'agent réfléchit et consulte ses outils..."):
                    try:
                        # Création de l'agent avec les outils liés au vectorstore actuel
                        agent_executor = get_agent_executor(st.session_state.vectorstore)
                        
                        # Exécution
                        response = agent_executor.invoke({"input": question})
                        
                        # Affichage réponse finale
                        st.write(response["output"])
                        
                    except Exception as e:
                        st.error(f"Erreur agent : {e}")

if __name__ == "__main__":
    main()
