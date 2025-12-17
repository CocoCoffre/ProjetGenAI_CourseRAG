import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Tes Imports (Adaptés pour Streamlit Cloud) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Note : 'langchain_classic' est un dossier local, sur le cloud on utilise la lib officielle :
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Config ---
st.set_page_config(page_title="Projet RAG Étudiant", page_icon="🎓")
load_dotenv()

# --- Fonctions Logiques (Remplacement de tes modules locaux) ---

def process_documents(uploaded_files):
    """
    Remplace 'download_data' et 'clean_transform'.
    Gère le stockage temporaire et le chargement via PyPDFLoader.
    """
    documents = []
    for file in uploaded_files:
        # Création fichier temporaire pour PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            documents.extend(docs)
        finally:
            os.remove(tmp_path) # Nettoyage
    return documents

def build_vector_store(documents):
    """
    Remplace ton module 'build_embeddings'.
    Split le texte et crée la base FAISS.
    """
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Vector Store
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_rag_chain(vectorstore):
    """Configuration de la chaîne RAG (LCEL)"""
    
    # Récupération API Key
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("Clé API Groq manquante dans les secrets Streamlit.")
        st.stop()

    # LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.3
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Réponds à la question en te basant uniquement sur le contexte fourni ci-dessous.
    Si tu ne trouves pas la réponse dans le contexte, dis simplement que tu ne sais pas.
    
    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Chaînes (LCEL standard)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)
    
    return retrieval_chain

# --- Application Streamlit ---

def main():
    st.title("🎓 Assistant RAG Étudiant")

    # Initialisation de la session
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar : Chargement des données
    with st.sidebar:
        st.header("1. Chargement des Cours")
        uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
        
        if st.button("Traiter les documents"):
            if uploaded_files:
                with st.spinner("Traitement en cours (Loader -> Split -> Embeddings)..."):
                    # Étape 1 : Chargement
                    raw_docs = process_documents(uploaded_files)
                    
                    # Étape 2 : Vector Store
                    st.session_state.vectorstore = build_vector_store(raw_docs)
                    
                    st.success(f"Succès ! {len(raw_docs)} pages indexées.")
            else:
                st.warning("Veuillez uploader un fichier PDF.")

    # Zone de Chat
    st.header("2. Discussion avec le cours")
    
    user_input = st.chat_input("Posez votre question ici...")

    if user_input:
        if st.session_state.vectorstore is None:
            st.warning("Veuillez d'abord traiter les documents dans la barre latérale.")
        else:
            # Affichage question utilisateur
            with st.chat_message("user"):
                st.write(user_input)

            # Génération réponse
            with st.chat_message("assistant"):
                with st.spinner("Réflexion..."):
                    rag_chain = get_rag_chain(st.session_state.vectorstore)
                    response = rag_chain.invoke({"input": user_input})
                    st.write(response["answer"])

if __name__ == "__main__":
    main()
