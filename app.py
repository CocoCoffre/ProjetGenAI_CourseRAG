import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Tes Imports Spécifiques (langchain-classic) ---
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Autres Imports nécessaires ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# --- Config ---
st.set_page_config(page_title="Projet RAG Étudiant", page_icon="🎓")

# --- Fonctions Backend ---

def process_documents(uploaded_files):
    """Gère le chargement PDF via fichiers temporaires."""
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
    """Crée les embeddings et la base vectorielle."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_rag_chain(vectorstore):
    """Crée la chaîne RAG avec langchain-classic."""
    
    # Récupération API Key
    groq_api_key = st.secrets.get("GROQ_API_KEY")
    if not groq_api_key:
        st.error("ERREUR : Clé GROQ_API_KEY manquante.")
        st.stop()

    # LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.3
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Réponds à la question en utilisant uniquement le contexte ci-dessous.
    Si tu ne sais pas, dis-le.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Utilisation des fonctions de langchain_classic
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Interface ---

def main():
    st.title("🎓 Assistant RAG Étudiant")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    with st.sidebar:
        st.header("1. Vos Cours")
        uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
        
        if st.button("Traiter les documents"):
            if uploaded_files:
                with st.spinner("Analyse en cours..."):
                    docs = process_documents(uploaded_files)
                    if docs:
                        st.session_state.vectorstore = build_vector_store(docs)
                        st.success(f"Terminé ! {len(docs)} pages analysées.")
                    else:
                        st.warning("Aucun texte extrait.")
            else:
                st.warning("Ajoutez un fichier d'abord.")

    user_input = st.chat_input("Votre question...")

    if user_input:
        if st.session_state.vectorstore:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Réflexion..."):
                    try:
                        chain = get_rag_chain(st.session_state.vectorstore)
                        res = chain.invoke({"input": user_input})
                        st.write(res["answer"])
                    except Exception as e:
                        st.error(f"Erreur : {e}")
        else:
            st.warning("Veuillez d'abord traiter les documents.")

if __name__ == "__main__":
    main()
