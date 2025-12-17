import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Tes Imports (Version Officielle pour le Cloud) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ICI: On remplace 'langchain_classic' par 'langchain' (la vraie librairie)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- Configuration ---
st.set_page_config(page_title="Projet RAG Étudiant", page_icon="🎓")

# --- Fonctions Logiques (Code identique à ta structure) ---

def process_documents(uploaded_files):
    """Gère le chargement PDF (Loader)."""
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
    """Gère le découpage et les embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Utilisation du modèle HuggingFace standard
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_rag_chain(vectorstore):
    """Crée la chaîne RAG avec create_retrieval_chain (ta méthode préférée)."""
    
    # Récupération API Key
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("ERREUR : La clé GROQ_API_KEY est absente des secrets Streamlit.")
        st.stop()

    # LLM
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.3
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Réponds à la question de l'étudiant en te basant UNIQUEMENT sur le contexte ci-dessous.
    Si la réponse n'est pas dans le cours, dis "Je ne trouve pas l'info dans vos documents".

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # 1. Chaîne Document (Stuff Documents)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # 2. Chaîne Retriever (Retrieval Chain)
    retriever = vectorstore.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# --- Interface Utilisateur ---

def main():
    st.title("🎓 Assistant RAG Étudiant")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar
    with st.sidebar:
        st.header("1. Vos Cours")
        uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
        
        if st.button("Traiter les documents"):
            if uploaded_files:
                with st.spinner("Analyse en cours..."):
                    raw_docs = process_documents(uploaded_files)
                    st.session_state.vectorstore = build_vector_store(raw_docs)
                    st.success(f"C'est prêt ! {len(raw_docs)} pages lues.")
            else:
                st.warning("Il faut ajouter un PDF.")

    # Chat
    user_input = st.chat_input("Posez votre question ici...")

    if user_input:
        if st.session_state.vectorstore is None:
            st.warning("Veuillez d'abord traiter les documents (Sidebar).")
        else:
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Réflexion..."):
                    chain = get_rag_chain(st.session_state.vectorstore)
                    response = chain.invoke({"input": user_input})
                    st.write(response["answer"])

if __name__ == "__main__":
    main()
