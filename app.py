import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- Configuration de la page ---
st.set_page_config(page_title="Assistant Étudiant AI", page_icon="🎓")

# --- CSS pour le chat (Look & Feel) ---
st.markdown("""
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
</style>
""", unsafe_allow_html=True)

# --- Fonctions Backend ---

def process_uploaded_files(uploaded_files):
    """
    Gère le chargement des fichiers :
    1. Sauvegarde temporaire sur le disque (requis par PyPDFLoader).
    2. Chargement avec LangChain (récupère texte + métadonnées page/source).
    3. Nettoyage des fichiers temporaires.
    """
    documents = []
    
    for uploaded_file in uploaded_files:
        # Création d'un fichier temporaire
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            # Chargement via PyPDFLoader
            loader = PyPDFLoader(tmp_file_path)
            loaded_docs = loader.load()
            
            # Ajout du nom du fichier d'origine aux métadonnées (plus propre que le chemin temp)
            for doc in loaded_docs:
                doc.metadata['source'] = uploaded_file.name
                documents.extend([doc])
                
        except Exception as e:
            st.error(f"Erreur lors du traitement de {uploaded_file.name}: {e}")
        finally:
            # Suppression du fichier temporaire pour ne pas saturer le serveur
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
            
    return documents

def get_vectorstore(documents):
    """Divise les documents et crée l'index vectoriel FAISS."""
    # Découpage intelligent (garde le contexte des phrases)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Création des embeddings (HuggingFace local sur CPU)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Création de la base vectorielle
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Initialise le LLM Groq et la mémoire de conversation."""
    
    # Récupération sécurisée de la clé API
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("🚨 Clé API Groq manquante ! Ajoutez-la dans les secrets Streamlit.")
        st.stop()

    # Modèle Llama 3 via Groq (Rapide et performant)
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama3-70b-8192",
        temperature=0.3
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer' # Important pour ConversationalRetrievalChain
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True # Permet de savoir d'où vient l'info (optionnel)
    )
    return conversation_chain

def handle_userinput(user_question):
    """Gère l'interface de chat."""
    if st.session_state.conversation is None:
        st.warning("Veuillez d'abord charger et analyser vos cours (PDF).")
        return

    # Appel au RAG
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Affichage de l'historique
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("assistant"):
                st.write(message.content)

# --- Main Application ---

def main():
    st.header("🎓 Assistant Étudiant RAG (Llama 3 & Groq)")

    # Initialisation Session State
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Zone de Chat principale
    user_question = st.chat_input("Posez une question sur vos cours...")
    if user_question:
        handle_userinput(user_question)

    # Sidebar (Menu de gauche)
    with st.sidebar:
        st.subheader("Vos Cours")
        pdf_docs = st.file_uploader(
            "Chargez vos PDFs ici", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Analyser les documents"):
            if not pdf_docs:
                st.warning("Ajoutez au moins un PDF avant de lancer l'analyse.")
            else:
                with st.spinner("Traitement des PDFs en cours..."):
                    # 1. Traitement des fichiers (Temp -> Loader)
                    raw_docs = process_uploaded_files(pdf_docs)
                    
                    # 2. Création Vector Store
                    vectorstore = get_vectorstore(raw_docs)
                    
                    # 3. Initialisation Chaîne
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.success(f"Analyse terminée ! {len(raw_docs)} pages traitées.")

if __name__ == '__main__':
    main()
