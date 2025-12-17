import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- Configuration de la page ---
st.set_page_config(page_title="Assistant Étudiant AI", page_icon="🎓")

# --- CSS personnalisé pour améliorer l'interface (optionnel) ---
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

# --- Fonctions Utilitaires ---

def get_pdf_text(pdf_docs):
    """Extrait le texte de plusieurs fichiers PDF."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Divise le texte en morceaux (chunks) gérables."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    """Crée la base de données vectorielle (FAISS) à partir des chunks."""
    # Utilisation d'un modèle léger et gratuit de HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    """Initialise la chaîne de conversation avec Groq."""
    
    # Récupération de la clé API depuis les secrets Streamlit
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        st.error("La clé API Groq n'est pas configurée dans les secrets.")
        return None

    # Initialisation du LLM Groq (Llama 3 est très performant)
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name="llama3-70b-8192",
        temperature=0.5
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    """Gère l'interaction utilisateur et l'affichage du chat."""
    if st.session_state.conversation is None:
        st.warning("Veuillez d'abord charger vos documents PDF.")
        return

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Affichage de l'historique inversé (le plus récent en bas est géré par streamlit chat logic standard)
    # Ici on affiche simplement l'échange
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

# --- Interface Principale ---

def main():
    st.header("🎓 Assistant Étudiant RAG (via Groq)")

    # Initialisation des variables de session
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # Zone de Chat
    user_question = st.chat_input("Posez une question sur vos cours...")
    if user_question:
        handle_userinput(user_question)

    # Sidebar pour le chargement des fichiers
    with st.sidebar:
        st.subheader("Vos Cours (PDF)")
        pdf_docs = st.file_uploader(
            "Chargez vos documents ici et cliquez sur 'Analyser'", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if st.button("Analyser les documents"):
            with st.spinner("Traitement en cours... (Lecture, Découpage, Indexation)"):
                if not pdf_docs:
                    st.error("Veuillez sélectionner au moins un fichier PDF.")
                else:
                    # 1. Extraction du texte
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Découpage en chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Création du Vector Store
                    vectorstore = get_vectorstore(text_chunks)
                    
                    # 4. Création de la chaîne de conversation
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    st.success("Terminé ! Vous pouvez maintenant discuter avec vos cours.")

if __name__ == '__main__':
    main()
