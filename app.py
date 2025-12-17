import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# --- Tes Imports Fonctionnels (Standardisés) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- Config ---
st.set_page_config(page_title="RAG Étudiant", page_icon="📚")
load_dotenv()

# --- Fonctions Backend ---

def get_documents_from_upload(uploaded_files):
    """Gère le chargement PDF via fichiers temporaires (requis pour PyPDFLoader)."""
    docs = []
    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.getvalue())
                tmp_path = tmp.name
            
            loader = PyPDFLoader(tmp_path)
            loaded_docs = loader.load()
            
            # Ajout métadonnée source propre
            for doc in loaded_docs:
                doc.metadata["source"] = file.name
                docs.append(doc)
                
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    return docs

def get_vectorstore(documents):
    """Crée le vectorstore FAISS."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def get_rag_chain(vectorstore):
    """Crée la chaîne RAG avec la nouvelle méthode create_retrieval_chain."""
    
    # 1. Configuration du LLM
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except:
        st.error("Clé API Groq manquante dans les secrets.")
        return None

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-70b-8192",
        temperature=0.3
    )

    retriever = vectorstore.as_retriever()

    # 2. Gestion de l'historique (Pour reformuler la question selon le contexte)
    context_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    context_prompt = ChatPromptTemplate.from_messages([
        ("system", context_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, context_prompt
    )

    # 3. Chaîne de réponse (QA Chain)
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 4. Chaîne Finale
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# --- Interface Utilisateur ---

def main():
    st.title("🎓 Assistant de Cours (RAG)")

    # Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # Sidebar
    with st.sidebar:
        st.header("Fichiers")
        uploaded_files = st.file_uploader("PDFs", type="pdf", accept_multiple_files=True)
        
        if st.button("Traiter les documents"):
            if uploaded_files:
                with st.spinner("Indexation..."):
                    docs = get_documents_from_upload(uploaded_files)
                    st.session_state.vectorstore = get_vectorstore(docs)
                    st.success("Prêt !")
            else:
                st.warning("Chargez un PDF d'abord.")

    # Chat Area
    user_input = st.chat_input("Votre question...")

    # Affichage historique
    for message in st.session_state.chat_history:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(message.content)

    # Traitement question
    if user_input:
        if st.session_state.vectorstore is None:
            st.warning("Veuillez d'abord traiter les documents.")
            return

        # Affichage message utilisateur
        st.chat_message("user").write(user_input)
        
        # Création de la chaîne (on la recrée pour être sûr d'avoir le bon contexte)
        rag_chain = get_rag_chain(st.session_state.vectorstore)
        
        if rag_chain:
            with st.chat_message("assistant"):
                # Invocation de la chaîne
                response = rag_chain.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.write(answer)
                
                # Mise à jour historique
                st.session_state.chat_history.extend([
                    HumanMessage(content=user_input),
                    AIMessage(content=answer)
                ])

if __name__ == "__main__":
    main()
