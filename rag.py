#!/usr/bin/env python
"""
RAG3_langchain.py
────────────────────────────
Retrieval-Augmented-Generation with full LangChain integration:

 • RecursiveCharacterTextSplitter (LangChain)
 • PDF + TXT ingestion (LangChain loaders)
 • FAISS vector search (LangChain wrapper)
 • OpenAI embeddings + chat (LangChain)
 • Chat history with InMemoryChatMessageHistory
 • Smart greeting detection

Dependencies
------------
pip install langchain langchain-openai langchain-community faiss-cpu pypdf tqdm python-dotenv
"""
from __future__ import annotations
import os
import textwrap
from pathlib import Path
import re

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from tqdm.auto import tqdm

# ─────────────────────────── Configuration ──────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("openai_key") 

DOCS_DIR = Path("C:\\Users\\j_aka\\Desktop\\Projet IA GENERATIVE\\Data")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
PERSIST_PATH = "faiss_store_langchain"

# ✅ Prompt amélioré qui permet les salutations
SYSTEM_PROMPT = (
    "You are a precise, concise tutor. "
    "For greetings and casual conversation, respond naturally and warmly. "
    "For knowledge questions, answer ONLY from the provided context. "
    "If the context doesn't contain the answer to a knowledge question, say 'I don't know.'"
)

# ─────────────────────────── Vérification de la clé API ──────────────────
assert OPENAI_API_KEY, "👉 Please set OPENAI_API_KEY (or openai_key) in your .env file first!"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ─────────────────────────── Détection des salutations ───────────────────
def is_greeting_or_casual(text: str) -> bool:
    "Détecte si le message est une salutation ou conversation casual."
    text_lower = text.lower().strip()
    
    # Liste de patterns pour salutations et conversations casual
    greeting_patterns = [
        r'\b(bonjour|salut|hello|hi|hey|coucou|bonsoir)\b',
        r'\b(ça va|comment ça va|comment vas-tu|how are you)\b',
        r'\b(merci|thank you|thanks)\b',
        r'\b(au revoir|bye|goodbye|à bientôt)\b',
        r'^(oui|non|ok|d\'accord|yes|no)$',
    ]
    
    for pattern in greeting_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Si la question est très courte (moins de 3 mots), probablement casual
    if len(text_lower.split()) <= 3 and '?' not in text_lower:
        return True
    
    return False

# ─────────────────────────── 1. Chargement et division des documents ─────
def load_and_split_documents() -> list:
    """Loads and splits documents from DOCS_DIR using LangChain loaders."""
    print(f"Scanning directory: {DOCS_DIR}")
    
    loaders = []
    
    # Collect all TXT files
    for path in DOCS_DIR.rglob("*.txt"):
        loaders.append(TextLoader(str(path), encoding="utf-8"))
    
    # Collect all PDF files
    for path in DOCS_DIR.rglob("*.pdf"):
        loaders.append(PyPDFLoader(str(path)))
    
    if not loaders:
        raise RuntimeError(f"No .txt or .pdf files found inside {DOCS_DIR.absolute()}")
    
    # Load all documents
    docs = []
    print("📖 Loading documents...")
    for loader in tqdm(loaders, desc="Loading files"):
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️  Error loading file: {e}")
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    split_docs = splitter.split_documents(docs)
    print(f"✓  Loaded {len(split_docs)} text chunks from {len(loaders)} files.")
    
    return split_docs

# ─────────────────────────── 2. Construction du Vector Store ─────────────
def get_or_build_vectorstore(documents: list) -> FAISS:
    """Loads existing FAISS store or builds a new one."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    if Path(PERSIST_PATH).exists():
        vectorstore = FAISS.load_local(
            PERSIST_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(PERSIST_PATH)
        print("✓  Vector store built and saved.")

    return vectorstore

# ─────────────────────────── 3. Configuration du RAG Chain ───────────────
def setup_rag_chain(vectorstore: FAISS):
    """Sets up the RAG chain with chat history support."""
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    
    # Create LLM
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    
    # ✅ Prompt pour les questions avec contexte RAG
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("placeholder", "{chat_history}"),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    # ✅ Prompt pour les conversations casual (sans contexte)
    casual_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a friendly, helpful assistant. Respond naturally to greetings and casual conversation."),
        ("placeholder", "{chat_history}"),
        ("human", "{question}")
    ])
    
    # Helper function to format retrieved documents
    def format_docs(docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"[Doc {i}]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
    # Build the RAG chain
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    
    # Build the casual chain (no retrieval)
    casual_chain = casual_prompt | llm | StrOutputParser()
    
    return rag_chain, casual_chain, retriever

# ─────────────────────────── 4. Boucle de Chat ───────────────────────────
def chat_loop(rag_chain, casual_chain, retriever):
    """Main chat loop with RAG and history."""
    
    # Initialize chat history
    chat_history = InMemoryChatMessageHistory()
    
    print("\n" + "="*60)
    print("        🤖 RAG Chat System Ready (LangChain Edition)")
    print("="*60)
    print("Type your questions below. Press Ctrl-C to quit.\n")
    
    while True:
        try:
            question = input("\n💬 You: ")
            if not question.strip():
                continue
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break
        
        # ✅ Détection du type de question
        is_casual = is_greeting_or_casual(question)
        
        # Prepare input with chat history
        input_data = {
            "question": question,
            "chat_history": chat_history.messages
        }
        
        # Get answer from appropriate chain
        try:
            if is_casual:
                # ✅ Utiliser la chaîne casual (pas de RAG)
                print("\n💭 (Casual conversation mode)")
                answer = casual_chain.invoke(input_data)
            else:
                # ✅ Utiliser la chaîne RAG (avec retrieval)
                retrieved_docs = retriever.invoke(question)
                
                # Show retrieved context
                print("\n🔍 Retrieved context:")
                print("─" * 60)
                for i, doc in enumerate(retrieved_docs, 1):
                    print(textwrap.indent(
                        textwrap.fill(doc.page_content, width=88), 
                        f"[Doc {i}] "
                    ))
                print("─" * 60)
                
                answer = rag_chain.invoke(input_data)
                
        except Exception as e:
            answer = f"An error occurred while calling the model: {e}"
        
        # Display answer
        print("\n🤖 Assistant:\n")
        print(textwrap.fill(answer, width=88))
        
        # Update chat history
        chat_history.add_user_message(question)
        chat_history.add_ai_message(answer)

# ─────────────────────────── Main ────────────────────────────────────────
if __name__ == "__main__":
    try:
        # 1. Load and split documents
        documents = load_and_split_documents()
        
        # 2. Get or build vector store
        vectorstore = get_or_build_vectorstore(documents)
        
        # 3. Setup RAG chain
        rag_chain, casual_chain, retriever = setup_rag_chain(vectorstore)
        
        # 4. Start chat loop
        chat_loop(rag_chain, casual_chain, retriever)
        
    except RuntimeError as e:
        print(f"\n❌ FATAL ERROR: {e}")
    except AssertionError as e:
        print(f"\n❌ FATAL ERROR: {e}")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
