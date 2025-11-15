#!/usr/bin/env python
"""
app.py
──────
Interface Chainlit minimaliste pour RAG

Utilisation:
    chainlit run app.py
"""
import chainlit as cl
from rag import (
    is_greeting_or_casual,
    load_and_split_documents,
    get_or_build_vectorstore,
    setup_rag_chain
)
from langchain_core.chat_history import InMemoryChatMessageHistory


@cl.on_chat_start
async def start():
    """Initialise le système RAG silencieusement."""
    
    try:
        # Initialisation silencieuse (pas de messages intermédiaires)
        documents = load_and_split_documents()
        vectorstore = get_or_build_vectorstore(documents)
        rag_chain, casual_chain, retriever = setup_rag_chain(vectorstore)
        
        # Sauvegarder dans la session
        cl.user_session.set("rag_chain", rag_chain)
        cl.user_session.set("casual_chain", casual_chain)
        cl.user_session.set("retriever", retriever)
        cl.user_session.set("chat_history", InMemoryChatMessageHistory())
        
        # Un seul message de bienvenue
        await cl.Message(
            content="👋 **Bonjour ! Je suis prêt à répondre à vos questions.**"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ Erreur d'initialisation : {str(e)}"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    """Traite chaque message utilisateur."""
    
    # Récupérer les composants
    rag_chain = cl.user_session.get("rag_chain")
    casual_chain = cl.user_session.get("casual_chain")
    retriever = cl.user_session.get("retriever")
    chat_history = cl.user_session.get("chat_history")
    
    question = message.content
    
    if not rag_chain or not casual_chain or not retriever:
        await cl.Message(content="⚠️ Système non initialisé...").send()
        return
    
    try:
        is_casual = is_greeting_or_casual(question)
        input_data = {
            "question": question,
            "chat_history": chat_history.messages
        }
        
        if is_casual:
            # Mode conversation casual (pas de sources)
            answer = casual_chain.invoke(input_data)
            await cl.Message(content=answer).send()
            
        else:
            # Mode RAG avec affichage des sources seulement
            retrieved_docs = retriever.invoke(question)
            
            # Afficher uniquement les documents pertinents
            sources_content = "📚 **Documents pertinents :**\n\n"
            
            for i, doc in enumerate(retrieved_docs, 1):
                preview = doc.page_content[:200].strip()
                if len(doc.page_content) > 200:
                    preview += "..."
                
                sources_content += f"**[Document {i}]**\n{preview}\n\n"
            
            # Afficher les sources
            await cl.Message(
                content=sources_content,
                author="Sources"
            ).send()
            
            # Générer et afficher la réponse
            answer = rag_chain.invoke(input_data)
            await cl.Message(content=answer).send()
        
        # Mettre à jour l'historique
        chat_history.add_user_message(question)
        chat_history.add_ai_message(answer)
        
    except Exception as e:
        await cl.Message(content=f"❌ Erreur : {str(e)}").send()
