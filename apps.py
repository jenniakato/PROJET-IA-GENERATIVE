#!/usr/bin/env python
"""
app.py
─────────────────────────────
Application Chainlit pour le système RAG multi-agents
Interface web qui utilise rag_langchain_agents.py
"""

import chainlit as cl
from pathlib import Path

# Import du système RAG existant
from RAG import (
    init_rag_system,
    SmartAssistant,
    DOCS_DIR,
    CHAT_MODEL,
    TOP_K_DOCUMENTS
)


# ═══════════════════════════════════════════════════════════════════════
#                          CHAINLIT EVENTS
# ═══════════════════════════════════════════════════════════════════════

@cl.on_chat_start
async def start():
    """Initialisation au démarrage du chat"""
    
    # Message de bienvenue avec loader
    msg = cl.Message(content="🔄 **Initialisation en cours...**\n\nChargement des documents et des outils...")
    await msg.send()
    
    try:
        # Initialisation du système RAG depuis le module existant
        rag_chain = init_rag_system()
        assistant = SmartAssistant(rag_chain)
        
        # Stockage dans la session utilisateur
        cl.user_session.set("assistant", assistant)
        cl.user_session.set("message_count", 0)
        
        # Message de bienvenue final
        welcome_msg = f"""# 🤖 Assistant RTE Intelligent

Bienvenue ! Je suis votre assistant spécialisé dans les **règles et systèmes RTE**.

## 📚 Mes capacités :

### 📄 Documents RTE (PRIORITÉ)
Recherche dans vos documents RTE : règles, services système, mécanisme d'ajustement, marchés...
- *Exemple : "Qu'est-ce que le mécanisme d'ajustement ?"*
- *Exemple : "Comment fonctionne NEBEF ?"*
- *Exemple : "Règles de contribution au RPT ?"*

### 🔢 Calculatrice
Calculs mathématiques précis
- *Exemple : "Calcule 1234 × 56"*

### 🌐 Recherche web
Informations externes en temps réel : météo, actualités
- *Exemple : "Quelle est la météo à Paris ?"*

### ⚡ Calendrier Tempo
Jours tarifaires Tempo EDF : Rouge, Blanc, Bleu
- *Exemple : "Quel est le jour Tempo aujourd'hui ?"*

### 💬 Conversation
Discussion naturelle et conviviale
- *Exemple : "Bonjour !"*

---

## ⚙️ Configuration actuelle :
- 📁 Documents RTE : `{DOCS_DIR.name}/`
- 🔍 Récupération : Top {TOP_K_DOCUMENTS} documents
- 🧠 Modèle : {CHAT_MODEL}

---

**Posez-moi votre question sur RTE ci-dessous !** 👇
"""
        
        # Mise à jour du message avec le contenu final
        msg.content = welcome_msg
        await msg.update()
        
    except Exception as e:
        error_msg = f"""❌ **Erreur lors de l'initialisation**

**Détails :** {str(e)}

**Solutions possibles :**
1. Vérifiez que le dossier `{DOCS_DIR}` existe et contient des documents
2. Vérifiez votre clé API OpenAI dans le fichier `.env`
3. Vérifiez que tous les modules sont installés : `pip install -r requirements_chainlit.txt`
4. Si le problème persiste, supprimez le dossier `faiss_store/` et relancez

**Besoin d'aide ?** Consultez `README_CHAINLIT.md`
"""
        msg.content = error_msg
        await msg.update()


@cl.on_message
async def main(message: cl.Message):
    """Traitement des messages utilisateur"""
    
    # Récupération de l'assistant depuis la session
    assistant = cl.user_session.get("assistant")
    
    if not assistant:
        await cl.Message(
            content="❌ **Erreur système**\n\nL'assistant n'est pas initialisé. Veuillez rafraîchir la page (F5).",
        ).send()
        return
    
    # Incrément du compteur de messages
    message_count = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", message_count)
    
    # Création d'un message vide avec loader
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Affichage du statut de traitement
        msg.content = "🔄 Traitement en cours..."
        await msg.update()
        
        # Traitement de la question par l'assistant
        response = await assistant.process(message.content)
        
        # Ajout d'un footer avec info contextuelle
        footer = f"\n\n---\n*Message #{message_count} • Posez une autre question ou dites 'merci' pour terminer*"
        
        # Mise à jour avec la réponse finale
        msg.content = response + footer
        await msg.update()
        
    except Exception as e:
        error_response = f"""❌ **Erreur lors du traitement**

**Votre question :** {message.content}

**Erreur :** {str(e)}

**Que faire ?**
- Reformulez votre question différemment
- Vérifiez que votre question est claire
- Si l'erreur persiste, rafraîchissez la page

Je reste à votre disposition pour d'autres questions ! 😊
"""
        msg.content = error_response
        await msg.update()


# ═══════════════════════════════════════════════════════════════════════
#                      CONFIGURATION CHAINLIT
# ═══════════════════════════════════════════════════════════════════════

@cl.set_starters
async def set_starters():
    """Messages de démarrage suggérés (boutons rapides)"""
    return [
        cl.Starter(
            label="📄 Mécanisme d'ajustement",
            message="Qu'est-ce que le mécanisme d'ajustement de RTE ?",
            icon="📄",
        ),
        cl.Starter(
            label="⚡ Services système",
            message="Explique-moi les règles des services système et fréquence",
            icon="⚡",
        ),
        cl.Starter(
            label="📊 NEBEF",
            message="Comment fonctionne NEBEF pour les effacements ?",
            icon="📊",
        ),
        cl.Starter(
            label="🔵 Tempo aujourd'hui",
            message="Quel est le jour Tempo aujourd'hui ?",
            icon="🔵",
        ),
    ]


@cl.on_chat_end
async def on_chat_end():
    """Actions lors de la fermeture du chat"""
    message_count = cl.user_session.get("message_count", 0)
    
    if message_count > 0:
        farewell = f"""# 👋 Au revoir !

Merci d'avoir utilisé l'assistant intelligent.

**Statistiques de cette session :**
- 💬 Messages échangés : {message_count}

À bientôt ! 😊
"""
        await cl.Message(content=farewell).send()


@cl.on_settings_update
async def on_settings_update(settings):
    """Gestion des mises à jour de paramètres (optionnel)"""
    # Possibilité d'ajouter des paramètres configurables par l'utilisateur
    pass


# ═══════════════════════════════════════════════════════════════════════
#                          ACTIONS PERSONNALISÉES
# ═══════════════════════════════════════════════════════════════════════

@cl.action_callback("refresh_docs")
async def on_refresh_docs(action: cl.Action):
    """Action pour rafraîchir la base de documents"""
    await cl.Message(
        content="🔄 **Rafraîchissement en cours...**\n\nSuppression de la base vectorielle et rechargement des documents..."
    ).send()
    
    try:
        # Suppression de la base existante
        import shutil
        if Path("faiss_store").exists():
            shutil.rmtree("faiss_store")
        
        # Réinitialisation
        rag_chain = init_rag_system()
        assistant = SmartAssistant(rag_chain)
        cl.user_session.set("assistant", assistant)
        
        await cl.Message(
            content="✅ **Base de documents rafraîchie !**\n\nVous pouvez maintenant poser vos questions sur les nouveaux documents."
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Erreur lors du rafraîchissement** : {str(e)}"
        ).send()


# Note : Pour activer cette action, ajoutez un bouton dans un message :
# actions = [cl.Action(name="refresh_docs", value="refresh", label="🔄 Rafraîchir les documents")]
# await cl.Message(content="...", actions=actions).send()


# ═══════════════════════════════════════════════════════════════════════
#                          POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Pour lancer l'application :
    # chainlit run app.py
    
    # Ou avec watch mode (rechargement auto) :
    # chainlit run app.py -w
    
    # Ou avec host et port personnalisés :
    # chainlit run app.py --host 0.0.0.0 --port 8080
    
    print("Pour lancer l'application Chainlit :")
    print("  chainlit run app.py")
    print("\nOu utilisez le script de démarrage :")
    print("  Windows: start_chainlit.bat")
    print("  Linux/Mac: ./start_chainlit.sh")