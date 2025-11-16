#!/usr/bin/env python
"""
app.py
─────────────────────────────
Application Chainlit pour le système RAG multi-agents (LCEL)
Interface web modernisée avec meilleure gestion des erreurs
"""

import chainlit as cl
from pathlib import Path
import asyncio
from datetime import datetime

# Import du système RAG modernisé rag_lcel
from RAG import (
    init_rag_system,
    SmartAssistant,
    DOCS_DIR,
    CHAT_MODEL,
    TOP_K_DOCUMENTS
)


# ═══════════════════════════════════════════════════════════════════════
#                          CONFIGURATION CHAINLIT
# ═══════════════════════════════════════════════════════════════════════

# Paramètres globaux
APP_NAME = "Assistant RTE Intelligent"


# ═══════════════════════════════════════════════════════════════════════
#                          CHAINLIT EVENTS
# ═══════════════════════════════════════════════════════════════════════

@cl.on_chat_start
async def start():
    """Initialisation au démarrage du chat"""
    
    # Message de bienvenue avec loader
    msg = cl.Message(content="🔄 **Initialisation en cours...**")
    await msg.send()
    
    try:
        # Initialisation du système RAG
        rag_chain, retriever = init_rag_system()
        assistant = SmartAssistant(rag_chain, retriever)
        
        # Stockage dans la session utilisateur
        cl.user_session.set("assistant", assistant)
        cl.user_session.set("message_count", 0)
        cl.user_session.set("start_time", datetime.now())
        
        # Message de bienvenue final
        welcome_msg = f"""# 🤖 {APP_NAME}

### 🎯 Que puis-je faire pour vous ?

Je suis votre assistant spécialisé capable de :

✅ **vous aidez principalement dans vos recherche sur les documents RTE**
- Mécanisme d'ajustement, NEBEF, services système
- Règles de fréquence, marchés de l'électricité
- Procédures et guides internes
💡 **Astuce**: Posez des questions claires et précises pour de meilleurs résultats !
\n
 🤖 Modèle: {CHAT_MODEL}
"""
        msg.content = welcome_msg
        await msg.update()
        
        # Log de démarrage
        print(f"✅ Session démarrée à {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        error_msg = f"""❌ **Erreur lors de l'initialisation**

**Détails**: {str(e)}

**Solutions possibles**:
1. Vérifiez que `{DOCS_DIR}` contient des documents (.txt ou .pdf)
2. Vérifiez votre clé API OpenAI dans `.env`
3. Installez les dépendances: `pip install -r requirements.txt`
4. Supprimez `faiss_store/` et relancez

**Type d'erreur**: {type(e).__name__}
"""
        msg.content = error_msg
        await msg.update()
        print(f"❌ Erreur d'initialisation: {e}")


@cl.on_message
async def main(message: cl.Message):
    """Traitement des messages utilisateur"""
    
    assistant = cl.user_session.get("assistant")
    
    if not assistant:
        await cl.Message(
            content="❌ **Erreur système**\n\nL'assistant n'est pas initialisé. "
                   "Veuillez rafraîchir la page (F5)."
        ).send()
        return
    
    # Incrément compteur
    message_count = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", message_count)
    
    # Message avec loader
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Indicateur de traitement
        msg.content = "🔄 **Analyse en cours...**"
        await msg.update()
        
        # Traitement par l'assistant
        start_time = datetime.now()
        result = await assistant.process(message.content)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Construction de la réponse
        response = result["answer"]
        
        # Ajout des métadonnées
        metadata_parts = []
        
        # Route utilisée
        route_emoji = {
            "RAG": "📚",
            "TOOL": "🔧",
            "SIMPLE": "💬"
        }
        metadata_parts.append(
            f"{route_emoji.get(result['route'], '❓')} Mode: {result['route']}"
        )
        
        # Sources
        if result.get("sources"):
            sources_str = ", ".join(result["sources"][:3])
            if len(result["sources"]) > 3:
                sources_str += f" (+{len(result['sources'])-3} autres)"
            metadata_parts.append(f"📄 Sources: {sources_str}")
        
        # Temps de traitement
        metadata_parts.append(f"⏱️ {processing_time:.2f}s")
        
        # Footer
        footer = f"\n\n---\n*{' • '.join(metadata_parts)} • Message #{message_count}*"
        
        # Mise à jour finale
        msg.content = response + footer
        await msg.update()
        
        # Log
        print(f"✅ Message #{message_count} traité en {processing_time:.2f}s ({result['route']})")
        
    except Exception as e:
        error_response = f"""❌ **Erreur lors du traitement**

**Votre question**: {message.content}

**Erreur**: {str(e)}

**Type**: {type(e).__name__}

**Actions recommandées**:
- Reformulez votre question différemment
- Vérifiez que votre question est claire
- Si l'erreur persiste, rafraîchissez la page (F5)

Je reste disponible pour d'autres questions ! 😊
"""
        msg.content = error_response
        await msg.update()
        print(f"❌ Erreur traitement: {e}")


@cl.on_chat_end
async def on_chat_end():
    """Actions lors de la fermeture du chat"""
    message_count = cl.user_session.get("message_count", 0)
    start_time = cl.user_session.get("start_time")
    
    if start_time:
        duration = datetime.now() - start_time
        minutes = int(duration.total_seconds() / 60)
        seconds = int(duration.total_seconds() % 60)
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = "N/A"
    
    if message_count > 0:
        farewell = f"""# 👋 Au revoir !

Merci d'avoir utilisé **{APP_NAME}**.

### 📊 Statistiques de cette session
- 💬 Messages échangés: **{message_count}**
- ⏱️ Durée: **{duration_str}**
- 🤖 Modèle: **{CHAT_MODEL}**

---

💡 **Vos retours sont précieux !** N'hésitez pas à partager vos suggestions.

À bientôt ! 😊
"""
        await cl.Message(content=farewell).send()
        print(f"👋 Session terminée: {message_count} messages en {duration_str}")


# ═══════════════════════════════════════════════════════════════════════
#                          STARTERS (BOUTONS RAPIDES)
# ═══════════════════════════════════════════════════════════════════════

@cl.set_starters
async def set_starters():
    """Messages de démarrage suggérés"""
    return [
        cl.Starter(
            label="📄 Mécanisme d'ajustement",
            message="Explique-moi le mécanisme d'ajustement de RTE",
            icon="📄",
        ),
        cl.Starter(
            label="⚡ Services système",
            message="Quels sont les services système de fréquence chez RTE ?",
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
        cl.Starter(
            label="🌤️ Météo Paris",
            message="Quelle est la météo à Paris aujourd'hui ?",
            icon="🌤️",
        ),
        cl.Starter(
            label="🔢 Calculatrice",
            message="Calcule 1250 * 0.85 + 340",
            icon="🔢",
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════
#                          ACTIONS PERSONNALISÉES
# ═══════════════════════════════════════════════════════════════════════

@cl.action_callback("refresh_docs")
async def on_refresh_docs(action: cl.Action):
    """Action pour rafraîchir la base de documents"""
    refresh_msg = cl.Message(
        content="🔄 **Rafraîchissement en cours...**\n\n"
               "Suppression de la base vectorielle et rechargement des documents..."
    )
    await refresh_msg.send()
    
    try:
        # Suppression de la base existante
        import shutil
        if Path("faiss_store").exists():
            shutil.rmtree("faiss_store")
            await asyncio.sleep(0.5)  # Pause pour la stabilité
        
        # Réinitialisation
        rag_chain, retriever = init_rag_system()
        assistant = SmartAssistant(rag_chain, retriever)
        cl.user_session.set("assistant", assistant)
        
        refresh_msg.content = (
            "✅ **Base de documents rafraîchie avec succès !**\n\n"
            "La nouvelle base vectorielle est maintenant active.\n"
            "Vous pouvez poser vos questions sur les documents mis à jour."
        )
        await refresh_msg.update()
        
        print("✅ Base documentaire rafraîchie")
        
    except Exception as e:
        refresh_msg.content = (
            f"❌ **Erreur lors du rafraîchissement**\n\n"
            f"**Détails**: {str(e)}\n\n"
            f"**Type**: {type(e).__name__}\n\n"
            f"Veuillez réessayer ou contacter l'administrateur."
        )
        await refresh_msg.update()
        print(f"❌ Erreur rafraîchissement: {e}")


@cl.action_callback("show_stats")
async def on_show_stats(action: cl.Action):
    """Affiche les statistiques de la session"""
    message_count = cl.user_session.get("message_count", 0)
    start_time = cl.user_session.get("start_time")
    
    if start_time:
        duration = datetime.now() - start_time
        duration_str = f"{int(duration.total_seconds() / 60)}m {int(duration.total_seconds() % 60)}s"
    else:
        duration_str = "N/A"
    
    stats_msg = f"""### 📊 Statistiques de session

**Messages échangés**: {message_count}
**Durée de session**: {duration_str}
**Modèle IA**: {CHAT_MODEL}
**Documents indexés**: {TOP_K_DOCUMENTS} chunks max par requête
**Base documentaire**: {DOCS_DIR.name}

---
*Session démarrée à {start_time.strftime('%H:%M:%S') if start_time else 'N/A'}*
"""
    
    await cl.Message(content=stats_msg).send()


# ═══════════════════════════════════════════════════════════════════════
#                          PARAMÈTRES UTILISATEUR
# ═══════════════════════════════════════════════════════════════════════

@cl.on_settings_update
async def on_settings_update(settings):
    """Gestion des paramètres utilisateur (extensible)"""
    # Possibilité d'ajouter des paramètres configurables
    # Ex: température du modèle, nombre de documents, etc.
    print(f"⚙️  Paramètres mis à jour: {settings}")


# ═══════════════════════════════════════════════════════════════════════
#                          GESTION DES ERREURS
# ═══════════════════════════════════════════════════════════════════════

@cl.on_stop
async def on_stop():
    """Appelé quand l'utilisateur arrête la génération"""
    print("⏸️  Génération arrêtée par l'utilisateur")


#@cl.password_auth_callback
#def auth_callback(username: str, password: str):
 #   """Authentification (optionnel - à activer si nécessaire)"""
    # Implémentez votre logique d'authentification ici
    # Pour l'instant, désactivé
    #return None


# ═══════════════════════════════════════════════════════════════════════
#                          POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  {APP_NAME}                                    ║
║  Version: {APP_VERSION}                                       ║
╚══════════════════════════════════════════════════════════════╝

Pour lancer l'application:
  chainlit run app.py -w

Options:
  -w        Mode watch (rechargement auto)
  --port    Spécifier le port (défaut: 8000)
  --host    Spécifier l'host (défaut: localhost)

Exemple:
  chainlit run app.py -w --port 8080

Documentation: https://docs.chainlit.io
""")
