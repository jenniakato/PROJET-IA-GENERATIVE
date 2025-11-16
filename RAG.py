#!/usr/bin/env python
"""
rag_langchain_agents.py
─────────────────────────────
RAG avec agents spécialisés et routage intelligent strict (RAG ou Agent)

Capacités:
▪️ RAG sur documents internes (PRIORITÉ ABSOLUE si sujets internes)
▪️ Calculatrice pour calculs mathématiques
▪️ Recherche web DuckDuckGo (externe uniquement)
▪️ Jours de tension RTE (Tempo)
▪️ Conversation simple sans outils

Stratégie de routage STRICTE (implémentée dans SmartAssistant.process):
1. Questions internes claires sur RTE/Règles/Marchés → Pipeline RAG DIRECT
2. Questions externes claires (météo, calculs, RTE tension) → Agent/Outils spécialisés
3. Sinon, conversation simple → Réponse directe
"""

from pathlib import Path
import os, textwrap, re
from typing import Dict, Any, List

# ── LangChain core ─────────────────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ── Agents & Tools ─────────────────────────────────────────────────────
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# ── Agent RTE personnalisé (assurez-vous que cette classe est disponible) ────────────────
# from Jourdetension import JourDeTensionRTE, format_reponse_rte 

from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key", "")

# ═══════════════════════════════════════════════════════════════════════
#                          CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

DOCS_DIR        = Path("C:\\Users\\j_aka\\Desktop\\Projet IA GENERATIVE\\Data")
EMBED_MODEL     = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o-mini"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
PERSIST_PATH    = "faiss_store"
TOP_K_DOCUMENTS = 10  # Nombre de documents à récupérer

# Export des variables pour app.py
__all__ = [
    'init_rag_system',
    'SmartAssistant',
    'DOCS_DIR',
    'TOP_K_DOCUMENTS',
    'CHAT_MODEL',
]

SYSTEM_PROMPT_RAG = (
    "You are a precise, concise tutor. "
    "Answer ONLY from the provided context. "
    "If the answer is missing, say 'I don't know.'"
)

# PROMPT SIMPLIFIÉ pour l'agent, car le RAG est géré en amont par le routeur principal.
SYSTEM_PROMPT_ROUTER = """Tu es un assistant intelligent spécialisé dans les règles et systèmes RTE (Réseau de Transport d'Électricité).
Tu as été sollicité pour une question qui nécessite l'un de tes outils spécialisés (calcul, météo, ou calendrier Tempo).

RÈGLE ABSOLUE: Une fois que l'outil a renvoyé un résultat, tu DOIS utiliser ce résultat pour formuler ta réponse à l'utilisateur. Ne dis jamais 'Je ne peux pas fournir d'informations...' si un outil t'a donné des données.
Si les données sont non pertinentes (ex: météo pour une autre ville), tu dois l'expliquer.

RÈGLES DE ROUTAGE STRICTES:
1. calculator: UNIQUEMENT pour calculs mathématiques explicites (nombres, opérations)
   → Exemples: "calcule 12*5", "combien font 100+50"

2. rte_tension: UNIQUEMENT pour les jours Tempo (Rouge/Blanc/Bleu) - calendrier tarifaire
   → Exemples: "jour Tempo aujourd'hui", "couleur Tempo demain"

3. web_search: UNIQUEMENT pour informations externes NON liées à RTE
   → Exemples: météo, actualités mondiales, événements récents externes
   → NE PAS utiliser pour questions sur RTE, règles, marchés, etc.

Le RAG sur documents internes n'est PAS un outil que tu dois appeler, car le routeur principal l'a déjà écarté ou l'a appelé directement.
Choisis l'outil le plus pertinent pour la requête utilisateur."""

# ═══════════════════════════════════════════════════════════════════════
#                          OUTILS / AGENTS
#               (Le RAG n'est plus ici, il est appelé directement)
# ═══════════════════════════════════════════════════════════════════════

# ────────────────────────── Calculatrice ──────────────────────────────
def calculatrice(expression: str) -> str:
    """Évalue une expression mathématique de manière sécurisée"""
    try:
        expression = expression.strip().replace(',', '.')
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
            return "❌ Expression invalide. Utilisez uniquement +, -, *, /, (), nombres."
        
        resultat = eval(expression, {"__builtins__": {}}, {})
        return f"✓ Résultat: {resultat}"
    except Exception as e:
        return f"❌ Erreur de calcul: {str(e)}"


# ────────────────────────── DuckDuckGo ────────────────────────────────
def creer_outil_web_search():
    """Crée l'outil de recherche web DuckDuckGo"""
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    search = DuckDuckGoSearchRun(api_wrapper=wrapper)
    
    async def run_search_async(query: str) -> str:
        """Version async de la recherche web"""
        import asyncio
        result = await asyncio.to_thread(search.run, query)
        return result
    
    return Tool.from_function(
        name="web_search",
        description="Recherche web en temps réel EXTERNE pour informations NON liées à RTE (Réseau de Transport d'Électricité). "
                   "UTILISE CET OUTIL pour la **météo**, l'**heure**, les **actualités générales**, les infos sur les prix spot ,sur les certificats d'economies d'energies."
                   "La requête de recherche doit être la plus **précise** possible (ex: 'météo Paris', 'heure Tokyo').",
                   
        func=run_search_async,
        coroutine=run_search_async
    )


# ────────────────────────── RTE Tension ───────────────────────────────
# NOTE: Cette fonction utilise JourDeTensionRTE, assurez-vous qu'elle est importable.
def creer_outil_rte():
    """Crée l'outil pour les jours de tension RTE"""
    
    # Simulation de la classe JourDeTensionRTE et de format_reponse_rte pour que le script soit complet
    # Dans un vrai projet, retirez ces simulations.
    try:
        from Jourdetension import JourDeTensionRTE, format_reponse_rte 
    except ImportError:
        class JourDeTensionRTE:
            def get_jour_tension(self, date=None): return "Simulated Blue Day"
            def get_prevision_semaine(self): return "Simulated 7-day forecast: Blue, Blue, White, Red..."
        def format_reponse_rte(data): return f"⚡ Tempo Result: {data}"
    
    agent_rte = JourDeTensionRTE()
    
    async def run_rte_async(query: str) -> str:
        """Execute l'agent RTE selon la requête (version async)"""
        import asyncio
        from datetime import datetime, timedelta
        
        query_lower = query.lower()
        
        def get_rte_data():
            if any(word in query_lower for word in ["semaine", "prochains jours", "prévision", "7 jours"]):
                return agent_rte.get_prevision_semaine()
            elif "demain" in query_lower:
                date_demain = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                return agent_rte.get_jour_tension(date_demain)
            else:
                return agent_rte.get_jour_tension()
        
        data = await asyncio.to_thread(get_rte_data)
        return format_reponse_rte(data)
    
    return Tool.from_function(
        name="rte_tension",
        description="UNIQUEMENT pour le calendrier Tempo EDF (jours Rouge/Blanc/Bleu) - tarification. "
                   "Exemples: 'jour Tempo aujourd'hui', 'couleur Tempo demain', 'prévision Tempo semaine'.",
        func=run_rte_async,
        coroutine=run_rte_async
    )


# ═══════════════════════════════════════════════════════════════════════
#                          INITIALISATION RAG
# ═══════════════════════════════════════════════════════════════════════

def load_documents():
    """Charge et découpe les documents"""
    loaders = []
    for path in DOCS_DIR.rglob("*.txt"):
        loaders.append(TextLoader(str(path)))
    for path in DOCS_DIR.rglob("*.pdf"):
        loaders.append(PyPDFLoader(str(path)))

    if not loaders:
        # Permet de continuer si DOCS_DIR est vide, pour tester les agents
        print(f"⚠️  Warning: No .txt or .pdf found inside {DOCS_DIR.absolute()}. RAG will be empty.")
        return []

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def init_rag_system():
    """Initialise le système RAG"""
    print("🔍  Loading documents from", DOCS_DIR)
    docs = load_documents()
    
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    if Path(PERSIST_PATH).exists():
        print("📚  Loading existing vector store...")
        vectordb = FAISS.load_local(
            PERSIST_PATH, embeddings, allow_dangerous_deserialization=True
        )
    elif docs:
        print("🔨  Building vector store (this may take a while)...")
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(PERSIST_PATH)
    else:
        # Crée un store vide si aucun doc et aucun store existant (pour éviter de planter)
        print("⚠️  No documents, creating empty vector store placeholder.")
        from langchain_core.documents import Document
        vectordb = FAISS.from_documents([Document(page_content="RTE general information placeholder")], embeddings)
        
    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_DOCUMENTS},
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    
    custom_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_RAG),
        HumanMessagePromptTemplate.from_template("Historique de conversation:\n{chat_history}\n\nContexte:\n{context}\n\nQuestion: {question}")
    ])
    
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )
    
    return rag_chain


# ═══════════════════════════════════════════════════════════════════════
#                      AGENT PRINCIPAL (ROUTER)
# ═══════════════════════════════════════════════════════════════════════

class SmartAssistant:
    """Assistant intelligent avec routage strict RAG/Agent/Conversation"""
    
    # Définition des types de routage
    RAG_ROUTE = "RAG"
    TOOL_ROUTE = "TOOL"
    SIMPLE_ROUTE = "SIMPLE"
    
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
        self.llm_conversational = ChatOpenAI(model=CHAT_MODEL, temperature=0.7)
        
        # Outils de l'Agent (SANS le RAG)
        self.tools = [
            Tool(
                name="calculator",
                description="UNIQUEMENT pour calculs mathématiques explicites avec nombres et opérations.",
                func=calculatrice
            ),
            creer_outil_rte(),
            creer_outil_web_search()
        ]
        
        # Agent avec outils
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_ROUTER),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    
    def determine_route(self, query: str) -> str:
        """Détermine le chemin optimal: RAG, TOOL (Agent) ou SIMPLE (LLM Conversation)"""
        query_lower = query.lower()
        
        # 1. DÉTECTION SIMPLE CONVERSATION
        simple_patterns = [
            r'\b(bonjour|salut|hello|hi|hey)\b',
            r'\b(comment ça va|ça va|ca va|comment vas-tu)\b',
            r'\b(merci|thanks|a bientot|à bientôt|au revoir|bye)\b',
            r'\b(oui|non|ok|d\'accord)\b',
        ]
        if any(re.search(pattern, query_lower) for pattern in simple_patterns):
            return self.SIMPLE_ROUTE
        
        if len(query.split()) <= 3 and not any(
            keyword in query_lower 
            for keyword in ['calcul', 'météo', 'meteo', 'tension', 'tempo', 'rte', 'règle', 'fréquence', 'ajustement']
        ):
            return self.SIMPLE_ROUTE

        # 2. DÉTECTION OUTILS SPÉCIALISÉS (Agent)
        tool_keywords = [
            # Calculatrice
            'calcul', '+', '-', '*', '/', 'fois', 'divisé', 'multiplié',
            # Météo / Web (non-RTE)
            'météo', 'meteo', 'température', 'actualité', 'news', 'temps qu\'il fait', 
            'actualite', 'qui est le president', 'dernières nouvelles',
            # RTE Tension (Tempo)
            'tempo', 'couleur tempo', 'jour tempo', 'rte tension', 
        ]
        if any(keyword in query_lower for keyword in tool_keywords):
            return self.TOOL_ROUTE
        
        # 3. DÉTECTION RAG (par défaut et pour mots-clés RTE)
        internal_keywords = [
            'rte', 'réseau', 'reseau', 'transport', 'règle', 'regle', 'service', 
            'système', 'systeme', 'fréquence', 'ajustement', 'mécanisme', 'mecanisme', 
            'ma', 'nebef', 'effacement', 'marché', 'marche', 'périmètre', 'perimetre', 
            'contribution', 'rpt', 'valorisation', 'document', 'guide', 'procédure', 'procedure'
        ]
        
        if any(keyword in query_lower for keyword in internal_keywords):
            return self.RAG_ROUTE

        # Si la question n'a pas déclenché de ROUTE SIMPLE, TOOL ou de RAG explicite
        # Par exemple, "qui a gagné" (non RTE mais pas dans les mots-clés TOOL)
        # On donne la priorité au RAG pour éviter de chercher sur le web
        return self.RAG_ROUTE
    
    
    # Fonction pour formater le résultat RAG (similaire à l'ancien creer_outil_rag)
    def format_rag_result(self, result: Dict[str, Any]) -> str:
        """Formate la réponse RAG avec les sources"""
        answer = result["answer"]
        sources = result.get("source_documents", [])
        
        if sources:
            fichiers = set()
            for doc in sources:
                source_path = doc.metadata.get('source', 'inconnu')
                fichier = Path(source_path).name if source_path != 'inconnu' else 'inconnu'
                fichiers.add(fichier)
            
            answer += f"\n\n📄 Sources: {len(sources)} chunk(s) trouvé(s) dans {len(fichiers)} fichier(s)"
            if len(fichiers) <= 5:
                answer += f"\n   Fichiers: {', '.join(sorted(fichiers))}"
        
        return answer

    
    async def process(self, user_input: str) -> str:
        """Traite l'entrée utilisateur en appliquant la stratégie de routage stricte (RAG > Agent > Simple)"""
        
        route = self.determine_route(user_input)
        
        print(f"🚦 Route choisie: {route}")  # Ajouté pour le debug
        
        if route == self.SIMPLE_ROUTE:
            # Chemin 1: Conversation simple
            response = await self.llm_conversational.ainvoke(user_input)
            return response.content if hasattr(response, 'content') else str(response)

        elif route == self.RAG_ROUTE:
            # Chemin 2: RAG Direct (pour questions internes)
            try:
                # Utilise ainvoke pour l'appel asynchrone
                result = await self.rag_chain.ainvoke({"question": user_input})
                return self.format_rag_result(result)
            except Exception as e:
                return f"❌ Erreur RAG: {str(e)}"

        elif route == self.TOOL_ROUTE:
            # Chemin 3: Agent (pour outils spécialisés)
            try:
                result = await self.agent_executor.ainvoke({"input": user_input})
                return result["output"]
            except Exception as e:
                return f"❌ Erreur Agent/Outil: {str(e)}"
            
        return "❌ Erreur de routage non gérée."


# ═══════════════════════════════════════════════════════════════════════
#                          BOUCLE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════

async def main():
    """Point d'entrée principal (async)"""
    if "OPENAI_API_KEY" not in os.environ:
        raise SystemExit("👉  export OPENAI_API_KEY=... and run again")
    
    # Initialisation
    rag_chain = init_rag_system()
    assistant = SmartAssistant(rag_chain)
    
    print("\n" + "="*70)
    print("🤖  ASSISTANT RTE INTELLIGENT (ROUTAGE STRICT)")
    print("="*70)
    print("Exemples:")
    print("  • RAG: Qu'est-ce que le mécanisme d'ajustement ?")
    print("  • TOOL: Quelle est la météo à Paris ?")
    print("  • SIMPLE: Bonjour !")
    print("\n(Ctrl-C pour quitter)")
    print("="*70)
    
    try:
        while True:
            user_input = input("\n💬  Vous: ").strip()
            if not user_input:
                continue
            
            print("\n🤖  Assistant:")
            print("-" * 70)
            response = await assistant.process(user_input) 
            print(textwrap.fill(response, width=88))
            print("-" * 70)
            
    except KeyboardInterrupt:
        print("\n\n👋  Au revoir !")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())