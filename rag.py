#!/usr/bin/env python
"""
RAG.py
─────────────────────────────
RAG avec LCEL et agents spécialisés avec routage intelligent

Capacités:
▪️ RAG sur documents internes (PRIORITÉ ABSOLUE si sujets internes)
▪️ Calculatrice pour calculs mathématiques
▪️ Recherche web DuckDuckGo (externe uniquement)
▪️ Jours de tension RTE (Tempo)
▪️ Conversation simple sans outils
"""

from pathlib import Path
import os
import textwrap
import re
from typing import Dict, Any
from datetime import datetime

# ── LangChain core avec LCEL ─────────────────────────────────────────
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# ── Agent RTE personnalisé ────────────────────────
from jourdetension import JourDeTensionRTE, format_reponse_rte
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("openai_key", "")

# ═════════════════════════════════════════════════════
#                          CONFIGURATION
# ═════════════════════════════════════════════════════

DOCS_DIR = Path("C:\\Users\\j_aka\\Desktop\\AII\\Data")
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_PATH = "faiss_store"
TOP_K_DOCUMENTS = 10

SYSTEM_PROMPT_RAG = """Tu es un expert et tuteur précis spécialisé dans RTE (Réseau de Transport d'Électricité).

RÈGLES STRICTES:
1. Base ta réponse EXCLUSIVEMENT sur le contexte fourni ci-dessous
2. Si tu trouves les informations pertinentes, synthétise une réponse claire et complète en français
3. Si le contexte est insuffisant, réponds: "Je n'ai pas trouvé cette information dans les documents fournis."
4. Cite les sources quand c'est pertinent
5. Sois concis mais complet

Contexte:
{context}

Historique de conversation:
{chat_history}

Question actuelle: {question}"""

SYSTEM_PROMPT_ROUTER = """Tu es un assistant intelligent spécialisé dans les règles et systèmes RTE.

RÈGLE ABSOLUE: Une fois qu'un outil a renvoyé un résultat, tu DOIS l'utiliser pour formuler ta réponse. 
Ne dis JAMAIS "Je n'ai pas pu obtenir..." si un outil t'a fourni des données.

RÈGLES DE ROUTAGE:
1. calculator: UNIQUEMENT pour calculs mathématiques explicites
2. rte_tension: UNIQUEMENT pour les jours Tempo (Rouge/Blanc/Bleu)
3. web_search: UNIQUEMENT pour informations externes NON liées à RTE
   → Exemples: météo, actualités, prix spot, CEE, heure actuelle

Choisis l'outil le plus pertinent pour la requête utilisateur."""

# ═════════════════════════════════════════════════════
#                          OUTILS / AGENTS
# ═════════════════════════════════════════════════════

def calculatrice(expression: str) -> str:
    try:
        expression = expression.strip().replace(',', '.')
        if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
            return "❌ Expression invalide. Utilisez uniquement des nombres et opérateurs (+, -, *, /, ())"
        resultat = eval(expression, {"__builtins__": {}}, {})
        return f"✓ Résultat: {resultat}"
    except Exception as e:
        return f"❌ Erreur de calcul: {str(e)}"

def creer_outil_web_search():
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=3)
    search = DuckDuckGoSearchRun(api_wrapper=wrapper)
    async def run_search_async(query: str) -> str:
        import asyncio
        return await asyncio.to_thread(search.run, query)
    return Tool.from_function(
        name="web_search",
        description=(
            "Recherche web pour informations EXTERNES non liées à RTE. "
            "Ex: météo, heure, actualités, prix spot, CEE."
        ),
        func=run_search_async,
        coroutine=run_search_async
    )

def creer_outil_rte():
    agent_rte = JourDeTensionRTE()
    async def run_rte_async(query: str) -> str:
        import asyncio
        from datetime import timedelta
        query_lower = query.lower()
        def get_rte_data():
            if any(word in query_lower for word in ["semaine", "prochains jours", "prévision", "7 jours"]):
                return agent_rte.get_prevision_semaine()
            elif "demain" in query_lower:
                from datetime import datetime
                date_demain = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                return agent_rte.get_jour_tension(date_demain)
            else:
                return agent_rte.get_jour_tension()
        data = await asyncio.to_thread(get_rte_data)
        return format_reponse_rte(data)
    return Tool.from_function(
        name="rte_tension",
        description="Calendrier Tempo EDF (jours Rouge/Blanc/Bleu).",
        func=run_rte_async,
        coroutine=run_rte_async
    )

# ═════════════════════════════════════════════════════
#                     RAG avec LCEL
# ═════════════════════════════════════════════════════

def load_documents():
    loaders = []
    for path in DOCS_DIR.rglob("*.txt"):
        loaders.append(TextLoader(str(path), encoding='utf-8'))
    for path in DOCS_DIR.rglob("*.pdf"):
        loaders.append(PyPDFLoader(str(path)))
    docs = []
    for loader in loaders:
        try: docs.extend(loader.load())
        except Exception as e: print(f"⚠️  Erreur chargement {loader}: {e}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def format_docs(docs):
    return "\n\n".join(f"[Source: {doc.metadata.get('source','inconnu')}]\n{doc.page_content}" for doc in docs)

def init_rag_system():
    print("📚 Chargement des documents depuis", DOCS_DIR)
    docs = load_documents()
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    if Path(PERSIST_PATH).exists():
        vectordb = FAISS.load_local(PERSIST_PATH, embeddings, allow_dangerous_deserialization=True)
    elif docs:
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(PERSIST_PATH)
    else:
        from langchain_core.documents import Document
        vectordb = FAISS.from_documents([Document(page_content="RTE general info placeholder")], embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K_DOCUMENTS})
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.3)
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_RAG)
    # RAG chain : conversion docs -> texte se fait après récupération
    rag_chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough(),
            "chat_history": lambda x: x.get("chat_history", "")
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# ═════════════════════════════════════════════════════
#                      ASSISTANT
# ═════════════════════════════════════════════════════

class SmartAssistant:
    RAG_ROUTE = "RAG"
    TOOL_ROUTE = "TOOL"
    SIMPLE_ROUTE = "SIMPLE"
    
    def __init__(self, rag_chain, retriever):
        self.rag_chain = rag_chain
        self.retriever = retriever
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.5)
        self.llm_conversational = ChatOpenAI(model=CHAT_MODEL, temperature=0.7)
        self.chat_history = InMemoryChatMessageHistory()
        # Outils
        self.tools = [
            Tool(name="calculator", description="Calculatrice", func=calculatrice),
            creer_outil_rte(),
            creer_outil_web_search()
        ]
        from langchain.agents import initialize_agent, AgentType
        self.agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def determine_route(self, query: str) -> str:
        q = query.lower()
        if any(re.search(p, q) for p in [r'\b(bonjour|salut|hello|hi|hey)\b', r'\b(comment ça va|ça va|ca va)\b', r'\b(merci|thanks|au revoir|bye)\b', r'\b(oui|non|ok|d\'accord)\b']):
            return self.SIMPLE_ROUTE
        if any(k in q for k in ['calcul','+','-','*','/','fois','divisé','météo','meteo','température','actualité','news','heure','prix spot','cee','tempo','jour tempo']):
            return self.TOOL_ROUTE
        if any(k in q for k in ['rte','réseau','reseau','transport','règle','regle','service','système','systeme','fréquence','ajustement','mécanisme','mecanisme','ma','nebef','effacement','marché','marche','contribution','valorisation']):
            return self.RAG_ROUTE
        return self.RAG_ROUTE
    
    def format_chat_history(self) -> str:
        msgs = self.chat_history.messages[-6:]
        return "\n".join(f"Utilisateur: {m.content}" if isinstance(m, HumanMessage) else f"Assistant: {m.content}" for m in msgs)
    
    async def process(self, user_input: str) -> Dict[str, Any]:
        route = self.determine_route(user_input)
        print(f"🚦 Route: {route}")
        try:
            if route == self.SIMPLE_ROUTE:
                r = await self.llm_conversational.ainvoke(user_input)
                answer = getattr(r, "content", str(r))
                sources = []
            elif route == self.RAG_ROUTE:
                docs = await self.retriever.ainvoke(user_input)  # Récupère les Document
                docs_text = format_docs(docs)
                answer = await self.rag_chain.ainvoke({
                    "question": user_input,
                    "context": docs_text,
                    "chat_history": self.format_chat_history()
                })
                sources = list({Path(doc.metadata.get('source','inconnu')).name for doc in docs})
            else:  # TOOL_ROUTE
                r = await self.agent_executor.ainvoke({"input": user_input})
                answer = r["output"]
                sources = []
            self.chat_history.add_user_message(user_input)
            self.chat_history.add_ai_message(answer)
            return {"answer": answer, "route": route, "sources": sources, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            return {"answer": f"❌ Erreur ({route}): {e}", "route": route, "sources": [], "timestamp": datetime.now().isoformat()}

# ═════════════════════════════════════════════════════
#                          BOUCLE PRINCIPALE
# ═════════════════════════════════════════════════════

async def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("❌ OPENAI_API_KEY non définie dans .env")
    print("\n" + "="*70)
    print("🤖  ASSISTANT RTE INTELLIGENT (LCEL)")
    print("="*70)
    rag_chain, retriever = init_rag_system()
    assistant = SmartAssistant(rag_chain, retriever)
    try:
        while True:
            user_input = input("\n💬  Vous: ").strip()
            if not user_input: continue
            if user_input.lower() in ['quit','exit','bye']:
                print("\n👋 Au revoir!")
                break
            print("\n🤖  Assistant:")
            print("-"*70)
            result = await assistant.process(user_input)
            print(textwrap.fill(result["answer"], width=88))
            if result["sources"]: print(f"\n📄 Sources: {', '.join(result['sources'])}")
            print("-"*70)
    except KeyboardInterrupt:
        print("\n👋 Au revoir!")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
