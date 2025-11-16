**Conception d'un Assistant Intelligent RAG MultiAgents**

Ce document présente l'architecture et la conception du système d'assistance intelligent spécialisé dans les domaines des mécanismes de flexibilité électriques proposés par le Réseau de Transport d'Électricité (RTE) et des marchés de l'énergie. L'objectif principal est de fournir un outil conversationnel capable de répondre avec précision à partir d'une base de connaissances interne (RAG) tout en étant capable de solliciter des outils spécialisés pour des requêtes externes (calculs, données en temps réel).

**Prérequis**
•	***Clé API OpenAI*** : Nécessaire pour les modèles d'embedding et de chat ( gpt-4o-mini ).
o	Configuration des Clés API
Création d’un fichier un fichier nommé .env à la racine du projet auquels nous y avions ajouté notre clé OpenAI :
•	***Documents de Base*** : Un dossier Data contenant les documents RTE (PDF).

**Architecture Globale**

***RAG sur Documents Internes (Priorité)*** : Répond aux questions sur les règles, mécanismes, et documents internes de RTE (fréquence, ajustement, NEBEF, etc.) . Le pipeline RAG est implémenté via ConversationalRetrievalChain de LangChain
•	Embeddings : OpenAIEmbeddings 
•	Vector Store : FAISS (stockage local et persistant dans faiss_store/).
•	Chargement/Chunking : PyPDFLoader et TextLoader sont utilisés, avec RecursiveCharacterTextSplitter pour découper les documents en morceaux de 1000 caractères avec un chevauchement de 200.
•	Prompt : Le SYSTEM_PROMPT_RAG impose à l'LLM de répondre exclusivement à partir du contexte fourni, garantissant une réponse ancrée.

***Agents Spécialisés***
Les agents sont regroupés et exécutés via un AgentExecutor utilisant le modèle gpt-4o-mini et l'outil create_openai_tools_agent.

•	***Calculator*** permettant l’ Évaluation sécurisée des expressions mathématiques. Implémenter grâce à une fonction Python interne utilisant eval() avec un filtrage Regex strict.
•	***rte_tension*** permettant la consultation des jours Tempo RTE. Implémenter via une Classe JourDeTensionRTE (jourdetension_agent.py) avec un mode démo , faute de ne pas avoir pu obtenu la clé API
•	***web_search*** permettant la recherche web en temps réel. Implémenter via un Wrapper asynchrone autour de DuckDuckGoSearchRun.
•	***Routage Strict*** Assure que le bon outil soit utilisé pour chaque requête, maximisant la précision et la pertinence

**Exécution du Chatbot**
Une fois les dépendances installées et le fichier .env configuré, le RGE et les outils développer , Le chatbot est accessible dans le navigateur web à l'adresse indiquée (http://localhost:8000 ). 
