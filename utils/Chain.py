from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(vector_store):
    # 1. Initialiser Gemini (Température basse pour plus de précision académique)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)

    # 2. Définir le Template de Prompt
    # C'est ici que tu donnes ses instructions à l'IA
    template = """
    Tu es un assistant IA expert en analyse et synthèse de documents techniques.

    RÈGLES ABSOLUES :
    1. Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. Aucune connaissance externe n'est autorisée.
    2. Si la réponse est absente ou insuffisante dans le contexte, réponds EXACTEMENT : "Je ne trouve pas cette information dans le document fourni."
    3. Ne complète jamais une réponse par des suppositions ou des connaissances générales.

    STYLE DE RÉPONSE :
    - Sois direct et structuré : commence par une réponse courte, puis développe si nécessaire.
    - Utilise des listes à puces pour les énumérations.
    - Cite le passage pertinent du contexte entre guillemets si c'est utile.
    - Réponds dans la même langue que la question posée.
    
    CONTEXTE :
    {context}
    
    QUESTION :
    {question}
    
    RÉPONSE :
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Configurer le Retriever (Récupération des 3 meilleurs chunks)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. Construire la chaîne LCEL
    # Le flux : Question -> Recherche Contexte -> Remplissage Prompt -> LLM -> Texte
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain