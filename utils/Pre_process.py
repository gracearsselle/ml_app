# Dans Utils/Pre_process.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# LES NOUVEAUX CHEMINS 2026 :
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

import time

load_dotenv()

# --- CONFIGURATION CENTRALE ---

def get_embedding_model():
    """Initialise le modèle d'embedding Gemini stable (v2026)."""
    # Utilisation du nom stable officiel sans préfixe selon la doc 2026
    base_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", 
        task_type="retrieval_document"
    )
    
    # On utilise toujours le stockage local classic
    store = LocalFileStore("./cache_embeddings/")
    
    return CacheBackedEmbeddings.from_bytes_store(
        base_embeddings, 
        store,      
        namespace="v2" # On change le namespace pour isoler ce nouveau modèle
    )

# --- FONCTIONS DE TRAITEMENT ---

def load_pdf(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")
    loader = PyPDFLoader(file_path)
    return loader.load()

def chunking(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
        add_start_index=True 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Nombre total de morceaux (chunks) créés : {len(chunks)}")
    return chunks

def get_or_create_vector_store(chunks=None):
    embeddings = get_embedding_model()
    persist_db = "./chroma_db"

    # 1. On charge toujours la base existante (ou on en crée une vide)
    # Cela évite de réinitialiser la base à chaque appel
    vector_store = Chroma(
        persist_directory=persist_db,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )

    if chunks:
        # Batch size de 5 est beaucoup plus sûr pour l'offre gratuite
        batch_size = 5 
        print(f"Début de l'indexation de {len(chunks)} chunks par lots de {batch_size}...")
        
        # On ajoute les documents par lots
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            try:
                vector_store.add_documents(batch)
                print(f"✅ Ajout lot {i//batch_size + 1} ({min(i + batch_size, len(chunks))}/{len(chunks)})")
                
                # Pause courte obligatoire pour le Free Tier
                time.sleep(2) 
                
            except Exception as e:
                # Si une erreur survient, on affiche le détail
                print(f"❌ Erreur lors de l'ajout du lot {i}: {e}")
                # Selon la gravité, on peut arrêter le processus
                if "429" in str(e):
                    print("⚠️ Quota atteint. Veuillez attendre quelques minutes avant de relancer.")
                    break 
        
        print("✅ Indexation terminée.")
    else:
        print("📂 Base de données chargée depuis le disque.")
    
    return vector_store
    embeddings = get_embedding_model()
    persist_db = "./chroma_db"

    if chunks:
        # On réduit un peu la taille du lot pour plus de sécurité
        batch_size = 40 
        
        print(f"Début de l'indexation par lots de {batch_size}...")
        
        # 1. Premier lot
        vector_store = Chroma.from_documents(
            documents=chunks[:batch_size],
            embedding=embeddings,
            persist_directory=persist_db,
            collection_metadata={"hnsw:space": "cosine"}
        )
        print(f"Indexation : {min(batch_size, len(chunks))}/{len(chunks)} terminés.")
        
        # 2. Ajout des lots suivants avec une pause
        if len(chunks) > batch_size:
            for i in range(batch_size, len(chunks), batch_size):
                # Pause de 10 secondes entre chaque lot pour réinitialiser le quota RPM
                time.sleep(10) 
                
                batch = chunks[i : i + batch_size]
                vector_store.add_documents(batch)
                print(f"Indexation : {min(i + len(batch), len(chunks))}/{len(chunks)} terminés...")
        
        print("✅ Base de données vectorielle créée avec succès.")
    else:
        vector_store = Chroma(
            persist_directory=persist_db,
            embedding_function=embeddings
        )
        print("📂 Base de données chargée depuis le disque.")
    
    return vector_store