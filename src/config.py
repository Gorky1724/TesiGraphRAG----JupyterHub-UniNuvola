import os

# ======================================================================
# SELEZIONE MODELLO LLM (solo UNA delle opzioni)
# ======================================================================

LLM_MODEL = "llama3.2"       # Llama 3.2 (3B) -> Sviluppo fluido e bilanciato su CPU (Default)
#LLM_MODEL = "llama3.2:1b"  # Llama 3.2 (1B) -> Ultra-veloce per coding e debug rapido
#LLM_MODEL = "llama3.1"     # Llama 3.1 (8B) -> Massima qualità per i test finali


# ======================================================================
# SELEZIONE MODELLO EMBEDDING (solo UNA delle opzioni)
# ======================================================================

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5" # Nomic v1.5 gestito direttamente da FastEmbed (768 dim)
#EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Alternate lightweight option (384 dim)

# ======================================================================
# CONFIGURAZIONE SERVIZI BACKEND E PARAMETRI CPU
# ======================================================================

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "2h"
DEFAULT_NUM_THREAD = 4

# ======================================================================
# SELEZIONE COLLEZIONE QDRANT DA UTILIZZARE (solo UNA delle opzioni)
# ======================================================================

COLLECTION_NAME = "ds1_graphrag_chunks"
#da cambiare se si dovesse utilizzare qualche altra collezione

# ======================================================================
# PARAMETRI GRAFO E VISUALIZZAZIONE D3
# ======================================================================
SIMILARITY_THRESHOLD = 0.45  # Soglia minima Cosine Similarity per tracciare un arco tra 2 chunk -- consigliato [0.45, 0.55]
BASE_GRAPH_DISTANCE = 300   # Lunghezza base in pixel (per similarità 1.0)

# ======================================================================
# PARAMETRI RETRIEVAL E RERANKING
# ======================================================================
RETRIEVAL_TOP_K = 7  # Numero chunk retrieved da Qdrant con il vector_similarity_search
RERANKING_TOP_N = 4   # Numero chunk restituiti dopo il reranking


