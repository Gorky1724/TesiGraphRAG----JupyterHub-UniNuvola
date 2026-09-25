import os

# ======================================================================
# SELEZIONE MODELLO LLM (solo UNA delle opzioni)
# ======================================================================

LLM_MODEL = "llama3.2"       # Llama 3.2 (3B) -> Sviluppo fluido e bilanciato su CPU (Default)
#LLM_MODEL = "llama3.2:1b"  # Llama 3.2 (1B) -> Ultra-veloce per coding e debug rapido
#LLM_MODEL = "llama3.1"     # Llama 3.1 (8B) -> Massima qualità per i test finali


# ======================================================================
# SELEZIONE MODELLO EMBEDDING (solo UNA delle opzioni).
#  Il cambio di embedding richiede la re-Ingestion dei DataSet (per poter essere manipolato con quel certo modello).
#  In data/collection_registry.json sono salvate le info di ingestione e associazione collection-embedding_model
# ======================================================================

# 1024 dim
#EMBEDDING_MODEL = "intfloat/multilingual-e5-large" # Ottimo embedding, 2 o 3 volte più lento di quelli a 768 dim

# 768 dim
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5" # Nomic v1.5 gestito direttamente da FastEmbed
#EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2" # Nomic FastEmbed con supporto multilingua

# 364 dim
#EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # Opzione lightweight
#EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # Nomic FastEmbed con supporto multilingua - lightweight

# ======================================================================
# CONFIGURAZIONE SERVIZI BACKEND E PARAMETRI CPU
# ======================================================================

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
DEFAULT_KEEP_ALIVE = "2h"
DEFAULT_NUM_THREAD = 4

# ======================================================================
# SELEZIONE COLLEZIONE QDRANT DA UTILIZZARE (solo UNA delle opzioni)
#  Scommentare solo la collezione che si sta utilizzando
# ======================================================================

COLLECTION_NAME = "unilanguage_armstrong"

#COLLECTION_NAME = "ds1_unilingual_base"
#COLLECTION_NAME = "ds1_multilingual_base"

#COLLECTION_NAME = "ds1_graphrag_chunks"

# ======================================================================
# PARAMETRI GRAFO E VISUALIZZAZIONE D3
# ======================================================================
#""" UNILINGUA
SIMILARITY_THRESHOLD = 0.80  # Soglia minima Cosine Similarity per tracciare un arco tra 2 chunk -- consigliato [0.45, 0.55]
#"""

""" MULTILINGUA
SIMILARITY_THRESHOLD = 0.60
"""

BASE_GRAPH_DISTANCE = 300   # Lunghezza base in pixel (per similarità 1.0)

# ======================================================================
# PARAMETRI GENERALI RETRIEVAL E RERANKING
# ======================================================================
RETRIEVAL_TOP_K = 7  # Numero chunk retrieved da Qdrant con il vector_similarity_search
RERANKING_TOP_N = 4   # Numero chunk restituiti dopo il reranking

# ======================================================================
# CONFIGURAZIONE TAG RERANKER e ASSIGNER
# ======================================================================

TAG_BOOST_FACTOR = 0.20   # Boost del +20% per ogni tag in comune tra query e chunk
TAG_MALUS_FACTOR = 0.00   # Malus dello 0% (nessuna penalizzazione per chunk privi di match)
# TAG_MALUS_FACTOR = 0.05 # Opzione alternativa per malus leggerissimo (-5%) se desiderato
MAX_BOOST = 0.50 # Limita il boost per evitare sbilanciamento se combaciano troppi chunk

""" MULTILANGUAGE
TAG_ASSIGN_THRESHOLD = 0.32
TAG_WEIGHT_COSINE = 0.75
TAG_WEIGHT_OVERLAP = 0.25
"""

# Armstrong ds test values
TAG_ASSIGN_THRESHOLD = 0.5
TAG_WEIGHT_COSINE = 0.75
TAG_WEIGHT_OVERLAP = 0.25


""" UNILANGUAGE
TAG_ASSIGN_THRESHOLD = 0.42
TAG_WEIGHT_COSINE = 0.75
TAG_WEIGHT_OVERLAP = 0.25
"""

""" 2° test values - troppe assegnazioni
TAG_ASSIGN_THRESHOLD = 0.35
TAG_WEIGHT_COSINE = 0.85
TAG_WEIGHT_OVERLAP = 0.15
"""

""" 1° test values - troppe poche assegnazioni
TAG_ASSIGN_THRESHOLD = 0.50
TAG_WEIGHT_COSINE = 0.70
TAG_WEIGHT_OVERLAP = 0.30
"""