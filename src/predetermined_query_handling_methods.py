import sys
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union, List
import numpy as np

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import PREDET_QUERY_THRESHOLD
from src.sidecar_manager import SidecarManager

logger = logging.getLogger(__name__)

def compute_cosine_similarity(vec1: Union[List[float], np.ndarray], vec2: Union[List[float], np.ndarray]) -> float:
    """Calcola la similarità coseno tra due vettori."""
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))

def verify_if_pred_query(
    query_text: str,
    sidecar_manager: SidecarManager,
    query_vector: Optional[Union[List[float], np.ndarray]] = None,
    embedder: Optional[Any] = None,
    threshold: float = PREDET_QUERY_THRESHOLD
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Verifica se una query utente corrisponde a una query pre-determinata salvata nel Sidecar.

    Returns:
        Tuple[Optional[str], Optional[Dict[str, Any]]]:
            Una tupla (predq_id, query_data) dove:
            - predq_id (str || None): L'ID univoco della query pre-determinata riconosciuta (es. 'predq_101'),
              oppure None se non viene trovata alcuna corrispondenza.
            - query_data (Dict[str, Any] || None): Il dizionario completo estratto dal file Sidecar
              contenente i metadati e le modifiche legate alla query, oppure None se non trovata.

            In caso di match (predq_id != None), 'query_data' include le seguenti chiavi:
                > "query_text" (str): Il testo originale della query memorizzata.
                > "adjacencies" (Dict[str, Dict[str, str]]): Le adiacenze salvate tra il nodo query
                  e i chunk, con le relative classi di distanza modificate 
                  (es. {"chunk_12": {"distance_class": "AVVICINATI"}}).
                > "graphically_assigned_tags" (List[str]): Lista dei tag grafici/manuali 
                  assegnati dall'utente a questo specifico nodo query.
                > "vector" / "embedding" (List[float] | None): Il vettore di embedding 
                  della query memorizzato nel sidecar.
    """
    if not query_text or not isinstance(query_text, str):
        return None, None

    pred_queries = sidecar_manager.get_predetermined_queries()
    if not pred_queries:
        return None, None

    norm_input_text = query_text.strip().lower()

    # Tentativo match testuale esatto
    for pred_id, data in pred_queries.items():
        stored_text = data.get("query_text", "") or data.get("text", "")
        if stored_text and stored_text.strip().lower() == norm_input_text:
            logger.info(f"<<| Match testuale esatto per Query Pre-Determinata: [{pred_id}] |>>")
            return pred_id, data

    # Tentativo con cos_sim
    if query_vector is None and embedder is not None:
        try:
            if hasattr(embedder, "embed_query"):
                query_vector = embedder.embed_query(query_text)
            elif hasattr(embedder, "encode"):
                query_vector = embedder.encode(query_text)
            elif callable(embedder):
                query_vector = embedder(query_text)
        except Exception as e:
            logger.warning(f"<[WARNING]> Errore nella generazione dell'embedding di query: {e}")
            query_vector = None

    if query_vector is not None:
        best_pred_id = None
        best_data = None
        max_sim = -1.0

        for pred_id, data in pred_queries.items():
            stored_vector = data.get("vector") or data.get("embedding")
            if stored_vector is not None:
                sim = compute_cosine_similarity(query_vector, stored_vector)
                if sim > max_sim:
                    max_sim = sim
                    best_pred_id = pred_id
                    best_data = data
            #logger.info(f"<[DEBUG]> sim({pred_id}, query): {sim}")

        if max_sim >= threshold and best_pred_id is not None:
            logger.info(f"<<| Match vettoriale trovato per Query Pre-Determinata [{best_pred_id}] con similarità {max_sim:.4f} (>= {threshold:.2f}) |>>")
            return best_pred_id, best_data

    logger.info(f"<<| Nessun match vettoriale con Query Pre-Determinate trovat per la query: < {query_text} >  |>>")
    return None, None

def add_predetermined_query(
    query_text: str,
    sidecar_manager: SidecarManager,
    query_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    embedder: Optional[Any] = None
) -> str:
    """
    Aggiunge una query pre-determinata salvando testo ed embedding nel Sidecar.
    """
    if not query_text or not isinstance(query_text, str):
        raise ValueError("!!!> query_text non valido o vuoto.")

    norm_input_text = query_text.strip().lower()
    data = sidecar_manager.load_data()
    pred_queries = data.setdefault("predetermined_queries", {})

    # Controllo duplicati
    for existing_id, existing_data in pred_queries.items():
        stored_text = existing_data.get("query_text", "") or existing_data.get("text", "")
        if stored_text and stored_text.strip().lower() == norm_input_text:
            logger.info(f"i>> Query già presente nel Sidecar con ID '[{existing_id}]': '{query_text}'")
            return existing_id

    # Per garantire prefisso 'predq_'
    if query_id:
        q_id_str = str(query_id).strip()
        final_id = q_id_str if q_id_str.startswith("predq_") else f"predq_{q_id_str}"
    else:
        existing_indices = [
            int(k.replace("predq_", ""))
            for k in pred_queries.keys()
            if k.startswith("predq_") and k.replace("predq_", "").isdigit()
        ]
        next_idx = max(existing_indices, default=0) + 1
        final_id = f"predq_{next_idx}"

    if final_id in pred_queries:
        logger.warning(f"!>> L'ID '[{final_id}]' è già presente nel Sidecar.")
        return final_id

    # Calcolo embedding
    vector = None
    if embedder is not None:
        try:
            if hasattr(embedder, "embed_query"):
                vector = embedder.embed_query(query_text)
            elif hasattr(embedder, "encode"):
                vector = embedder.encode(query_text)
            elif callable(embedder):
                vector = embedder(query_text)
        except Exception as e:
            logger.warning(f"!!!> Impossibile generare l'embedding per '{query_text}': {e}")

    if vector is not None and hasattr(vector, "tolist"):
        vector = vector.tolist()

    # Inizializzazione struttura 
    pred_queries[final_id] = {
        "query_text": query_text,
        "vector": vector,
        "graphically_assigned_tags": [],
        "adjacencies": {}
    }
    sidecar_manager.save_data(data)

    # Assegnazione tag
    if tags:
        sidecar_manager.add_tag_override_for_chunk(final_id, tags)

    logger.info(f"<<| Query Pre-Determinata registrata: ID=[{final_id}] |>>")
    return final_id


### 3. Conversione Batch da File (es. eval_queries.json)
def add_predetermined_queries_from_file(
    queries_path: Union[str, Path],
    sidecar_manager: SidecarManager,
    embedder: Optional[Any] = None
) -> List[str]:
    """
    Legge un file JSON di query (es. eval_queries.json) e le converte tutte in query pre-determinate.
    """
    path = Path(queries_path).expanduser().resolve()
    if not path.exists():
        logger.error(f"!!!> File di query non trovato: {path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        eval_queries = json.load(f)

    if not isinstance(eval_queries, list):
        logger.error("!!!> Formato JSON non valido: ci si attende una lista.")
        return []

    added_ids = []
    for q_entry in eval_queries:
        if not isinstance(q_entry, dict):
            continue

        query_text = q_entry.get("query") or q_entry.get("query_text") or q_entry.get("text")
        query_id = q_entry.get("id") or q_entry.get("query_id")

        expected_tag = q_entry.get("expected_tag")
        tags = [expected_tag] if expected_tag else []

        if query_text:
            pred_id = add_predetermined_query(
                query_text=query_text,
                sidecar_manager=sidecar_manager,
                query_id=query_id,
                tags=tags,
                embedder=embedder
            )
            added_ids.append(pred_id)

    logger.info(f"<<| Convertite {len(added_ids)} query da '{path.name}' |>>")
    return added_ids

