import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import TAG_BOOST_FACTOR, TAG_MALUS_FACTOR, MAX_BOOST, RERANKING_TOP_N
from src.sidecar_manager import SidecarManager
from src.tag_assigner import TagAssigner

class TagReranker:
    """
    Modulo di Reranking che altera lo score di similarità dei chunk candidati
    in base alla corrispondenza tra i tag individuati nella query e i tag dei chunk candidati.

    ---------------------------------------------------------------------------
    Nota di utilizzo consigliata per il Retrieval (Qdrant):
    Si raccomanda di recuperare i chunk candidati da Qdrant tramite:
        response = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedded,
            limit=RETRIEVAL_TOP_K,
            with_vectors=True,
            with_payload=True
        )
    L'uso di query_points garantisce l'accesso diretto a hit.score (score CS di base),
    hit.payload (testo e metadati) e hit.vector.
    ---------------------------------------------------------------------------
    """

    def __init__(
        self,
        sidecar_manager: Optional[SidecarManager] = None,
        tag_assigner: Optional[TagAssigner] = None,
        boost_factor: float = TAG_BOOST_FACTOR,
        malus_factor: float = TAG_MALUS_FACTOR,
        max_boost: float = MAX_BOOST
    ):
        self.sidecar_manager = sidecar_manager or SidecarManager()
        self.tag_assigner = tag_assigner or TagAssigner()
        self.boost_factor = boost_factor
        self.malus_factor = malus_factor
        self.max_boost = max_boost

    def rerank(
        self,
        query_text: str,
        retrieved_points: List[Any],
        query_vector: Optional[Union[List[float], np.ndarray]] = None,
        top_n: int=RERANKING_TOP_N
    ) -> List[Dict[str, Any]]:
        """
        Riorordina i punti estratti da Qdrant applicando il moltiplicatore weight_factor.
        """
        if not retrieved_points:
            return []

        # Recupero tag globali attivi
        sidecar_data = self.sidecar_manager.load_data()
        global_tags_dict = sidecar_data.get("global_tags", {})
        candidate_tags = list(global_tags_dict.keys())

        # Assegnazione dei tag alla query e recupero tag dei chunk
        query_tags = []
        if candidate_tags:
            query_tags = self.tag_assigner.assign_tags(
                text=query_text, candidate_tags=candidate_tags, vector=query_vector
            )

        tag_overrides = sidecar_data.get("tag_overrides", {})

        # Analisi chunk recuperati e calcolo punteggio finale
        reranked_results = []
        for idx, hit in enumerate(retrieved_points):
            payload = getattr(hit, "payload", {}) or {}
            initial_score = max(0.0, float(hit.score)) # Per sicurezza, così qualunque score non sarà negativo

            # Estrazione ID
            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
            doc_id = payload.get("doc_id") or meta.get("doc_id") or "doc"
            raw_idx = payload.get("chunk_index") if payload.get("chunk_index") is not None else meta.get("chunk_index")
            chunk_idx = raw_idx if raw_idx is not None else idx_global
            chunk_id = payload.get("chunk_id") or meta.get("chunk_id") or f"{doc_id}_chunk_{chunk_idx}"

            # TAG del chunk in Analisi
            chunk_info = tag_overrides.get(chunk_id, {})
            chunk_tags = chunk_info.get("user_tags", [])

            # Identificazione n° di tag comuni tra query e chunk
            matched_tags = list(set(query_tags).intersection(set(chunk_tags)))

            # Calcolo weight_factor
            if query_tags and chunk_tags:
                if matched_tags:
                    # MATCH
                    raw_weight= 1.0 + (self.boost_factor * len(matched_tags))
                    weight_factor = min(raw_weight, 1.0 + self.max_boost)
                else:
                    # MISMATCH
                    weight_factor = max(0.1, 1.0 - self.malus_factor)
            else:
                # NIENTE TAG sulla query
                weight_factor = 1.0

            raw_final_score = initial_score * weight_factor

            # Oggetto di output
            result_item = {
                "chunk_id": chunk_id,
                "payload": payload,
                "initial_score": round(initial_score, 4),
                "final_score": round(raw_final_score, 4),
                "_raw_final_score": raw_final_score, # temporanea per il sorting
                "weight_factor": round(weight_factor, 2),
                "query_tags": query_tags,
                "chunk_tags": chunk_tags,
                "matched_tags": matched_tags,
                "vector": getattr(hit, "vector", None),
            }

            reranked_results.append(result_item)

        # Ordinamento decrescente
        reranked_results.sort(key=lambda x: x["_raw_final_score"], reverse=True)

        # Rimozione chiave temporanea
        for item in reranked_results:
            del item["_raw_final_score"]

        # Ritorno rei TOP_N dopo il rerank
        if top_n is not None and top_n > 0:
            return reranked_results[:top_n]

        return reranked_results
