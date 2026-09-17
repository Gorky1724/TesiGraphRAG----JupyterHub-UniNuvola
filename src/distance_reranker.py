import math
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.sidecar_manager import SidecarManager
from src.config import (
    HARD_FILTERING_THRESHOLD,
    RERANKING_TOP_N,
)

class DistanceReranker:
    def __init__(self, sidecar_manager: SidecarManager, hard_filter_threshold = HARD_FILTERING_THRESHOLD):
        """
        :param sidecar_manager: Istanza di SidecarManager per accedere al file JSON.
        :param hard_filter_threshold: Soglia di distance_factor oltre la quale un chunk viene considerato escluso.
        """
        self.sidecar = sidecar_manager
        self.hard_filter_threshold = hard_filter_threshold

    def _extract_chunk_id(self, item: Any) -> str:
        """
        Estrae l'ID univoco del chunk (funziona con oggetti LangChain, dict o tuple).
        L'ID, se non nativamente presente nei metadati, è restituito nel formato: DOCID_chunk_CHUNKINDEX
        """
        doc = item[0] if isinstance(item, (tuple, list)) else item

        if isinstance(doc, dict):
            if "chunk_id" in doc:
                return str(doc["chunk_id"])
            metadata = doc.get("metadata", doc)
            doc_id = metadata.get("doc_id", metadata.get("source", "doc"))
            chunk_idx = metadata.get("chunk_index", 0)
            return f"{doc_id}_chunk_{chunk_idx}"

        if hasattr(doc, "metadata"):
            doc_id = doc.metadata.get("doc_id", doc.metadata.get("source", "doc"))
            chunk_idx = doc.metadata.get("chunk_index", 0)
            return f"{doc_id}_chunk_{chunk_idx}"

        return str(doc)

    def rerank(self, candidates: List[Any], top_n: int = RERANKING_TOP_N) -> List[Dict[str, Any]]:
        """
        Ricalcola i punteggi basandosi sui distance_factor salvati nel sidecar.

        :param candidates: Lista di candidate chunk con punteggio iniziale [(doc, score), ...]
        :param top_n: Numero massimo di chunk da selezionare dopo il re-ranking per Ollama.
        :return: Lista dei primi Top-N chunk sopravvissuti, ordinati per score finale.
        """
        sidecar_data = self.sidecar.load_data()
        pairwise_deltas = sidecar_data.get("pairwise_deltas", {})

        parsed_candidates = []
        for item in candidates:
            if isinstance(item, (tuple, list)):
                doc, initial_score = item[0], float(item[1])
            elif isinstance(item, dict):
                doc = item
                initial_score = float(item.get("score", 0.5))
            else:
                doc = item
                initial_score = 0.5

            chunk_id = self._extract_chunk_id(doc)
            parsed_candidates.append({
                "raw_doc": doc,
                "chunk_id": chunk_id,
                "initial_score": initial_score,
                "final_score": initial_score,
                "applied_penalties": [],
                "excluded": False
            })

        # Ricalcolo Score tramite Pairwise Deltas
        for c1 in parsed_candidates:
            id1 = c1["chunk_id"]
            max_penalty = 1.0

            for c2 in parsed_candidates:
                id2 = c2["chunk_id"]
                if id1 == id2:
                    continue

                # Chiave simmetrica A_AND_B
                pair_key = "_AND_".join(sorted([str(id1), str(id2)]))

                if pair_key in pairwise_deltas:
                    delta_info = pairwise_deltas[pair_key]
                    factor = float(delta_info.get("distance_factor", 1.0))

                    if factor > max_penalty:
                        max_penalty = factor

                    c1["applied_penalties"].append({
                        "paired_with": id2,
                        "distance_factor": factor
                    })

            # Hard Filtering o Soft Reweighting
            if max_penalty >= self.hard_filter_threshold:
                c1["excluded"] = True
            else:
                c1["final_score"] = c1["initial_score"] / max_penalty

        # Filtraggio ed estrazione dei Top-N
        valid_candidates = [c for c in parsed_candidates if not c["excluded"]]
        valid_candidates.sort(key=lambda x: x["final_score"], reverse=True)

        return valid_candidates[:top_n]
