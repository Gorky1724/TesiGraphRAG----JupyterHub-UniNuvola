import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from qdrant_client import QdrantClient

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.sidecar_manager import SidecarManager
from src.distance_retriever import DistanceRetriever, build_query_tags
from src.config import (
    COLLECTION_NAME,
    TAG_BOOST_ASSIGNED_CONV, TAG_BOOST_AUTO, TAG_MAX_BOOST,
    RETRIEVAL_TOP_K, RERANKING_TOP_N
)

logger = logging.getLogger(__name__)

def _extract_chunk_id(hit: Any, fallback_index: Optional[int] = None) -> str:
    """Estrae l'ID del chunk da un oggetto Qdrant (ScoredPoint o Record)."""
    payload = getattr(hit, "payload", {}) or {}
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload

    chunk_id = payload.get("chunk_id") or meta.get("chunk_id")
    if not chunk_id:
        doc_id = payload.get("doc_id") or meta.get("doc_id") or "doc"
        chunk_idx = payload.get("chunk_index") if payload.get("chunk_index") is not None else meta.get("chunk_index")
        if chunk_idx is None:
            chunk_idx = fallback_index
        chunk_id = f"{doc_id}_chunk_{chunk_idx}"

    return chunk_id

class DistanceTagReranker:
    """
    Reranker basato sui Tag in comune, che inoltre fa uso delle distanze modificate (Classi di Distanza) per recuperare
    ulteriori chunk qualora quelli originari non fossero sufficienti.
    I chunk originariamente recuperati e quelli recuperati tramite il distance_retriever vengono poi suddivisi in 4 categorie,
    la cui importanza è 1 > 2 > 3 > 4:
            - categoria 1: i chunk del 1° retrieval che posseggono i conversation_tags (nel loro intero se più di uno)
            - categoria 2: i chunk del 2° retrieval che posseggono i conversation_tags (nel loro intero se più di uno)
            - categoria 3: i chunk del 2° retrieval che NON condividono i conversation_tags; un possesso parziale è usato per assegnare bonus
            - categoria 4: i chunk del 1° retrieval che NON condividono i conversation_tags; un possesso parziale è usato per assegnare bonus

    Combina i candidati del 1° retrieval (vettoriale puro) e del 2° retrieval (distance_retriever),
    riordinandoli internamente per ciascuna categoria e troncando la lista finale ai TOP_J richiesti.

    Ciascuna categoria è ordinata per final_score, il quale è calcolato secondo i criteri passati da ORDER_BY; supponendo "TAGS" "DIST:
            - categoria 1: ordinata secondo la sola sovrapposizione di assigned e automatic tags, con pesi relativi; la presenza dei conv_tag è ovviamente scontata
            - categoria 2: ordinata secondo la sovrapposizione di assigned e automatic tags, più il distance_factor; presenza dei conv_tag scontata
            - categoria 3: ordinata secondo la sovrapposizione di conversation tags parziali e assigned e automatic tags, più il distance_factor
            - categoria 4: ordinata secondo la sovrapposizione di conversation tags parziali e assigned e automatic tags  
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        sidecar_manager: SidecarManager,
        collection_name: str=COLLECTION_NAME,
        distance_retriever: Optional[DistanceRetriever] = None,
    ):
        if sidecar_manager is None:
            raise ValueError("!!!>>> sidecar_manager è obbligatorio e non può essere None.")
        if qdrant_client is None:
            raise ValueError("!!!>>> qdrant_client è obbligatorio e non può essere None.")

        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.sidecar_manager = sidecar_manager
        self.distance_retriever = distance_retriever or (
            DistanceRetriever(
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                sidecar_manager=self.sidecar_manager,
            )
        )

    def generic_rerank(
        self,
        retrieved_points: List[Any],
        order_by: List[str],
        top_j: int,
        query_vector: Union[List[float], np.ndarray],
        query_tags: Dict[str, List[str]],
        distance_retrieved_records: Optional[List[Dict[str, Any]]] = None,
        top_m: Optional[int]= None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Esegue il reranking dividendo in 4 categorie, distintamente rioridnate secondo i criteri di ORDER_BY e restituisce
        i TOP_J post-reorder.
        Tra di questi, se c'è posto, ci sono i TOP_M recurati tramite il distance_retriever.

        Args:
            retrieved_points: Punti recuperati da Qdrant tramite retrieval vettoriale.
                              Devono essere oggetti di Qdrant; configurato per gestire quanto di seguito:
                              response = qdrant_client.query_points(
                                  collection_name=COLLECTION_NAME,
                                  query=query_vector,
                                  limit=PERSONALISED_TOP_K,
                                  with_vectors=True,
                                  with_payload=True
                              )
                              raw_records = response.points
                              -> .generic_rerank(retrieved_points=raw_records, ...)
            distace_retrieved_records: Record recuperati tramite distance_retrieval; se non passati, vengono calcolati.
                                       Devono essere dizionari di return di distance_retriever.retrieve_close()
            order_by: Lista di flag che regolano filtri e moltiplicatori:
                    - "CONV_TAG_ONLY" (esclude nativamente categorie 3 e 4): Filtro rigido (ALL match sui conversation_tags).
                    - "DIST" (per le categorie 2 e 3): Applica il fattore moltiplicativo della classe di distanza.
                    - "TAGS" (per le categorie 1-4): Applica bonus di sovrapposizione dei tag.
                    - "SCORE" o "" (per le categorie 1-4): Calcola unicamente la similarità coseno base. Se presente con altri, questo NON ha la priorità.
            top_j: Numero massimo di vicini da restituire dopo l'ordinamento.
            query_vector: Vettore di embedding della query.
            query_tags: Dizionario contenente i tag della query divisi per categoria:
                        {"conversation_tags": [...], "assigned_tags": [...],
                         "automatic_tags": [...], "all_tags": [...]}.
            top_m: Eventuale valore di override di quello calcolato nel metodo che impone un numero massimo di vicini da recuperare e restituire.

        Returns:
            Dizionario con
                    -[Debug] Lista di dizionari descrittivi con final_score calcolato e ordinati per categorie e in modo decrescente.
                    -I TOP_J richiesti di tale lista
                    -[Debug] I punti recuperati dal distance_retriever
        """
        empty_response = {
            "top_j_ranked": [],
            "final_ranked": [],
            "distance_retrieved_records": []
        }
        if not retrieved_points and not distance_retrieved_records:
            return empty_response

        if top_j is None or top_j <= 0:
            logger.warning("!>> Reranking annullato per il valore di TOP_J non valido: %s", top_j)
            return empty_response

        # Rimozione eventuali tag duplicati nelle gerarchie inferiori
        query_tags = query_tags or {} # se dovesse essere vuoto
        query_tags = build_query_tags(
            conversation_tags=query_tags.get("conversation_tags", []),
            assigned_tags=query_tags.get("assigned_tags", []),
            automatic_tags=query_tags.get("automatic_tags", []),
        )

        # Normalizzazione flag di order_by
        order_by_to_upper = [str(item).upper().strip() for item in (order_by or []) if item]

        sidecar_data = self.sidecar_manager.load_data()
        tag_overrides = sidecar_data.get("tag_overrides", {})

        conv_tags = query_tags.get("conversation_tags", [])
        ass_tags = query_tags.get("assigned_tags", [])
        auto_tags = query_tags.get("automatic_tags", [])
        all_query_tags = query_tags.get("all_tags", [])

        conv_set = set(conv_tags)
        ass_set = set(ass_tags)
        auto_set = set(auto_tags)
        all_query_set = set(all_query_tags)

        ### Partizionamento in categorie 1 e 4 dei retrieved_points
        category1_candidates = []
        category4_candidates = []
        for idx, hit  in enumerate(retrieved_points or []):
            payload = getattr(hit, "payload", {}) or {}
            chunk_id = _extract_chunk_id(hit, fallback_index=idx)

            # cos_sim con query
            initial_score = max(0.0, float(getattr(hit, "score", 0.0)))

            chunk_info = tag_overrides.get(chunk_id, {})
            chunk_tags = chunk_info.get("user_tags", payload.get("user_tags", []))
            chunk_tags_set = set(chunk_tags)

            # Requisito conversation_tag
            if "CONV_TAG_ONLY" in order_by_to_upper and conv_tags:
                if not conv_set.issubset(chunk_tags_set):
                    continue

            # Controllo generale appartenenza a categoria 1 o 4 (se chunk ha il conv_tag o meno)
            has_all_conv_tag = bool(conv_tags and conv_set.issubset(chunk_tags_set))

            # Recupero vettore da inserire in output
            raw_vector = getattr(hit, "vector", None)
            if isinstance(raw_vector, dict): # gestisce sia liste/array che dict come vettori
              chunk_vector = next(iter(raw_vector.values())) if raw_vector else None
            else:
              chunk_vector = raw_vector

            # Distance Factor inutile in questa fase
            dist_factor = 1.0

            # Moltiplicatore per TAG coincidenti
            tag_alteration = 1.0
            matched_all = []
            if "TAGS" in order_by_to_upper:
                matched_conv = list(conv_set.intersection(chunk_tags_set))
                matched_ass = list(ass_set.intersection(chunk_tags_set))
                matched_auto = list(auto_set.intersection(chunk_tags_set))
                matched_all = list(all_query_set.intersection(chunk_tags_set))

                bonus_conv = len(matched_conv) * TAG_BOOST_ASSIGNED_CONV
                bonus_ass = len(matched_ass) * TAG_BOOST_ASSIGNED_CONV
                bonus_auto = len(matched_auto) * TAG_BOOST_AUTO

                if not has_all_conv_tag: # categoria 4: contano i match parziali
                    total_bonus = bonus_conv + bonus_ass + bonus_auto
                else: # categoria 1: tutti fanno perfect_match con il conv_tag, contano solo assigned e auto
                    total_bonus = bonus_ass + bonus_auto

                if total_bonus > 0:
                    tag_alteration = 1.0 + min(total_bonus, TAG_MAX_BOOST)

            # Punteggio finale
            raw_final_score = initial_score * dist_factor * tag_alteration

            candidate_item = {
                "chunk_id": chunk_id,
                "point": hit,
                "initial_score": round(initial_score, 4),
                "dist_factor": round(dist_factor, 3),
                "tag_alteration": round(tag_alteration, 3),
                "final_score": round(raw_final_score, 4),
                "_raw_final_score": raw_final_score,
                "source_retrieval": "vectorial_retriever",
                "category": 1 if has_all_conv_tag else 4,
                "vector": chunk_vector,
                "payload": payload,
                "chunk_tags": chunk_tags,
                "matched_tags": matched_all,
            }

            if has_all_conv_tag:
                category1_candidates.append(candidate_item)
            else:
                category4_candidates.append(candidate_item)

        ### Distance Retrieval se necessario per partizionamento in categorie 2 e 3
        if distance_retrieved_records is None:
            n_category1 = len(category1_candidates)
            if top_m is not None:
                effective_top_m = top_m # Override
            else:
                effective_top_m = max(0, top_j - n_category1)

            # Necessità il distance_retrieval
            if effective_top_m > 0 and category1_candidates:
                to_analyze_nodes = category1_candidates
                if to_analyze_nodes and query_vector is not None:
                    distance_retrieved_records = self.distance_retriever.retrieve_close(
                        to_analyze=to_analyze_nodes,
                        order_by=order_by_to_upper,
                        top_m=effective_top_m,
                        query_vector=query_vector,
                        query_tags=query_tags,
                    )

        ### Sorting dei candidati per categorie 1 e 4 (2 e 3 già ordinate da retrieve_close())
        category1_candidates.sort(key=lambda x: x["_raw_final_score"], reverse=True)
        category4_candidates.sort(key=lambda x: x["_raw_final_score"], reverse=True)

        # Deduplicazione di eventuali chunk della cat4 presenti anche in cat3.
        dist_retrieved_ids = {item["chunk_id"] for item in (distance_retrieved_records or [])}
        category4_candidates = [c for c in category4_candidates if c["chunk_id"] not in dist_retrieved_ids]

        final_ranked = category1_candidates + (distance_retrieved_records or []) + category4_candidates

        for item in final_ranked:
            if "_raw_final_score" in item:
                del item["_raw_final_score"]

        ### Return per debug e wrapping
        return {
            "top_j_ranked": final_ranked[:top_j],
            "final_ranked": final_ranked,
            "distance_retrieved_records": distance_retrieved_records or [],
        }

    def rerank_top_j(
        self,
        retrieved_points: List[Any],
        order_by: List[str],
        top_j: int,
        query_vector: Union[List[float], np.ndarray],
        query_tags: Dict[str, List[str]],
        distance_retrieved_records: Optional[List[Dict[str, Any]]] = None,
        top_m: Optional[int]= None
    ) -> List[Dict[str, Any]]:
        """
        Wrapper di generic_rerank() che returna di default solo i TOP_J records
        """
        rrnk_rslt = self.generic_rerank(
            retrieved_points=retrieved_points,
            order_by=order_by,
            top_j=top_j,
            query_vector=query_vector,
            query_tags=query_tags,
            distance_retrieved_records=distance_retrieved_records,
            top_m=top_m,
        )
        return rrnk_rslt.get("top_j_ranked", [])

    def rerank_top_k(
        self,
        retrieved_points: List[Any],
        order_by: List[str],
        query_vector: Union[List[float], np.ndarray],
        query_tags: Dict[str, List[str]],
        top_k: int=RETRIEVAL_TOP_K,
        distance_retrieved_records: Optional[List[Dict[str, Any]]] = None,
        top_m: Optional[int]= None
    ) -> List[Dict[str, Any]]:
        """
        Wrapper di generic_rerank() che returna di default solo src.config.RETRIEVAL_TOP_K records
        """
        return self.rerank_top_j(
            retrieved_points=retrieved_points,
            order_by=order_by,
            top_j=top_k,
            query_vector=query_vector,
            query_tags=query_tags,
            distance_retrieved_records=distance_retrieved_records,
            top_m=top_m,
        )

    def rerank(
        self,
        retrieved_points: List[Any],
        order_by: List[str],
        query_vector: Union[List[float], np.ndarray],
        query_tags: Dict[str, List[str]],
        top_n: int=RERANKING_TOP_N,
        distance_retrieved_records: Optional[List[Dict[str, Any]]] = None,
        top_m: Optional[int]= None
    ) -> List[Dict[str, Any]]:
        """
        Wrapper di generic_rerank() che returna di default solo src.config.RERANKING_TOP_N records
        """
        return self.rerank_top_j(
            retrieved_points=retrieved_points,
            order_by=order_by,
            top_j=top_n,
            query_vector=query_vector,
            query_tags=query_tags,
            distance_retrieved_records=distance_retrieved_records,
            top_m=top_m,
        )

