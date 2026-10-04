import sys
import uuid
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from qdrant_client import QdrantClient

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.sidecar_manager import SidecarManager
from src.config import (
    COLLECTION_NAME,
    DISTANCE_CLASS_FACTORS,
    TAG_BOOST_ASSIGNED_CONV, TAG_BOOST_AUTO, TAG_MAX_BOOST,
    MIN_COSINE_THRESHOLD
)

logger = logging.getLogger(__name__)

def cosine_sim(v1: Union[List[float], np.ndarray], v2: Union[List[float], np.ndarray]) -> float:
    """Calcola la Cosine Similarity tra due vettori."""
    arr1 = np.array(v1, dtype=float)
    arr2 = np.array(v2, dtype=float)
    norm1 = np.linalg.norm(arr1)
    norm2 = np.linalg.norm(arr2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))

# Per garantire che i query_tag non si ripetano nelle varie categorie
def build_query_tags(
    conversation_tags: Optional[List[str]] = None,
    assigned_tags: Optional[List[str]] = None,
    automatic_tags: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """
    Costruisce il dizionario query_tags garantendo la disgiunzione gerarchica dei tag:
    conversation_tags > assigned_tags > automatic_tags.
    """
    seen = set()

    clean_conv: List[str] = []
    for t in conversation_tags or []:
        if t and t not in seen:
            clean_conv.append(t)
            seen.add(t)

    clean_assigned: List[str] = []
    for t in assigned_tags or []:
        if t and t not in seen:
            clean_assigned.append(t)
            seen.add(t)

    clean_auto: List[str] = []
    for t in automatic_tags or []:
        if t and t not in seen:
            clean_auto.append(t)
            seen.add(t)

    all_tags: List[str] = clean_conv + clean_assigned + clean_auto

    return {
        "conversation_tags": clean_conv,
        "assigned_tags": clean_assigned,
        "automatic_tags": clean_auto,
        "all_tags": all_tags,
    }

class DistanceRetriever:
    """
    Modulo di 2° Retrieval basato sulle sole adiacenze e distanze modificate nel sidecar.
    Recupera da Qdrant i chunk vicini ("AVVICINATI" / "MOLTO_AVVICINATI") rispetto
    ai nodi estratti nel 1° retrieval, calcolandone e restituendo la cos_sim con la query.
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        sidecar_manager: SidecarManager,
        collection_name: str = COLLECTION_NAME
    ):
        if sidecar_manager is None:
            raise ValueError("!!!>>> sidecar_manager è obbligatorio e non può essere None.")
        if qdrant_client is None:
            raise ValueError("!!!>>> qdrant_client è obbligatorio e non può essere None.")

        self.qdrant_client = qdrant_client
        self.sidecar_manager = sidecar_manager
        self.collection_name = collection_name

    def retrieve_close(
        self,
        to_analyze: List[Any],
        order_by: List[str],
        top_m: int,
        query_vector: Union[List[float], np.ndarray],
        query_tags: Dict[str, List[str]],
        predq_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Esegue il retrieval su base di distanza e calcola il final_score per tutti i candidati adiacenti.

        Args:
            to_analyze: Punti da analizzare per trovare i vicini. Possono essere oggetti di Qdrant o dizionari
            order_by: Lista di flag che regolano filtri e moltiplicatori:
                    - "CONV_TAG_ONLY": Filtro rigido (ALL match sui conversation_tags).
                    - "DIST": Applica il fattore moltiplicativo della classe di distanza.
                    - "TAGS": Applica bonus di sovrapposizione dei tag.
                    - "SCORE" o "": Calcola unicamente la similarità coseno base. Se presente con altri, questo NON ha la priorità.
            top_m: Numero massimo di vicini da restituire dopo l'ordinamento.
            query_vector: Vettore di embedding della query.
            query_tags: Dizionario contenente i tag della query divisi per categoria:
                        {"conversation_tags": [...], "assigned_tags": [...],
                         "automatic_tags": [...], "all_tags": [...]}.
                        Se presente il predq_id, si presuppone che i graphically_assigned_tags siano già stati aggiunti.
            predq_id: l'ID della query pre-determinata che sta venendo eseguita (o la cui query posta era simile).
                      Se presente, vuol dire che stiamo analizzando una query pre-determinata. Se assente, la query è una qualunque.
                      Si considera nel calcolo del final_anche il DistanceFactor chunk-query, indipendentemente dal valore di order_by
                      La suddivisione in categorie non cambia, viene solo cambiato il punteggio interno.

        Returns:
            Lista di dizionari descrittivi con final_score calcolato e ordinati per categorie e in modo decrescente.
        """
        if (not to_analyze and not predq_id) or query_vector is None:
            return []

        if top_m is None or top_m <= 0:
            logger.warning("!>> Distance Retrieval annullato per il valore di TOP_M non valido: %s", top_m)
            return []

        # Rimozione eventuali tag duplicati nelle gerarchie inferiori -- 
        query_tags = query_tags or {} # se dovesse essere vuoto
        query_tags = build_query_tags(
            conversation_tags=query_tags.get("conversation_tags", []),
            assigned_tags=query_tags.get("assigned_tags", []),
            automatic_tags=query_tags.get("automatic_tags", []),
        )

        # Normalizzazione flag di order_by
        order_by_to_upper = [str(item).upper().strip() for item in (order_by or []) if item]

        sidecar_data = self.sidecar_manager.load_data()
        modified_adjacencies = sidecar_data.get("modified_adjacencies", {})
        tag_overrides = sidecar_data.get("tag_overrides", {})
        pred_queries = sidecar_data.get("predetermined_queries", {})

        # ID dei chunk da analizzare per evitare duplicazioni
        seen_ids = set()
        for idx, hit in enumerate(to_analyze):
            if isinstance(hit, dict): # Se hit è un dizionario
                payload = hit.get("payload", {}) or {}
                meta = (
                    payload.get("metadata")
                    if isinstance(payload.get("metadata"), dict)
                    else payload
                )
                chunk_id = (
                    hit.get("chunk_id") or payload.get("chunk_id") or meta.get("chunk_id")
                )
            else: # Se hit è un oggetto di Qdrant
                payload = getattr(hit, "payload", {}) or {}
                meta = (
                    payload.get("metadata")
                    if isinstance(payload.get("metadata"), dict)
                    else payload
                )
                chunk_id = payload.get("chunk_id") or meta.get("chunk_id")

            # Fallback se l'id ancora non è definito
            if not chunk_id:
                doc_id = payload.get("doc_id") or meta.get("doc_id") or "doc"
                chunk_idx = (
                    payload.get("chunk_index")
                    if payload.get("chunk_index") is not None
                    else meta.get("chunk_index")
                )
                chunk_id = f"{doc_id}_chunk_{chunk_idx}"

            seen_ids.add(chunk_id)

        # Mappatura adiacenze
        candidate_neighbors: Dict[str, Dict[str, Any]] = {}

        # Vicini a to_analyze
        for cid in seen_ids:
            if cid in modified_adjacencies:
                neighbors = modified_adjacencies[cid]
                for neighbor_id, info in neighbors.items():
                    if neighbor_id not in seen_ids:
                        d_class = info.get("distance_class", "INVARIATI")
                        if d_class in ["AVVICINATI", "MOLTO_AVVICINATI"]:
                            factor = DISTANCE_CLASS_FACTORS.get(d_class, 1.0)
                            # Prendiamo dist_factor massimo trai possibili
                            if neighbor_id not in candidate_neighbors or factor > candidate_neighbors[neighbor_id]["distance_factor"]:
                                candidate_neighbors[neighbor_id] = {
                                    "distance_class": d_class,
                                    "distance_factor": factor,
                                    "predq_alteration": 1.0
                                }

        # Vicini alla pred-query (se presente)
        if predq_id and predq_id in pred_queries:
            predq_data = pred_queries[predq_id] or {}
            predq_adj = predq_data.get("adjacencies") or {}
            for neighbor_id, info in predq_adj.items():
                if neighbor_id not in seen_ids:
                    d_class = info.get("distance_class", "INVARIATI")
                    if d_class in ["AVVICINATI", "MOLTO_AVVICINATI"]:
                        factor = DISTANCE_CLASS_FACTORS.get(d_class, 1.0)
                        if neighbor_id not in candidate_neighbors:
                            candidate_neighbors[neighbor_id] = {
                                "distance_class": d_class,
                                "distance_factor": 1.0,
                                "predq_alteration": factor
                            }
                        else:
                            existing_predq_factor = candidate_neighbors[neighbor_id].get("predq_alteration", 1.0)
                            if factor > existing_predq_factor:
                                candidate_neighbors[neighbor_id]["predq_alteration"] = factor


        if not candidate_neighbors:
            return []

        # Recupero chunk da Qdrant via uuid
        chunk_id_to_uuid = {
            cid: str(uuid.uuid5(uuid.NAMESPACE_DNS, cid))
            for cid in candidate_neighbors.keys()
        }
        target_uuids = list(chunk_id_to_uuid.values())
        uuid_to_chunk_id = {v: k for k, v in chunk_id_to_uuid.items()}  # Mappa inversa

        try:
            retrieved_records = self.qdrant_client.retrieve(
                collection_name=self.collection_name,
                ids=target_uuids,
                with_payload=True,
                with_vectors=True,
            )
        except Exception as e:
            logger.error("!!!> [DistanceRetriever] Errore critico nel retrieval da Qdrant: %s", e, exc_info=True)
            return []

        # Liste tag
        conv_tags = query_tags.get("conversation_tags", [])
        assigned_tags = query_tags.get("assigned_tags", [])
        auto_tags = query_tags.get("automatic_tags", [])
        all_query_tags = query_tags.get("all_tags", [])

        # Elaborazione, filtraggio e calcolo dei final score,
        #  sui criteri di order_by
        category2_candidates = []
        category3_candidates = []
        for rec in retrieved_records:
            payload = getattr(rec, "payload", {}) or {}

            # Recupero chunk_id con mappa inversa
            rec_id_str = str(rec.id)
            chunk_id = uuid_to_chunk_id.get(rec_id_str)

            # Fallback calcolo chunk_id se non trovato
            if not chunk_id:
                meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
                doc_id = payload.get("doc_id") or meta.get("doc_id") or "doc"
                chunk_idx = payload.get("chunk_index") if payload.get("chunk_index") is not None else meta.get("chunk_index")
                chunk_id = payload.get("chunk_id") or meta.get("chunk_id") or f"{doc_id}_chunk_{chunk_idx}"

            # Recupero info chunk
            chunk_info = tag_overrides.get(chunk_id, {})
            chunk_tags = chunk_info.get("user_tags", payload.get("user_tags", []))

            # Requisito conversation_tag
            if "CONV_TAG_ONLY" in order_by_to_upper:
                # Richiesta inclusione totale dei conv_tags
                if conv_tags:
                    has_all = all(cnvtag in chunk_tags for cnvtag in conv_tags)
                    if not has_all:
                        continue

            # Controllo generale appartenenza a categoria 2 o 3 (se chunk ha il conv_tag o meno)
            has_all_conv_tag = bool(conv_tags and all(cnvtag in chunk_tags for cnvtag in conv_tags))

            # cos_sim con query, sempre necessaria
            raw_vector = getattr(rec, "vector", None)
            if isinstance(raw_vector, dict): # gestisce sia liste/array che dict come vettori
                chunk_vector = next(iter(raw_vector.values())) if raw_vector else None
            else:
                chunk_vector = raw_vector

            if chunk_vector is None:
                logger.warning("!> Impossibile reperire il vettore per '%s'. Chunk SALTATO.", chunk_id,)
                continue
            else:
                cos_score = max(0.0, cosine_sim(query_vector, chunk_vector))

            if cos_score < MIN_COSINE_THRESHOLD:
                continue

            neighbor_info = candidate_neighbors.get(chunk_id, {})

            # Moltiplicatore su base Distance Class
            dist_factor = 1.0
            if "DIST" in order_by_to_upper:
                dist_factor = float(neighbor_info.get("distance_factor", 1.0))

            tag_alteration = 1.0
            matched_all = []
            if "TAGS" in order_by_to_upper:
                matched_conv = list(set(conv_tags).intersection(set(chunk_tags)))
                matched_assigned = list(set(assigned_tags).intersection(set(chunk_tags)))
                matched_auto = list(set(auto_tags).intersection(set(chunk_tags)))
                matched_all = list(set(all_query_tags).intersection(set(chunk_tags)))

                bonus_conv = len(matched_conv) * TAG_BOOST_ASSIGNED_CONV
                bonus_assigned = len(matched_assigned) * TAG_BOOST_ASSIGNED_CONV
                bonus_auto = len(matched_auto) * TAG_BOOST_AUTO

                if not has_all_conv_tag: # categoria 3: contano i match parziali
                    total_bonus = bonus_conv + bonus_assigned + bonus_auto
                else: # categoria 2: tutti fanno perfect_match con il conv_tag
                    total_bonus = bonus_assigned + bonus_auto

                if total_bonus > 0:
                    tag_alteration = 1.0 + min(total_bonus, TAG_MAX_BOOST)

            # Moltiplicatore VICINANZA a query pre-determinata
            predq_alteration = 1.0
            if predq_id:
                predq_alteration = float(neighbor_info.get("predq_alteration", 1.0))

            # Punteggio finale
            raw_final_score = cos_score * dist_factor * tag_alteration * predq_alteration

            candidate_data = {
                "chunk_id": chunk_id,
                "point": rec,
                "initial_score": round(cos_score, 4),
                "dist_factor": round(dist_factor, 3),
                "tag_alteration": round(tag_alteration, 3),
                "predq_alteration": round(predq_alteration, 3),
                "final_score": round(raw_final_score, 4),
                "_raw_final_score": raw_final_score,
                "source_retrieval": "distance_retriever",
                "category": 2 if has_all_conv_tag else 3,
                "vector": chunk_vector,
                "payload": payload,
                "chunk_tags": chunk_tags
            }

            if has_all_conv_tag:
                category2_candidates.append(candidate_data)
            else:
                category3_candidates.append(candidate_data)


        # Riordinamento e unione categorie per return top_m
        category2_candidates.sort(key=lambda x: x["_raw_final_score"], reverse=True)
        category3_candidates.sort(key=lambda x: x["_raw_final_score"], reverse=True)

        prioritized_candidates = category2_candidates + category3_candidates
        for item in prioritized_candidates:
            del item["_raw_final_score"]

        return prioritized_candidates[:top_m]

