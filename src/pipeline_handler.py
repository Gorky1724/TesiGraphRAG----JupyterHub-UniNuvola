import sys
import json
import math
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama import ChatOllama
from src.config import (
    DEFAULT_KEEP_ALIVE,
    DEFAULT_NUM_THREAD,
    LLM_MODEL,
    OLLAMA_URL,
)

from src.config import (
    QDRANT_URL, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL, 
    RETRIEVAL_TOP_K,
    RERANKING_TOP_N
)

from src.sidecar_manager import SidecarManager
from src.graph_builder import KnowledgeGraphBuilder
from src.chunk_widget import ChunkGraphWidget
from src.tag_assigner import TagAssigner

from src.distance_retriever import DistanceRetriever, build_query_tags
from src.distance_tag_reranker import DistanceTagReranker

# [DEBUG: Logger]
logger = logging.getLogger(__name__)

class PipelineHandler:
    """
    Gestore centralizzato della pipeline graphRAG implementata.
    Offre i metodi per le varie fasi e la gestione e modifica dei parametri utilizzati
    per poter effettuare le necessarie prove.
    """

    ### Inizializzazione moduli
    def __init__(
        self,
        sidecar_path: str=None,
        queries_path: Optional[Union[str, Path]] = None
    ):
        # Inizializzazione sidecar_manager
        self.sidecar_path = sidecar_path

        self.sidecar_manager = None 
        if not self.sidecar_path:
            self.sidecar_manager=SidecarManager()
        else:
            self.sidecar_manager=SidecarManager(filepath=self.sidecar_path)

        # Inizializzazione Vector Store con FastEmbed nativo
        self.embeddings = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
        self.qdrant_client = QdrantClient(url=QDRANT_URL)

        # Inizializzazione llama
        self.llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_URL,
            keep_alive=DEFAULT_KEEP_ALIVE,
            num_thread=DEFAULT_NUM_THREAD,
            temperature=0,
        )

        # Inizializzazione widget
        self.widget = ChunkGraphWidget(sidecar=self.sidecar_manager)

        # Inizializzazione moduli gestinoe Tag
        self.tag_assigner = TagAssigner()

        # Inizializzazione Reranker
        self.dt_reranker = DistanceTagReranker(qdrant_client=self.qdrant_client, sidecar_manager=self.sidecar_manager)

        # Parametri query
        self.query_text = ""
        self.queries_path = queries_path
        self.query_vector = None

        self.conversation_tags = []
        self.assigned_tags = []
        self.automatic_tags = []
        self.auto_tag_assignation = False
        self.query_tags = {}

        # Parametri retrieval e reranking
        self.personalised_top_n = RERANKING_TOP_N
        self.personalised_top_k = RETRIEVAL_TOP_K
        self.personalised_top_j = self.personalised_top_k
        self.order_by = [""]

        # Record Retrieval Vettoriale
        self.raw_records = None

        # Record Retrieval su Classi di Distanza
        self.distance_retrieved_records = None

        # Record post-DistanceTagReranking
        self.rerank_generic_rslt = None
        self.all_reranked = None
        self.top_j_reranked = None

        logger.info("?> Pipeline Inizializzata")

    ### Gestione Parametri
    def set_parameters(
        self,
        top_j: Optional[int] = None,
        top_k: Optional[int] = None,
        top_n: Optional[int] = None,
        order_by: Optional[List[str]] = None,
    ):
        """Aggiorna le costanti e i parametri di esecuzione della pipeline. Sostituisce la lista di order_by"""
        if top_j is not None:
            self.personalised_top_j = top_j
        if top_k is not None:
            self.personalised_top_k = top_k
        if top_n is not None:
            self.personalised_top_n = top_n
        if order_by is not None:
            self.order_by = order_by

        logger.info(f"?> Parametri aggiornati: top_j={self.personalised_top_j}, top_k={self.personalised_top_k}, top_n={self.personalised_top_n}, order_by={self.order_by}")

    ### Gestione Query
    def update_query(
        self,
        new_text: Optional[str] = None,
        new_id: Optional[str] = None,
    ):
        """
        Aggiorna il testo della query e il suo embedding.
        Se forniti entrambi i parametri, da la priorità all'ID.
        """
        if new_id:
            if not self.queries_path:
                logger.warning("!!!> update_query(...) -> OPERAZIONE ABORTITA: Fornito un QUERY_ID ma manca il QUERY_PATH.")
                return

            q_path = Path(self.queries_path).expanduser().resolve()
            if not q_path.exists():
                logger.error(f"!!!> File query non trovato: {q_path}")
                return

            with open(q_path, "r", encoding="utf-8") as f:
                benchmark_queries = json.load(f)

            matched = [q for q in benchmark_queries if str(q.get("id")) == str(new_id)]
            if not matched:
                logger.error(f"!!!> Query ID '{new_id}' non trovato nel file JSON.")
                return

            query_obj = matched[0]
            self.query_text = query_obj["query"]
            logger.info(f"?> Testo Query aggiornato: {self.queries_path}--ID>{new_id} ")
        elif new_text:
            self.query_text = new_text
            logger.info(f"?> Testo Query aggiornato con il testo fornito ")
        else:
            logger.warning(f"!> update_query(...) -> Invocazione senza alterazioni")
            return

        self.query_vector = self.embeddings.embed_query(self.query_text)
        logger.info(f"?> Vettore embedding Query aggiornato: text:{self.query_text}")

    def print_current_global_tags(self):
        """
        Stampa i tag globali attualmente in uso.
        """
        sidecar_data = self.sidecar_manager.load_data()
        global_tags_dict = sidecar_data.get("global_tags", {})
        global_tags = list(global_tags_dict.keys())
        print(f">> Testo Query: '{self.query_text}'\n")
        print(f">> Tag Globali: {global_tags}")

    def update_query_tags(
        self,
        new_conv_list: Optional[List[str]] = None,
        new_ass_list: Optional[List[str]] = None,
        auto_ass_flag: Optional[bool] = None,
        print_debug: Optional[bool] = False
    ):
        """
        Aggiorna le liste dei tag assegnati alla query, sostituendole con quelle passate.
        """
        if new_conv_list is not None:
            self.conversation_tags = new_conv_list
        if new_ass_list is not None:
            self.assigned_tags = new_ass_list
        if auto_ass_flag is not None:
            self.auto_tag_assignation = auto_ass_flag

        sidecar_data = self.sidecar_manager.load_data()
        global_tags_dict = sidecar_data.get("global_tags", {})
        global_tags = list(global_tags_dict.keys())

        automatic_tags = []
        if global_tags and self.auto_tag_assignation:
            automatic_tags = self.tag_assigner.assign_tags(
                text=self.query_text, candidate_tags=global_tags, vector=self.query_vector
            )

        self.query_tags = build_query_tags(
            conversation_tags=self.conversation_tags,
            assigned_tags=self.assigned_tags,
            automatic_tags=automatic_tags,
        )

        logger.info(f"?> Query vettorizzata. Tag assegnati: {self.query_tags.get('all_tags', [])}")

        if print_debug:
            print("!>> Tag assegnati alla query:")
            print(f"   >>> Tag di Conversazione: {self.query_tags.get('conversation_tags', [])}")
            print(f"   >>> Tag Manuali della Query: {self.query_tags.get('assigned_tags', [])}")
            if self.auto_tag_assignation:
                if global_tags:
                    print(f"   >>> Tag Automaticamente assegnati alla Query: {self.query_tags.get('automatic_tags', [])}")
                    print(f"       |> Tag Globali disponibili: {global_tags}")
                else:
                    print("   >>> Assegnazione Automatica abilitata, ma non sono presenti Tag Globali da assegnare")
            else:
                print("   >>> Assegnazione Tag Automatici disabilitata")
            print(f"<<< Lista completa: {self.query_tags.get('all_tags', [])}")

    ### Retrieval vettoriale
    def run_vectorial_retrieval(self):
        """Esegue il retrieval vettoriale restituendo i TOP_K qdrant.points con gli associati vettori e payload."""
        if self.query_vector is None:
            raise ValueError("!!!>>> query_vector non presente. Esegui prima update_query().")

        limit = max(1, self.personalised_top_k)
        response = self.qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_vector,
            limit=limit,
            with_vectors=True,
            with_payload=True
        )
        self.raw_records = response.points

        logger.info(f"?> Recuperati {len(self.raw_records)} record con relativi vettori di embedding")

    ### [DEBUG] Distance Retriever
    def run_distance_retriever(self, to_retrieve: int, print_debug: Optional[bool] = False):
        """[DEBUG] Istanzia ed esegue il distance retriever, non aggiorna la variabile di classe ma ritorna il risultato"""
        distance_retriever = DistanceRetriever(qdrant_client=self.qdrant_client, sidecar_manager=self.sidecar_manager)

        distance_retrieved_records = distance_retriever.retrieve_close(
                to_analyze=self.raw_records,
                order_by=self.order_by,
                top_m=to_retrieve,
                query_vector=self.query_vector,
                query_tags=self.query_tags
        )

        logger.info(f"?[DEBUG]> Distance Retriever eseguito")

        if print_debug:
            print(f"!>> Il Distance Retriever ha recuperato {len(distance_retrieved_records)} record")
            for r in distance_retrieved_records:
                print(f"    >>> cid: {[r.get('chunk_id', 'not-found')]}")
                print(f"        |> category: {[r.get('category', 'not-found')]}")
                print(f"           |> final_score: {[r.get('final_score', 'not-found')]}")
        return distance_retrieved_records

    ### DistanceTagReranking
    def run_top_j_distance_tag_reranking(self, print_debug: Optional[bool] = False):
        """Esegue il distance_tag_reranker.generic_reranking(...) con personalised_top_j come parametro."""
        self.rerank_generic_rslt = self.dt_reranker.generic_rerank(
            retrieved_points=self.raw_records,
            order_by=self.order_by,
            top_j=self.personalised_top_j,
            query_vector=self.query_vector,
            query_tags=self.query_tags
        )

        self.all_reranked = self.rerank_generic_rslt.get("final_ranked", [])
        self.top_j_reranked = self.rerank_generic_rslt.get("top_j_ranked", [])
        self.distance_retrieved_records = self.rerank_generic_rslt.get("distance_retrieved_records", [])

        if print_debug:
            print(f"?> [DEBUG INFO] all_reranked:")
            print(f"         >> Il Reranker ha restituito {len(self.all_reranked)} record")
            for r in self.all_reranked:
                print(f"    >>> cid: {[r.get('chunk_id', 'not-found')]}")
                print(f"        |> category: {[r.get('category', 'not-found')]}")
                print(f"           |> final_score: {[r.get('final_score', 'not-found')]}")
            print("="*80)
            print(f"?> [DEBUG INFO] distance_retrieved_records:")
            print(f"         >> Durante il reranking il Distance Retriever ha recuperato {len(self.distance_retrieved_records)} record")
            for r in self.distance_retrieved_records:
                print(f"    >>> cid: {[r.get('chunk_id', 'not-found')]}")
                print(f"        |> category: {[r.get('category', 'not-found')]}")
                print(f"           |> final_score: {[r.get('final_score', 'not-found')]}")
            print("="*80)
            print(f"!>>> top_j_reranked, restituiti top_j={len(self.top_j_reranked)} ")
            for r in self.top_j_reranked:
                print(f"    >>> cid: {[r.get('chunk_id', 'not-found')]}")
                print(f"        |> category: {[r.get('category', 'not-found')]}")
                print(f"           |> final_score: {[r.get('final_score', 'not-found')]}")

    ### Visualizzazione Grafo
    def build_graph_data(self, records: List[Any]) -> Dict[str, Any]:
        """
        Costruisce il dizionario JSON del grafo a partire da una lista di record
        (accetta sia oggetti ScoredPoint di Qdrant che dizionari del Reranker).
        I record del PipelineHandler sono pubblici.
        Altrimenti si può utilizzare il metodo helper get_phase_graph_data(...).
        """
        builder = KnowledgeGraphBuilder()
        tag_overrides = self.sidecar_manager.load_data().get("tag_overrides", {})
        seen_chunk_ids = set()

        for idx, rec in enumerate(records or []):
            if isinstance(rec, dict):
                payload = rec.get("payload", rec)
                vector = rec.get("vector")
                cid = rec.get("chunk_id")
            else:
                payload = getattr(rec, "payload", {}) or {}
                vector = getattr(rec, "vector", None)
                cid = getattr(rec, "id", None)

            meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload
            doc_id = payload.get("doc_id") or meta.get("doc_id") or "doc"

            raw_idx = payload.get("chunk_index") if payload.get("chunk_index") is not None else meta.get("chunk_index")
            chunk_idx = raw_idx if raw_idx is not None else idx

            chunk_id = cid or payload.get("chunk_id") or meta.get("chunk_id") or f"{doc_id}_chunk_{chunk_idx}"

            # Deduplica eventuali chunk_id già inseriti
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            text = payload.get("text", payload.get("page_content", ""))

            base_tags = payload.get("user_tags") or meta.get("user_tags") or payload.get("tags") or meta.get("tags") or []
            chunk_sidecar_info = tag_overrides.get(chunk_id, {})
            tags = chunk_sidecar_info.get("user_tags", base_tags)

            builder.add_chunk_node(
                chunk_id=chunk_id,
                text=text,
                vector=vector,
                tags=tags,
            )

        builder.auto_connect_nodes()
        return builder.to_json_data()

    def get_phase_graph_data(self, phase: str = "vectorial") -> Dict[str, Any]:
        """
        Restituisce il JSON del grafo per la fase specificata; valori consigliati:
            - 'vectorial' (1° Retrieval)
            - 'distance' (2° Retrieval / Vicini)
            - 'reranked' (Post-Reranking - top_j)
            - 'all_reranked' (Post-Reranking - all)
        """
        phase = phase.lower().strip()
        if phase in ["vectorial"]:
            return self.build_graph_data(self.raw_records or [])
        elif phase in ["distance"]:
            # Unisce i record del 1° retrieval con quelli recuperati dal distance_retriever
            all_pts = list(self.raw_records or []) + list(self.distance_retrieved_records or [])
            return self.build_graph_data(all_pts or [])
        elif phase in ["reranked"]:
            top_j_pts = self.top_j_reranked
            return self.build_graph_data(top_j_pts or [])
        elif phase in ["all_reranked"]:
            all_rer_pts = self.all_reranked
            return self.build_graph_data(all_rer_pts or [])
        else:
            raise ValueError(f"!!!>>> Fase sconosciuta '{phase}'. Usare: 'vectorial', 'distance', 'reranked', 'all_reranked'.")

    def update_widget_view(self, phase: str = "vectorial"):
        """Aggiorna l'istanza unica del widget fornita con i dati della fase scelta."""
        graph_data = self.get_phase_graph_data(phase)
        self.widget.load_graph(graph_data)
        logger.info(f"?> Widget aggiornato alla fase '{phase}' ({len(graph_data['nodes'])} nodi)")
