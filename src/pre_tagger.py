import sys
import logging
from typing import List, Optional, Dict, Union
from pathlib import Path
from qdrant_client import QdrantClient
from tqdm import tqdm #Per barra di avanzamento

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import COLLECTION_NAME, QDRANT_URL
from src.tag_assigner import TagAssigner
from src.sidecar_manager import SidecarManager

import gc # Garbage Collector
import time
import numpy as np

# Per debug e vedere progresso
logger = logging.getLogger(__name__)

class PreTagger:
    """
    Classe per l'esecuzione del pretagging automatico sulla collezione indicata.
    Genera una istanza di SidecarManager per il .json corrispettivo, dunque
    è sempre bene fare il metodo pretagger.close() per eliminare tale istanza
    del sidecar_manager
    """

    def __init__(
        self,
        sidecar_path: str="sidecar_edits.json",
        tag_assigner: Optional[TagAssigner] = None,
        sidecar_manager: Optional[SidecarManager] = None,
        qdrant_client: Optional[QdrantClient] = None,
    ):
        self.sidecar_path = sidecar_path

        self.tag_assigner = tag_assigner or TagAssigner()

        # Tiene traccia se sidecar e qdrant sono stati creati qui (e quindi vanno rilasciati
        #  esplicitamente in close()) oppure passati dall'esterno (nel qual caso
        #  resta responsabilità di chi li ha creati).
        self._owns_sidecar_manager = sidecar_manager is None
        self.sidecar_manager = sidecar_manager or SidecarManager(filepath=self.sidecar_path)
        self.sidecar_path = self.sidecar_manager.filepath

        self._owns_qdrant_client = qdrant_client is None
        self.qdrant_client = qdrant_client or QdrantClient(url=QDRANT_URL)


    # Per sintassi con "with PreTagger(...) as tagger:"
    #  Garantisce la chiusura se durante run() viene sollevata un eccezione
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # Metodi
    def run(
        self,
        candidate_tags: List[str],
        collection_name: str = COLLECTION_NAME,
        batch_size: int = 100,
        start_offset: Optional[Union[int, str]] = None,
        max_chunks: Optional[int] = None,
    ) -> None:
        """
        Scansiona i chunk della collezione e assegna i tag della lista che superano la soglia
        Importando start_offset e max_chunks è possibile delimitare quali chunk della collezione saranno coinvolti
        """
        if not candidate_tags:
            logger.warning("!> Nessun tag candidato fornito")
            return

        logger.info(f"?>> Inizio pre-tagging sulla collezione '{collection_name}' con {len(candidate_tags)} tag")

        # Per barra caricamento (tqdm)
        total_count = None
        try:
            total_count = self.qdrant_client.count(collection_name=collection_name).count
            if max_chunks is not None:
                total_count = min(max_chunks, total_count)
        except Exception as e:
            logger.warning(f"!!!> Impossibile recuperare il conteggio totale dei punti da Qdrant: {e}") 

        pbar = tqdm(total=total_count, desc=f"Pre-tagging '{collection_name}'", unit="chunk")

        next_offset = start_offset # None se il parametro non è passato
        total_processed = 0
        idx_global = 0

        while True:
            # Calcolo quanti elementi chiedere nel batch per non superare il max_count
            current_limit = batch_size
            if max_chunks is not None:
                remaining = max_chunks - total_processed
                if remaining <= 0:
                    logger.info(f">> Pre-Taggato l'insieme passato in input")
                    break
                current_limit = min(batch_size, remaining)

            # Paginazione su Qdrant
            records, next_offset = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=current_limit,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )

            if not records:
                logger.info(f">> No records disponibili")
                break

            # Accumulo dei tag assegnati per l'intero batch
            #  Invece di scrivere per ogni chunk, così si ottimizza l'operazione
            page_tags: Dict[str, List[str]] = {}

            for rec in records:
                try:
                    payload = rec.payload or {}
                    chunk_vector = rec.vector

                    # Estrazione ID
                    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else payload

                    # Estrazione sicura con fallback protetti da 'or' (evita il bug del valore None)
                    doc_id = payload.get("doc_id") or meta.get("doc_id") or "doc"

                    raw_idx = payload.get("chunk_index") if payload.get("chunk_index") is not None else meta.get("chunk_index")
                    chunk_idx = raw_idx if raw_idx is not None else idx_global

                    chunk_id = payload.get("chunk_id") or meta.get("chunk_id") or f"{doc_id}_chunk_{chunk_idx}"

                    # Estrazione del testo del chunk
                    chunk_text = payload.get("text") or payload.get("page_content") or meta.get("page_content") or ""

                    if chunk_text:
                        # Assegnazione automatica dei tag
                        assigned_tags = self.tag_assigner.assign_tags(
                            text=chunk_text,
                            candidate_tags=candidate_tags,
                            vector=chunk_vector
                        )

                        # Salvataggio in batch dei tag aggiunti
                        if assigned_tags:
                            page_tags[chunk_id] = assigned_tags

                except Exception as e:
                    # Se un record specifico fallisce, stampiamo l'errore ed continuiamo col successivo!
                    logger.error(f"!!!> Errore/Blocco sul record all'indice {idx_global} (ID Qdrant: {getattr(rec, 'id', 'sconosciuto')}): {e}")

                total_processed += 1
                idx_global += 1
                pbar.update(1)

            if page_tags:
                try:
                    self.sidecar_manager.add_tag_overrides_batch(page_tags)
                except Exception as e:
                    tqdm.write(f"\n!!!> [ERRORE SCRITTURA SIDECAR] Fallita scrittura batch al chunk {idx_global}: {e}")

                page_tags.clear() # Svuota il dizionario per liberare memoria

            # Libera la memoria e rallenta il processo per permettere il flush dell'I/O
            gc.collect()
            time.sleep(0.05)

            if next_offset is None:
                break

        pbar.close()
        logger.info(f">> Pre-tagging completato con successo su {total_processed} chunk.")

    def close(self) -> None:
        """
        Rilascia le risorse create da questa istanza. Le risorse iniettate
        dall'esterno (sidecar_manager, qdrant_client) restano responsabilità
        di chi le ha create.
        """
        if hasattr(self, "sidecar_manager"):
            if getattr(self, "_owns_sidecar_manager", False):
                self.sidecar_manager.release_path()
            del self.sidecar_manager

        if hasattr(self, "tag_assigner"):
            del self.tag_assigner

        if hasattr(self, "qdrant_client"):
            if getattr(self, "_owns_qdrant_client", False) and hasattr(self.qdrant_client, "close"):
                try:
                    self.qdrant_client.close()
                except Exception as e:
                    logger.warning(f"!!!> Errore durante la chiusura del client Qdrant: {e}")
            del self.qdrant_client

        logger.info("Istanze interne di PreTagger eliminate dalla memoria.")
