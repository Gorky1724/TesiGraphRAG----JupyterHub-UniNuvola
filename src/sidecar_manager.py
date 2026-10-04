import sys
import json
import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.config import DISTANCE_CLASS_FACTORS

class SidecarManager:
    """
    Istanziare una singola volta per filepath (.json).
    La stessa accortezza va realizzata non solo nello stesso notebook,
    ma anche tra notebook DIVERSI.
    Evitare cioè modifiche concorrenti, visto che non c'è nessun lock che le impedisca.
    Nello stesso notebook c'è un WARNING, ma tra notebook diversi non c'è avviso.
    """

    # Tiene traccia dei path per cui è stata creata un istanza.
    #  L'utilizzo di una cache infatti rende problematica la presenza di più istanze:
    #  ciascuna avrebbe una propria cache in RAM indipendente con il rischio che il loro contenuto
    #  diverga senza avvisi.
    #  Un implementazione singleton avrebbe complicato molto il codice, invece tramite
    #  questa accortezza se si istanzia una seconda istanza viene stampato un avviso (non bloccante)
    _seen_paths: set = set()

    DEFAULT_PALETTE = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"
    ] # Colori di Default se non ne vengono assegnati altri

    def __init__(self, filepath="sidecar_edits.json"):
        # Inizializzazione di default; modificabile passandogli un diverso path come parametro
        self.filepath = (
            Path(filepath).expanduser().resolve()
            if filepath else Path("sidecar_edits.json").resolve()
        )

        # Verifica istanziazione singola
        if self.filepath in SidecarManager._seen_paths:
            print(
                f"[WARNING] Creata una seconda istanza di SidecarManager per lo stesso file "
                f"({self.filepath}). Ogni istanza mantiene una propria cache in RAM: se non e' "
                f"la stessa istanza ad essere condivisa e riutilizzata ovunque, le due cache "
                f"possono disallinearsi silenziosamente. Crea una sola istanza e passala "
                f"esplicitamente a tutti i componenti che leggono/scrivono il sidecar."
            )
        SidecarManager._seen_paths.add(self.filepath)

        # Crea la cache
        self._cache: Optional[Dict[str, Any]] = None

        self._ensure_file_exists()

    ### Accesso e gestione dati e cache

    def _ensure_file_exists(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self.reset_all()

    @property
    def data(self) -> dict:
        """Accesso ai dati aggiornati forzando automaticamente il reload"""
        return self.load_data(force_reload=False)

    def load_data(self, force_reload: bool=False) -> dict:
        """
        Carica i dati del sidecaar.
        Sfrutta la cache in RAM per evitare I/O ripetuto e eventuale collo di bottiglia.
        Restituisce sempre la deepcopy, così eventuali mutazioni da parte del chiamante
        (prima di un save_data) non sporcano la cache_interna.
        """
        if self._cache is not None and not force_reload:
            return copy.deepcopy(self._cache)

        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
                return copy.deepcopy(self._cache)
            except Exception as e:
                # Il file esiste ma non è leggibile (es. JSON corrotto per
                # un'interruzione a metà scrittura). Prima di scartarlo ne
                # salviamo una copia: la prossima save_data() sovrascriverà
                # il file originale, quindi senza backup il contenuto
                # eventualmente ancora recuperabile andrebbe perso per sempre.
                backup_path = self.filepath.with_suffix(f".corrotto.{int(time.time())}.json")
                try:
                    self.filepath.replace(backup_path)
                    print(f"<[WARNING]> File sidecar illeggibile ({e}). "
                          f"Backup del file originale salvato in: {backup_path}")
                except Exception:
                    print(f"<[WARNING]> File sidecar illeggibile ({e}) e impossibile crearne un backup.")


        default_structure = {
            "modified_adjacencies": {},
            "tag_overrides": {},
            "global_tags": {},
            "predetermined_queries": {}
        }
        self._cache = default_structure
        return copy.deepcopy(self._cache)

    def save_data(self, data: dict):
        """Salva il dizionario su disco in modo atomico e aggiorna la cache in RAM."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Scrittura atomica: si scrive prima su un file temporaneo e poi lo
        # si sostituisce al file definitivo. Path.replace (os.replace) e'
        # atomico su POSIX, quindi anche un'interruzione a meta' (crash,
        # restart del kernel) non puo' lasciare sidecar_edits.json in uno
        # stato troncato/corrotto: resta o la vecchia versione, o gia'
        # completa quella nuova.
        tmp_path = self.filepath.with_suffix(self.filepath.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(self.filepath)

        self._cache = copy.deepcopy(data)

    def invalidate_cache(self):
        """Forza la rilettura da filesystem alla successiva chiamata di load_data()."""
        self._cache = None

    def release_path(self) -> None:
        """
        Rimuove il path di questa istanza dal registro delle istanze viste.
        Da chiamare quando l'istanza non verrà più usata (es. in PreTagger.close()),
        così una nuova SidecarManager sullo stesso file non genera un falso avviso
        di "seconda istanza". NON funziona tra notebook diversi.
        """
        SidecarManager._seen_paths.discard(self.filepath)

    def get_global_tags(self) -> dict:
        """
        Restituisce il dizionario dei tag attivi, con contatore e colore corrente.
        """
        data = self.load_data()
        return data.get("global_tags", {})

    ### Manipolazione Classi di Distanza
    def save_pairwise_class_edits(self, chunk_id_1: str, chunk_id_2: str, distance_class: Optional[str]):
        """
        Salva o aggiorna la classe di distanza tra una coppia di chunk.
        Garantisce simmetria (chunk_1 <-> chunk_2) e pulizia automatica se la classe
        dovesse essere "INVARIATI" o non presente in DISTANCE_CLASS_FACTORS.
        Reindirizza automaticamente a save_predetermined_query_adjacency se uno degli ID riguarda una query pre-determinata
        """
        c1, c2 = str(chunk_id_1), str(chunk_id_2)
        if c1 == c2:
            return

        # Instradamento automatico verso le query pre-determinate
        if c1.startswith("predq_") or c2.startswith("predq_"):
            self.save_predetermined_query_adjacency(c1, c2, distance_class)
            return

        data = self.load_data()
        adj = data.setdefault("modified_adjacencies", {})

        norm_class = str(distance_class).upper().strip() if distance_class is not None else "INVARIATI"
        if norm_class not in DISTANCE_CLASS_FACTORS:
            norm_class = "INVARIATI"

        if norm_class == "INVARIATI":
            if c1 in adj and c2 in adj[c1]:
                del adj[c1][c2]
                if not adj[c1]:
                    del adj[c1]

            if c2 in adj and c1 in adj[c2]:
                del adj[c2][c1]
                if not adj[c2]:
                    del adj[c2]

            print(f"<<| Adiacenza rimossa (ripristino a INVARIATI) tra [{c1}] e [{c2}] |>>")
        else:
            adj.setdefault(c1, {})[c2] = {"distance_class": norm_class}
            adj.setdefault(c2, {})[c1] = {"distance_class": norm_class}
            print(f"<<| Modifica adiacenza salvata tra [{c1}] e [{c2}]: classe={norm_class} |>>")

        self.save_data(data)

    def save_pairwise_class_edits_batch(self, edits: List[Dict[str, Any]]):
        """
        Salva un blocco di modifiche adiacenze in un'unica operazione di I/O.
        Ogni voce di 'edits' deve contenere: 'chunk_1', 'chunk_2', 'distance_class'.
        Smista automaticamente gli elementi coinvolgenti 'predq_' al relativo batch dedicato.
        """
        if not edits or not isinstance(edits, list):
            return

        standard_edits = []
        predq_edits = []

        for edit in edits:
            if edit and "chunk_1" in edit and "chunk_2" in edit:
                c1, c2 = str(edit["chunk_1"]), str(edit["chunk_2"])
                if c1.startswith("predq_") or c2.startswith("predq_"):
                    predq_edits.append(edit)
                else:
                    standard_edits.append(edit)

        # Esegue prima l'eventuale sotto-batch per le query pre-determinate
        if predq_edits:
            self.save_predetermined_query_adjacencies_batch(predq_edits)

        if not standard_edits:
            return

        data = self.load_data()
        adj = data.setdefault("modified_adjacencies", {})
        updated = False

        for edit in standard_edits:
            c1, c2 = str(edit["chunk_1"]), str(edit["chunk_2"])
            if c1 == c2:
                continue
            d_cls = edit.get("distance_class")
            norm_class = str(d_cls).upper().strip() if d_cls is not None else "INVARIATI"

            if norm_class not in DISTANCE_CLASS_FACTORS:
                norm_class = "INVARIATI"

            if norm_class == "INVARIATI":
                if c1 in adj and c2 in adj[c1]:
                    del adj[c1][c2]
                    if not adj[c1]:
                        del adj[c1]
                if c2 in adj and c1 in adj[c2]:
                    del adj[c2][c1]
                    if not adj[c2]:
                        del adj[c2]
            else:
                adj.setdefault(c1, {})[c2] = {"distance_class": norm_class}
                adj.setdefault(c2, {})[c1] = {"distance_class": norm_class}
            updated = True

        if updated:
            self.save_data(data)
            print(f"<<| Batch adiacenze standard salvato: {len(standard_edits)} elementi processati |>>")
    # Estrazione adiacenze
    def get_chunk_adjacencies(self, chunk_id: str) -> Dict[str, Dict[str, str]]:
        """
        Restituisce il dizionario delle adiacenze modificate per un singolo chunk.
        es. return format: {'chunk_2': {'distance_class': 'AVVICINATI'}}
        """
        data = self.load_data()
        adj = data.get("modified_adjacencies", {})
        return adj.get(str(chunk_id), {})

    def get_all_modified_adjacencies(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Restituisce l'intero dizionario delle adiacenze modificate."""
        data = self.load_data()
        return data.get("modified_adjacencies", {})

    ### Manipolazione TAG
    def add_tag_override(self, chunk_id: str, tag: str, color: Optional[str] = None):
        """
        Aggiunge un singolo tag a un chunk o a una query pre-determinata.
        Aggiorna global_tags SOLO per i chunk standard del database.
        """
        if not tag:
            return
        self.add_tag_override_for_chunk(chunk_id, [tag])

    def add_tag_override_for_chunk(self, chunk_id: str, tags: List[str]):
        """
        Assegna una lista di tag a un singolo chunk o a una query pre-determinata, aggiornando il registro globale
        ed eseguendo un uica operazione di scrittura su disco per l'intero chunk.
        Da utilizzare ad esempio per il pre-tagging, ottimizzando così l'operazione.
        Usato anche nel singolo tag_override per centralizzare l'operazione.
        Aggiorna global_tags SOLO per i chunk del database.
        """
        if not tags:
            return

        data = self.load_data()
        chunk_id = str(chunk_id)

        # Gestione Query Pre-Determinata (NON tocca global_tags)
        #  I tag graficamente aggiunti, sono considerati assigned_tags; conv_tags e auto_tags non possono essere manipolati.
        if chunk_id.startswith("predq_"):
            pred_queries = data.setdefault("predetermined_queries", {})
            query_entry = pred_queries.setdefault(chunk_id, {})
            current_tags = query_entry.setdefault("graphically_assigned_tags", [])

            updated = False
            for t in tags:
                if t not in current_tags:
                    current_tags.append(t)
                    updated = True

            if updated:
                self.save_data(data)
            return

        # Gestione Chunk Standard del Database
        if "global_tags" not in data:
            data["global_tags"] = {}
        if "tag_overrides" not in data:
            data["tag_overrides"] = {}

        if chunk_id not in data["tag_overrides"]:
            data["tag_overrides"][chunk_id] = {"user_tags": []}

        current_tags = data["tag_overrides"][chunk_id].setdefault("user_tags", [])

        updated = False
        for t in tags:
            if t not in current_tags:
                current_tags.append(t)
                updated = True

                if t not in data["global_tags"]:
                    used_colors = {
                        t_info.get("color") for t_info in data["global_tags"].values()
                        if isinstance(t_info, dict)
                    }
                    available = [c for c in self.DEFAULT_PALETTE if c not in used_colors]
                    col = (
                        available[0] if available
                        else self.DEFAULT_PALETTE[len(data["global_tags"]) % len(self.DEFAULT_PALETTE)]
                    )
                    data["global_tags"][t] = {"count": 1, "color": col}
                else:
                    data["global_tags"][t]["count"] += 1

        if updated:
            self.save_data(data)

    def add_tag_overrides_batch(self, chunk_tags_map: Dict[str, List[str]]) -> None:
        """
        Assegna tag a più chunk in un'unica operazione di lettura+scrittura su disco.
        chunk_tags_map: {chunk_id: [tag1, tag2, ...], ...}
        """
        if not chunk_tags_map:
            return

        data = self.load_data()
        if "tag_overrides" not in data:
            data["tag_overrides"] = {}
        if "global_tags" not in data:
            data["global_tags"] = {}

        updated = False

        for chunk_id, tags in chunk_tags_map.items():
            if not tags:
                continue

            cid_str = str(chunk_id)

            # Ramo Query Pre-Determinata
            if cid_str.startswith("predq_"):
                pred_queries = data.setdefault("predetermined_queries", {})
                query_entry = pred_queries.setdefault(cid_str, {})
                current_tags = query_entry.setdefault("graphically_assigned_tags", [])

                for t in tags:
                    if t not in current_tags:
                        current_tags.append(t)
                        updated = True
                continue

            # Ramo Chunk Standard
            if cid_str not in data["tag_overrides"]:
                data["tag_overrides"][cid_str] = {"user_tags": []}

            current_tags = data["tag_overrides"][cid_str].setdefault("user_tags", [])

            for t in tags:
                if t not in current_tags:
                    current_tags.append(t)
                    updated = True

                    if t not in data["global_tags"]:
                        used_colors = {
                            t_info.get("color") for t_info in data["global_tags"].values()
                            if isinstance(t_info, dict)
                        }
                        available = [c for c in self.DEFAULT_PALETTE if c not in used_colors]
                        col = (
                            available[0] if available
                            else self.DEFAULT_PALETTE[len(data["global_tags"]) % len(self.DEFAULT_PALETTE)]
                        )
                        data["global_tags"][t] = {"count": 1, "color": col}
                    else:
                        data["global_tags"][t]["count"] += 1

        if updated:
            self.save_data(data)

    def remove_tag_override(self, chunk_id: str, tag: str):
        """
        Rimuove un tag da un chunk o da una query pre-determinata, decrementando il contatore globale.
        Se questo scende a 0, rimuove il tag da tale registro globale.
        Decrementa global_tags SOLO per i chunk del database.
        """
        data = self.load_data()
        chunk_id = str(chunk_id)

        # Rimozione da Query Pre-Determinata
        if chunk_id.startswith("predq_"):
            pred_queries = data.get("predetermined_queries", {})
            if chunk_id in pred_queries:
                current_tags = pred_queries[chunk_id].get("graphically_assigned_tags", [])
                if tag in current_tags:
                    current_tags.remove(tag)
                    self.save_data(data)
            return

        # Rimozione da Chunk Standard
        tag_removed = False
        if "tag_overrides" in data and chunk_id in data["tag_overrides"]:
            current_tags = data["tag_overrides"][chunk_id].get("user_tags", [])
            if tag in current_tags:
                current_tags.remove(tag)
                tag_removed = True
                if not current_tags:
                    del data["tag_overrides"][chunk_id]

        if tag_removed and "global_tags" in data and tag in data["global_tags"]:
            data["global_tags"][tag]["count"] -= 1
            if data["global_tags"][tag]["count"] <= 0:
                del data["global_tags"][tag]
            self.save_data(data)

    ### RESET file sidecar
    def reset_all(self):
        """Ripristina il file sidecar azzerando le modifiche (UNDO globale)."""
        self.save_data({
            "modified_adjacencies": {},
            "tag_overrides": {},
            "global_tags": {},
            "predetermined_queries": {}
        })
        print("<<! File sidecar ripristinato allo stato iniziale !>>")

    ### Gestione Query Pre-Determinate
    # Estrazione dati
    def get_predetermined_queries(self) -> Dict[str, Dict[str, Any]]:
        """Restituisce l'intero dizionario delle query pre-determinate salvate."""
        data = self.load_data()
        return data.get("predetermined_queries", {})

    def get_predetermined_query(self, pred_query_id: str) -> Dict[str, Any]:
        """Restituisce i dati e le adiacenze per una specifica query pre-determinata."""
        queries = self.get_predetermined_queries() or {}
        return queries.get(str(pred_query_id), {})

    def get_predetermined_query_adjacencies(self, pred_query_id: str) -> Dict[str, Dict[str, str]]:
        """
        Restituisce il dizionario delle adiacenze modificate per una specifica query pre-determinata.
        Format di ritorno:
        {'chunk_12': {'distance_class': 'AVVICINATI'}}
        """
        query_entry = self.get_predetermined_query(pred_query_id)
        return query_entry.get("adjacencies", {})

    # Salvataggio adiacenze
    def save_predetermined_query_adjacency(self, chunk_id_1: str, chunk_id_2: str, distance_class: Optional[str]):
        """
        Salva o aggiorna l'adiacenza tra una query pre-determinata (ID con prefisso 'predq_') e un chunk.
        Se la classe e' INVARIATI, rimuove la voce dal JSON.
        """
        c1, c2 = str(chunk_id_1), str(chunk_id_2)
        if c1 == c2:
            return

        if c1.startswith("predq_"):
            pred_id, target_chunk = c1, c2
        elif c2.startswith("predq_"):
            pred_id, target_chunk = c2, c1
        else:
            self.save_pairwise_class_edits(c1, c2, distance_class)
            return

        data = self.load_data()
        pred_queries = data.setdefault("predetermined_queries", {})
        query_entry = pred_queries.setdefault(pred_id, {"adjacencies": {}, "graphically_assigned_tags": []})
        adj = query_entry.setdefault("adjacencies", {})

        norm_class = str(distance_class).upper().strip() if distance_class is not None else "INVARIATI"
        if norm_class not in DISTANCE_CLASS_FACTORS:
            norm_class = "INVARIATI"

        if norm_class == "INVARIATI":
            if target_chunk in adj:
                del adj[target_chunk]
            print(f"<<| Adiacenza query pre-determinata rimossa (INVARIATI) tra [{pred_id}] e [{target_chunk}] |>>")
        else:
            adj[target_chunk] = {"distance_class": norm_class}
            print(f"<<| Modifica adiacenza query pre-determinata salvata tra [{pred_id}] e [{target_chunk}]: classe={norm_class} |>>")

        self.save_data(data)

    def save_predetermined_query_adjacencies_batch(self, edits: List[Dict[str, Any]]):
        """Salva un blocco di modifiche di adiacenza per query pre-determinate in un'unica operazione di I/O."""
        if not edits or not isinstance(edits, list):
            return

        data = self.load_data()
        pred_queries = data.setdefault("predetermined_queries", {})
        updated = False

        for edit in edits:
            if not edit or "chunk_1" not in edit or "chunk_2" not in edit:
                continue
            c1, c2 = str(edit["chunk_1"]), str(edit["chunk_2"])
            if c1 == c2:
                continue

            if c1.startswith("predq_"):
                pred_id, target_chunk = c1, c2
            elif c2.startswith("predq_"):
                pred_id, target_chunk = c2, c1
            else:
                continue

            query_entry = pred_queries.setdefault(pred_id, {"adjacencies": {}, "graphically_assigned_tags": []})
            adj = query_entry.setdefault("adjacencies", {})

            d_cls = edit.get("distance_class")
            norm_class = str(d_cls).upper().strip() if d_cls is not None else "INVARIATI"
            if norm_class not in DISTANCE_CLASS_FACTORS:
                norm_class = "INVARIATI"

            if norm_class == "INVARIATI":
                if target_chunk in adj:
                    del adj[target_chunk]
            else:
                adj[target_chunk] = {"distance_class": norm_class}
            updated = True

        if updated:
            self.save_data(data)
            print(f"<<| Batch adiacenze query pre-determinate salvato: {len(edits)} elementi processati |>>")

    ### Recupero generalizzato TAG e ADIACENZE
    def get_node_tags(self, node_id: str) -> list:
        """Restituisce i tag di un nodo gestendo sia i chunk sia le predq_."""
        data = self.data
        node_id = str(node_id)
        if node_id.startswith("predq_"):
            q_info = data.get("predetermined_queries", {}).get(node_id, {})
            return q_info.get("graphically_assigned_tags", [])
        else:
            return data.get("tag_overrides", {}).get(node_id, {}).get("user_tags", [])

