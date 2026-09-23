import json
import copy
import time
from pathlib import Path
from typing import Any, Dict, Optional, List


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

    def _ensure_file_exists(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            self.reset_all()

    ### Accesso e gestione dati e cache
    @property
    def data(self) -> dict:
        """Garantisce l'accesso diretto ai dati leggendoli sempre aggiornati dal file."""
        return self.load_data()

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
            "pairwise_deltas": {},
            "tag_overrides": {},
            "global_tags": {}
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

    ### Manipolazione Distanze
    def save_pairwise_delta(self, chunk_id_1: str, chunk_id_2: str, distance_factor: float):
        """
        Salva o aggiorna il fattore di distanza tra una coppia di chunk.
        Garantisce la simmetria della relazione (A_B == B_A).
        """
        data = self.load_data()

        # Ordiniamo gli ID per garantire che la relazione sia simmetrica
        pair_key = "_AND_".join(sorted([str(chunk_id_1), str(chunk_id_2)]))

        if "pairwise_deltas" not in data:
            data["pairwise_deltas"] = {}

        data["pairwise_deltas"][pair_key] = {
            "chunk_1": str(chunk_id_1),
            "chunk_2": str(chunk_id_2),
            "distance_factor": round(distance_factor, 3)
        }

        self.save_data(data)
        print(f"<<| Modifica salvata per la coppia [{pair_key}]: factor={distance_factor:.2f} |>>")

    def save_pairwise_deltas_batch(self, edits: list):
        """
        Salva un blocco di modifiche pairwise in un unica operazione I/=
        """
        if not edits or not isinstance(edits, list):
            return

        data = self.load_data()
        if "pairwise_deltas" not in data:
            data["pairwise_deltas"] = {}

        updated = False
        for edit in edits:
            if edit and "chunk_1" in edit and "chunk_2" in edit:
                c1, c2 = str(edit["chunk_1"]), str(edit["chunk_2"])
                pair_key = "_AND_".join(sorted([c1, c2]))
                data["pairwise_deltas"][pair_key] = {
                    "chunk_1": c1,
                    "chunk_2": c2,
                    "distance_factor": round(float(edit["distance_factor"]), 3)
                }
                updated = True

        if updated:
            self.save_data(data)
            print(f"<<| Batch salvato: {len(edits)} distanze aggiornate nel sidecar |>>")


    ### Manipolazione TAG
    DEFAULT_PALETTE = [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"
    ] # Colori di Default se non ne vengono assegnati altri

    def add_tag_override(self, chunk_id: str, tag: str, color: str=None):
        """
        Aggiunge un tag a un chunk, assegna un colore e aggiorna il registro globale
        """
        data = self.load_data()
        chunk_id = str(chunk_id)

        if "tag_overrides" not in data:
            data["tag_overrides"] = {}
        if "global_tags" not in data:
            data["global_tags"] = {}

        if chunk_id not in data["tag_overrides"]:
            data["tag_overrides"][chunk_id] = {"user_tags": []}

        current_tags = data["tag_overrides"][chunk_id].get("user_tags", [])

        # Aggiunge il tag solo se non già presente
        if tag not in current_tags:
            current_tags.append(tag)
            data["tag_overrides"][chunk_id]["user_tags"] = current_tags

            # Aggiorna il registro globale
            if tag not in data["global_tags"]:
                # Se color=None si assegna un colore tra quelli di default
                if not color:
                    used_colors = {
                        t_info.get("color")
                        for t_info in data["global_tags"].values()
                        if isinstance(t_info, dict)
                    }
                    available = [c for c in self.DEFAULT_PALETTE if c not in used_colors]
                    color = (
                        available[0]
                        if available
                        else self.DEFAULT_PALETTE[len(data["global_tags"]) % len(self.DEFAULT_PALETTE)]
                    ) # Gestione ciclica array
                data["global_tags"][tag] = {"count": 1, "color": color}
            else:
                data["global_tags"][tag]["count"] += 1
                if color: # Aggiornamento colore se esplicitamente fornito
                    data["global_tags"][tag]["color"] = color

            self.save_data(data)
            print(f"<<! Tag '{tag}' ({data['global_tags'][tag]['color']}) aggiunto a [{chunk_id}]. Conteggio globale: {data['global_tags'][tag]['count']}")

    def add_tag_override_for_chunk(self, chunk_id: str, tags: List[str]):
        """
        Assegna una lista di tag a un singolo chunk, aggiornando il registro globale
        ed eseguendo un uica operazione di scrittura su disco per l'intero chunk.
        Da utilizzare ad esempio per il pre-tagging, ottimizzando così l'operazione.
        """
        if not tags:
            return

        data = self.load_data()
        chunk_id = str(chunk_id)

        if "tag_overrides" not in data:
            data["tag_overrides"] = {}
        if "global_tags" not in data:
            data["global_tags"] = {}

        if chunk_id not in data["tag_overrides"]:
            data["tag_overrides"][chunk_id] = {"user_tags": []}

        current_tags = data["tag_overrides"][chunk_id].get("user_tags", [])
        updated = False

        for tag in tags:
            if tag not in current_tags:
                current_tags.append(tag)
                updated = True

                if tag not in data["global_tags"]:
                    used_colors = {
                        t_info.get("color")
                        for t_info in data["global_tags"].values()
                        if isinstance(t_info, dict)
                    }
                    available = [c for c in self.DEFAULT_PALETTE if c not in used_colors]
                    color = (
                        available[0]
                        if available
                        else self.DEFAULT_PALETTE[len(data["global_tags"]) % len(self.DEFAULT_PALETTE)]
                    )
                    data["global_tags"][tag] = {"count": 1, "color": color}
                else:
                    data["global_tags"][tag]["count"] += 1

        if updated:
            data["tag_overrides"][chunk_id]["user_tags"] = current_tags
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

            chunk_id = str(chunk_id)

            if chunk_id not in data["tag_overrides"]:
                data["tag_overrides"][chunk_id] = {"user_tags": []}

            # Recupero sicuro di user_tags
            current_tags = data["tag_overrides"][chunk_id].setdefault("user_tags", [])

            for tag in tags:
                if tag not in current_tags:
                    current_tags.append(tag)
                    updated = True

                    if tag not in data["global_tags"]:
                        used_colors = {
                            t.get("color") for t in data["global_tags"].values()
                            if isinstance(t, dict)
                        }
                        available = [c for c in self.DEFAULT_PALETTE if c not in used_colors]
                        color = (
                            available[0] if available
                            else self.DEFAULT_PALETTE[len(data["global_tags"]) % len(self.DEFAULT_PALETTE)]
                        )
                        data["global_tags"][tag] = {"count": 1, "color": color}
                    else:
                        data["global_tags"][tag]["count"] += 1

        if updated:
            self.save_data(data)

    def remove_tag_override(self, chunk_id: str, tag: str):
        """
        Rimuove un tag da un chunk, decrementando il contatore globale.
        Se questo scende a 0, rimuove il tag da tale registro globale.
        """
        data = self.load_data()
        chunk_id = str(chunk_id)

        if "tag_overrides" in data and chunk_id in data["tag_overrides"]:
            current_tags = data["tag_overrides"][chunk_id].get("user_tags", [])
            if tag in current_tags:
                current_tags.remove(tag)
                data["tag_overrides"][chunk_id]["user_tags"] = current_tags

                # Rimuove la voce "tag_overrides" se non presente alcun tag
                if not current_tags:
                    del data["tag_overrides"][chunk_id]

                # Decrementa registro, rimuovendo la voce se necessario
                if "global_tags" in data and tag in data["global_tags"]:
                    data["global_tags"][tag]["count"] -= 1
                    if data["global_tags"][tag]["count"] <= 0:
                        del data["global_tags"][tag]
                        print(f"X>> Tag '{tag}' rimosso definitivamente dal registro globale (conteggio = 0).")
                    else:
                        print(f"<<| Tag '{tag}' rimosso da [{chunk_id}]. Conteggio residuo: {data['global_tags'][tag]['count']}")

                self.save_data(data)

    ### RESET file sidecar
    def reset_all(self):
        """Ripristina il file sidecar azzerando le modifiche (UNDO globale)."""
        self.save_data({
            "pairwise_deltas": {},
            "tag_overrides": {},
            "global_tags": {}
        })
        print("<<! File sidecar ripristinato allo stato iniziale !>>")
