import json
from pathlib import Path

class SidecarManager:
    def __init__(self, filepath="sidecar_edits.json"):
        # Inizializzazione di default; modificabile passandogli un diverso path come parametro
        self.filepath = Path(filepath) if filepath else Path("sidecar_edits.json")
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        if not self.filepath.exists():
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "pairwise_deltas": {},
                    "tag_overrides": {},
                    "global_tags": {}}, # Registro globale tag
                          f, indent=2)

    ### Accesso dati
    @property
    def data(self) -> dict:
        """Garantisce l'accesso diretto ai dati leggendoli sempre aggiornati dal file."""
        return self.load_data()

    def load_data(self) -> dict:
        if self.filepath.exists():
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"pairwise_deltas": {}, "tag_overrides": {}, "global_tags": {}}
        return {"pairwise_deltas": {}, "tag_overrides": {}, "global_tags": {}}

    def save_data(self, data: dict):
         with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

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
                    used_colors = {t_info.get("color") for t_info in data["global_tags"].values()}
                    available = [c for c in self.DEFAULT_PALETTE if c not in used_colors]
                    color = available[0] if available else self.DEFAULT_PALETTE[len(data["global_tags"]) % len(self.DEFAULT_PALETTE)] # Gestione ciclica array
                data["global_tags"][tag] = {"count": 1, "color": color}
            else:
                data["global_tags"][tag]["count"] += 1
                if color: # Aggiornamento colore se esplicitamente fornito
                    data["global_tags"][tag]["color"] = color

            self.save_data(data)
            print(f"<<! Tag '{tag}' ({data['global_tags'][tag]['color']}) aggiunto a [{chunk_id}]. Conteggio globale: {data['global_tags'][tag]['count']}")

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
                    if data["global_tags"][tag]["count"] == 0:
                        del data["global_tags"][tag]
                        print(f"X>> Tag '{tag}' rimosso definitivamente dal registro globale (conteggio = 0).")
                    else:
                        print(f"<<| Tag '{tag}' rimosso da [{chunk_id}]. Conteggio residuo: {data['global_tags'][tag]['count']}")

                self.save_data(data)

    ### RESET file sidecar
    def reset_all(self):
        """Ripristina il file sidecar azzerando le modifiche (UNDO globale)."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({
                    "pairwise_deltas": {},
                    "tag_overrides": {},
                    "global_tags": {}},
                          f, indent=2)
        print("<<! File sidecar ripristinato allo stato iniziale !>>")
