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
                json.dump({"pairwise_deltas": {}, "tag_overrides": {}}, f, indent=2)

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
                return {"pairwise_deltas": {}, "tag_overrides": {}}
        return {"pairwise_deltas": {}, "tag_overrides": {}}

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

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"<<| Modifica salvata per la coppia [{pair_key}]: factor={distance_factor:.2f} |>>")

    def reset_all(self):
        """Ripristina il file sidecar azzerando le modifiche (UNDO globale)."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump({"pairwise_deltas": {}, "tag_overrides": {}}, f, indent=2)
