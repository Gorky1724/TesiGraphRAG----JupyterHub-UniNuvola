import json
import os

class SidecarManager:
    def __init__(self, filepath="sidecar_edits.json"):
        self.filepath = filepath
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w") as f:
                json.dump({"pairwise_deltas": {}, "tag_overrides": {}}, f, indent=2)

    def load_data(self) -> dict:
        with open(self.filepath, "r") as f:
            return json.load(f)

    def save_pairwise_delta(self, chunk_id_1: str, chunk_id_2: str, distance_factor: float):
        """
        Salva o aggiorna il fattore di distanza tra una coppia di chunk.
        'chunk_id_1' e 'chunk_id_2' devono essere gli ID univoci (es. 'GraphRAG_Microsoft_chunk_0').
        """
        data = self.load_data()

        # Ordiniamo gli ID per garantire che la relazione sia simmetrica (A_B == B_A)
        pair_key = "_AND_".join(sorted([str(chunk_id_1), str(chunk_id_2)]))

        data["pairwise_deltas"][pair_key] = {
            "chunk_1": str(chunk_id_1),
            "chunk_2": str(chunk_id_2),
            "distance_factor": round(distance_factor, 3)
        }

        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"<<| Modifica salvata per la coppia [{pair_key}]: factor={distance_factor:.2f} |>>")

    def reset_all(self):
        """!!!>> Ripristina il file sidecar azzerando le modifiche (UNDO globale)."""
        with open(self.filepath, "w") as f:
            json.dump({"pairwise_deltas": {}, "tag_overrides": {}}, f, indent=2)
