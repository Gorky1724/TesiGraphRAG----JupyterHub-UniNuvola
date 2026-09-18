import sys
from pathlib import Path
import networkx as nx
import numpy as np
project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.config import SIMILARITY_THRESHOLD

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()
        self._vectors = {}

    def add_chunk_node(self, chunk_id: str, text: str, vector: list = None, tags: list = None):
        """Aggiunge un nodo al grafo e ne memorizza opzionalmente il vettore di embedding."""
        self.graph.add_node(chunk_id, text=text, tags=tags or [])
        if vector is not None:
            self._vectors[chunk_id] = np.array(vector, dtype=float)

    def add_relation(self, source_id: str, target_id: str, weight: float = 1.0):
        """Aggiunge un arco tra due nodi con un peso=similarità."""
        self.graph.add_edge(source_id, target_id, weight=weight)

    def auto_connect_nodes(self, records: list=None):
        """
        Calcola la Cosine Similarity tra tutte le coppie di nodi e crea un arco se supera SIMILARITY_THRESHOLD.

        - Se 'records' è None (PREDILETTO): usa self._vectors memorizzati internamente.
        - Se 'records' è fornito (LEGACY): usa la lista esterna di oggetti con attributi .id e .vector.
        """
        def cosine_sim(v1, v2):
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))

        # Modalità LEGACY con lista esterna
        if records is not None:
            n = len(records)
            for i in range(n):
                for j in range(i + 1, n):
                    v1, v2 = records[i].vector, records[j].vector
                    sim = float(cosine_sim(v1, v2))
                    if sim >= SIMILARITY_THRESHOLD:
                        self.add_relation(
                            source_id=str(records[i].id),
                            target_id=str(records[j].id),
                            weight=sim
                        )
        else: # Modalità PREDILETTA con dizionario interno
            items = list(self._vectors.items())
            n = len(items)
            for i in range(n):
                id1, v1 = items[i]
                for j in range(i + 1, n):
                    id2, v2 = items[j]
                    sim = float(cosine_sim(v1, v2))
                    if sim >= SIMILARITY_THRESHOLD:
                        self.add_relation(
                            source_id=id1,
                            target_id=id2,
                            weight=sim
                        )

    def to_json_data(self) -> dict:
        data = nx.node_link_data(self.graph)
        if "edges" in data and "links" not in data:
            data["links"] = data.pop("edges")
        return data
