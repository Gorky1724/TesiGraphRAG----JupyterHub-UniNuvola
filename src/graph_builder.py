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

    def add_chunk_node(self, chunk_id: str, text: str, tags: list = None):
        self.graph.add_node(chunk_id, text=text, tags=tags or [])

    def add_relation(self, source_id: str, target_id: str, weight: float = 1.0):
        self.graph.add_edge(source_id, target_id, weight=weight)

    def auto_connect_nodes(self, records: list):
        """
        Calcola la Cosine Similarity tra tutte le coppie di record e
        crea un arco se la similarità supera la soglia di limite
        """
        def cosine_sim(v1, v2):
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(v1, v2) / (norm1 * norm2))

        n = len(records)
        for i in range(n):
            for j in range(i+1, n):
                v1, v2 = records[i].vector, records[j].vector
                sim = float(cosine_sim(v1,v2))

                if sim >= SIMILARITY_THRESHOLD:
                    self.add_relation(
                        source_id=str(records[i].id),
                        target_id=str(records[j].id),
                        weight=sim
                    )

    def to_json_data(self) -> dict:
        data = nx.node_link_data(self.graph)
        if "edges" in data and "links" not in data:
            data["links"] = data.pop("edges")
        return data
