import networkx as nx
from src.config import HARD_FILTERING_THRESHOLD, BASE_DISTANCE_PX

class GraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def add_chunk_node(self, chunk_id: str, text: str, metadata: dict = None):
        """Aggiunge un nodo rappresentante un chunk di testo."""
        attrs = {"text": text, "type": "chunk"}
        if metadata:
            attrs.update(metadata)
        self.graph.add_node(chunk_id, **attrs)

    def add_relation(self, source_id: str, target_id: str, similarity: float):
        """
        Crea un arco basato sulla Cosine Similarity calcolata da Qdrant.
        Calcola la distanza base fisica in pixel (D_base = BASE_DISTANCE_PX * (1 - similarity)).
        """
        if source_id in self.graph and target_id in self.graph:
            # Minima distanza consentita per evitare sovrapposizioni (clamp a 0.1)
            distance_scale = max(0.1, 1.0 - float(similarity))
            base_dist = BASE_DISTANCE_PX * distance_scale

            self.graph.add_edge(
                source_id,
                target_id,
                similarity=float(similarity),
                base_distance=round(base_dist, 2)
            )

    def to_json_data(self) -> dict:
        """Esporta il grafo in formato compatibile con D3.js ed inietta la soglia di config."""
        data = nx.node_link_data(self.graph)
        if "edges" in data and "links" not in data:
            data["links"] = data.pop("edges")

        data["hard_filter_threshold"] = HARD_FILTERING_THRESHOLD
        return data
