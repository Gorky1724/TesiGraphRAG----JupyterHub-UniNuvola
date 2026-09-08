import networkx as nx

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def build_graph(self, chunks, similarity_pairs):
        """Popola il grafo NetworkX con nodi (chunk) ed archi (similarità/pesi)."""
        self.graph.clear()
        
        for chunk in chunks:
            chunk_id = getattr(chunk, 'id', str(chunk))
            chunk_text = getattr(chunk, 'text', str(chunk))
            self.graph.add_node(chunk_id, text=chunk_text, tags=[])

        for a, b, score in similarity_pairs:
            self.graph.add_edge(a, b, weight=score)

    def to_json_data(self) -> dict:
        """Esporta la struttura del grafo nel formato JSON per D3.js."""
        return nx.node_link_data(self.graph)
