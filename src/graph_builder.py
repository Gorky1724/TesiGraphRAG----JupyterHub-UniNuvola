import networkx as nx

class KnowledgeGraphBuilder:
    def __init__(self):
        self.graph = nx.Graph()

    def add_chunk_node(self, chunk_id: str, text: str, tags: list = None):
        self.graph.add_node(chunk_id, text=text, tags=tags or [])

    def add_relation(self, source_id: str, target_id: str, weight: float = 1.0):
        self.graph.add_edge(source_id, target_id, weight=weight)

    def to_json_data(self) -> dict:
        data = nx.node_link_data(self.graph)
        if "edges" in data and "links" not in data:
            data["links"] = data.pop("edges")
        return data
