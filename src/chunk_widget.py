import sys
from pathlib import Path
import anywidget
import traitlets

project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.sidecar_manager import SidecarManager
from src.config import HARD_FILTERING_THRESHOLD

class ChunkGraphWidget(anywidget.AnyWidget):
    _esm = Path(__file__).parent / "frontend" / "chunk_graph.js"

    graph_data = traitlets.Dict({"nodes": [], "links": []}).tag(sync=True)
    selected_tag = traitlets.Dict({}).tag(sync=True)
    pairwise_edit = traitlets.Dict({}).tag(sync=True)

    def __init__(self, sidecar_path=None, **kwargs):
        super().__init__(**kwargs)
        self.sidecar = SidecarManager(filepath=sidecar_path) if sidecar_path else SidecarManager()
        self.observe(self._on_pairwise_edit, names=["pairwise_edit"])

    def load_graph(self, raw_graph_data: dict):
        """
        Carica la struttura del grafo applicando i distance_factor registrati nel sidecar
        e iniettando la soglia hard_filter_threshold.
        """
        deltas = self.sidecar.data.get("pairwise_deltas", {})
        links = raw_graph_data.get("links", raw_graph_data.get("edges", []))
        enriched_links = []

        for link in links:
            l_copy = dict(link)
            src = l_copy["source"]["id"] if isinstance(l_copy["source"], dict) else l_copy["source"]
            tgt = l_copy["target"]["id"] if isinstance(l_copy["target"], dict) else l_copy["target"]

            k1 = f"{src}_AND_{tgt}"
            k2 = f"{tgt}_AND_{src}"

            factor = 1.0
            if k1 in deltas:
                factor = float(deltas[k1].get("distance_factor", 1.0))
            elif k2 in deltas:
                factor = float(deltas[k2].get("distance_factor", 1.0))

            l_copy["distance_factor"] = factor
            enriched_links.append(l_copy)

        self.graph_data = {
            "nodes": raw_graph_data.get("nodes", []),
            "links": enriched_links,
            "hard_filter_threshold": raw_graph_data.get("hard_filter_threshold", HARD_FILTERING_THRESHOLD)
        }

    def _on_pairwise_edit(self, change):
        edit = change["new"]
        if edit and "chunk_1" in edit and "chunk_2" in edit:
            self.sidecar.save_pairwise_delta(
                chunk_id_1=edit["chunk_1"],
                chunk_id_2=edit["chunk_2"],
                distance_factor=edit["distance_factor"]
            )
