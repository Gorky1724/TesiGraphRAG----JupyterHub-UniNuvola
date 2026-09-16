import pathlib
import anywidget
import traitlets
from sidecar_manager import SidecarManager

class ChunkGraphWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "frontend" / "chunk_graph.js"

    graph_data = traitlets.Dict({"nodes": [], "links": []}).tag(sync=True)
    selected_tag = traitlets.Dict({}).tag(sync=True)
    pairwise_edit = traitlets.Dict({}).tag(sync=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sidecar = SidecarManager()
        # Registra un listener per intercettare i trascinamenti da D3.js
        self.observe(self._on_pairwise_edit, names=["pairwise_edit"])

    def _on_pairwise_edit(self, change):
        edit = change["new"]
        if edit and "chunk_1" in edit and "chunk_2" in edit:
            self.sidecar.save_pairwise_delta(
                chunk_id_1=edit["chunk_1"],
                chunk_id_2=edit["chunk_2"],
                distance_factor=edit["distance_factor"]
            )
