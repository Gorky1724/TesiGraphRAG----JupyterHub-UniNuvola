import pathlib
import anywidget
import traitlets

_FRONTEND_DIR = pathlib.Path(__file__).parent / "frontend"

class ChunkGraphWidget(anywidget.AnyWidget):
    _esm = _FRONTEND_DIR / "chunk_graph.js"

    graph_data = traitlets.Dict({"nodes": [], "links": []}).tag(sync=True)
    selected_tag = traitlets.Dict({}).tag(sync=True)
