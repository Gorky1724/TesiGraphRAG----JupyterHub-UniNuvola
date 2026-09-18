import sys
import pathlib
from pathlib import Path
import anywidget
import traitlets
project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import BASE_GRAPH_DISTANCE
from sidecar_manager import SidecarManager

class ChunkGraphWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "frontend" / "chunk_graph.js"

    # Traitlets per stato del grafo
    graph_data = traitlets.Dict({"nodes": [], "links": []}).tag(sync=True)
    selected_tag = traitlets.Dict({}).tag(sync=True)
    global_tags = traitlets.Dict({}).tag(sync=True)
    tag_action = traitlets.Dict({}).tag(sync=True)

    #Traitlets per modifiche sulla distanza in batch -- assicura di non perdere edit
    pairwise_edit = traitlets.Dict({}).tag(sync=True)
    pairwise_edits_batch = traitlets.List([]).tag(sync=True)

    def __init__(self, sidecar_path=None, **kwargs):
        super().__init__(**kwargs)
        self.sidecar = SidecarManager(filepath=sidecar_path) if sidecar_path else SidecarManager()
        self.observe(self._on_pairwise_edit, names=["pairwise_edit"])
        self.observe(self._on_pairwise_edits_batch, names=["pairwise_edits_batch"])
        self.observe(self._on_tag_action, names=["tag_action"])
        self.refresh_global_tags()

    ### Gestione TAG
    def refresh_global_tags(self):
        """Sincronizza il dizionatio dei tag globali con il frontend"""
        self.global_tags = self.sidecar.get_global_tags()

    def _on_tag_action(self, change):
        """Gestisce le azioni di aggiunta e rimozione dei TAG inviate da JavaScript"""
        action_data = change["new"]
        if not action_data:
            return

        action = action_data.get("action")
        chunk_id = action_data.get("chunk_id")
        tag = action_data.get("tag")
        color = action_data.get("color")

        if action == "add" and chunk_id and tag:
            self.sidecar.add_tag_override(chunk_id, tag, color=color)
        elif action == "remove" and chunk_id and tag:
            self.sidecar.remove_tag_override(chunk_id, tag)

        self.refresh_global_tags()
        self._update_node_tags_in_graph_data(chunk_id)

    def _update_node_tags_in_graph_data(self, chunk_id):
        """
        Aggiorna i tag del nodo specifico in graph_data per forzare il ridisegno.
        Semplicemente ricrea e riposiziona i dati (uguali) nel nodo del chunk per forzare il re-render
        """
        data = self.sidecar.load_data()
        chunk_overrides = data.get("tag_overrides", {}).get(str(chunk_id), {})
        user_tags = chunk_overrides.get("user_tags", [])

        new_graph_data = dict(self.graph_data)
        nodes = [dict(n) for n in new_graph_data.get("nodes", [])]
        for n in nodes:
            if str(n.get("id")) == str(chunk_id):
                n["user_tags"] = user_tags
        new_graph_data["nodes"] = nodes
        self.graph_data = new_graph_data

    ### Caricamento grafo
    def load_graph(self, raw_graph_data):
        """Carica il grafo applicando immediatamente i distance_factor salvati nel sidecar."""
        data = self.sidecar.load_data()
        deltas = data.get("pairwise_deltas", {})
        overrides = data.get("tag_overrides", {})

        links = raw_graph_data.get("links", [])
        enriched_links = []

        # archi
        for link in links:
            l_copy = dict(link)
            src = l_copy["source"]["id"] if isinstance(l_copy["source"], dict) else l_copy["source"]
            tgt = l_copy["target"]["id"] if isinstance(l_copy["target"], dict) else l_copy["target"]

            k1 = f"{src}_AND_{tgt}"
            k2 = f"{tgt}_AND_{src}"

            factor = 1.0
            if k1 in deltas:
                factor = deltas[k1].get("distance_factor", 1.0)
            elif k2 in deltas:
                factor = deltas[k2].get("distance_factor", 1.0)

            l_copy["distance_factor"] = factor
            enriched_links.append(l_copy)

        # info sui nodi
        nodes = raw_graph_data.get("nodes", [])
        enriched_nodes = []
        for node in nodes:
            n_copy = dict(node)
            cid = str(n_copy.get("id"))
            if cid in overrides:
                n_copy["user_tags"] = overrides[cid].get("user_tags", [])
            elif "user_tags" not in n_copy:
                n_copy["user_tags"] = n_copy.get("tags", [])
            enriched_nodes.append(n_copy)

        self.graph_data = {
            "nodes": enriched_nodes,
            "links": enriched_links,
            "base_distance": BASE_GRAPH_DISTANCE
        }
        self.refresh_global_tags()

    ### Gestione edit Distanze
    def _on_pairwise_edit(self, change):
        """Gestisce una singola modifica di distanza inviata da JS -- versione meno sicura di pairwise_edits_batch"""
        edit = change["new"]
        if edit and "chunk_1" in edit and "chunk_2" in edit:
            self.sidecar.save_pairwise_delta(
                chunk_id_1=edit["chunk_1"],
                chunk_id_2=edit["chunk_2"],
                distance_factor=edit["distance_factor"]
            )

    def _on_pairwise_edits_batch(self, change):
        """
        Gestisce un blocco (tramite lista) di modifiche simultanee inviato in una sola volta.
        Si verifica quando il trascinamento di 1 nodo altera la distanza con più nodi ad esso collegati.
        Delegata al SidecarManager.
        """
        edits = change["new"]
        self.sidecar.save_pairwise_deltas_batch(edits)
