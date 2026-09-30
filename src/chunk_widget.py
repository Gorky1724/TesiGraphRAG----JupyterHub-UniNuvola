import sys
import pathlib
from pathlib import Path
import anywidget
import traitlets
project_root = Path("~/tesi_graphrag").expanduser().resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import BASE_GRAPH_DISTANCE, DISTANCE_CLASS_FACTORS
from src.sidecar_manager import SidecarManager

class ChunkGraphWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "frontend" / "chunk_graph.js"

    # Traitlets per stato del grafo
    graph_data = traitlets.Dict({"nodes": [], "links": []}).tag(sync=True)
    selected_tag = traitlets.Dict({}).tag(sync=True)
    global_tags = traitlets.Dict({}).tag(sync=True)
    tag_action = traitlets.Dict({}).tag(sync=True)

    #Traitlets per modifiche sulle classi di distanza in batch -- assicura di non perdere edit
    pairwise_class_edit = traitlets.Dict({}).tag(sync=True)
    pairwise_edits_batch = traitlets.List([]).tag(sync=True)

    def __init__(self, sidecar=None, sidecar_path=None, **kwargs):
        """
        Per evitare di avere una doppia cache del sidecar_manager, è bene passare il sidecar_mngr
        e non il sidecar_path per evitare disallineamenti della cache RAM. L'opzione è stata lasciata per test e legacy, ma farlo rischia
        di dare un WARNING che è fondamentale non ignorare se non con coscienza.
        """
        super().__init__(**kwargs)

        if isinstance(sidecar, SidecarManager):
            self.sidecar = sidecar
        elif sidecar_path:
            self.sidecar = SidecarManager(filepath=sidecar_path)
        else:
            self.sidecar = SidecarManager()

        self._raw_graph_data = {"nodes": [], "links": []}

        self.observe(self._on_pairwise_edit, names=["pairwise_class_edit"])
        self.observe(self._on_pairwise_edits_batch, names=["pairwise_edits_batch"])
        self.observe(self._on_tag_action, names=["tag_action"])
        self.refresh_global_tags()

    ### Caricamento grafo
    def load_graph(self, raw_graph_data):
        """Carica il grafo applicando immediatamente i distance_factor salvati nel sidecar."""
        self._raw_graph_data = raw_graph_data
        data = self.sidecar.load_data()
        adj = data.get("modified_adjacencies", {})
        overrides = data.get("tag_overrides", {})

        # Info sui nodi
        nodes = raw_graph_data.get("nodes", [])
        enriched_nodes = []
        valid_node_ids = set()

        for node in nodes:
            n_copy = dict(node)
            cid = str(n_copy.get("id"))
            valid_node_ids.add(cid)
            if cid in overrides:
                n_copy["user_tags"] = overrides[cid].get("user_tags", [])
            elif "user_tags" not in n_copy:
                n_copy["user_tags"] = n_copy.get("tags", [])
            enriched_nodes.append(n_copy)

        # Archi e struttura a adiacenza
        raw_links = raw_graph_data.get("links", [])
        enriched_links = []
        existing_pairs = set()

        for link in raw_links:
            l_copy = dict(link)
            src = str(l_copy["source"]["id"] if isinstance(l_copy["source"], dict) else l_copy["source"])
            tgt = str(l_copy["target"]["id"] if isinstance(l_copy["target"], dict) else l_copy["target"])

            pair_key = tuple(sorted([src, tgt]))
            existing_pairs.add(pair_key)

            distance_class = "INVARIATI"
            if src in adj and tgt in adj[src]:
                distance_class = adj[src][tgt].get("distance_class", "INVARIATI")
            elif tgt in adj and src in adj[tgt]:
                distance_class = adj[tgt][src].get("distance_class", "INVARIATI")

            factor = DISTANCE_CLASS_FACTORS.get(distance_class, 1.0)

            l_copy["source"] = src
            l_copy["target"] = tgt
            l_copy["distance_class"] = distance_class
            l_copy["distance_factor"] = factor
            enriched_links.append(l_copy)

        # Aggiunge archi del grafo non presenti nel kg originale
        #  Ma derivanti da alterazioni del sidecar
        for src, targets in adj.items():
            for tgt, info in targets.items():
                pair_key = tuple(sorted([str(src), str(tgt)]))
                if pair_key not in existing_pairs:
                    d_cls = info.get("distance_class", "INVARIATI")
                    if d_cls != "INVARIATI":
                        existing_pairs.add(pair_key)
                        factor = DISTANCE_CLASS_FACTORS.get(d_cls, 1.0)
                        enriched_links.append({
                            "source": str(src),
                            "target": str(tgt),
                            "distance_class": d_cls,
                            "distance_factor": factor
                        })

       # Mantiene soli gli archi con sia source che target trai nodi attivi
        valid_links = [
            link for link in enriched_links
            if str(link["source"]) in valid_node_ids and str(link["target"]) in valid_node_ids
        ]

        self.graph_data = {
            "nodes": enriched_nodes,
            "links": valid_links,
            "base_distance": BASE_GRAPH_DISTANCE
        }
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

    ### Gestione Classi di Distanza
    def _on_pairwise_edit(self, change):
        """Gestisce una singola modifica di distanza inviata da JS -- versione meno sicura di pairwise_edits_batch"""
        edit = change["new"]
        if edit and "chunk_1" in edit and "chunk_2" in edit:
            c1 = edit["chunk_1"]
            c2 = edit["chunk_2"]
            d_cls = edit.get("distance_class")
            self.sidecar.save_pairwise_class_edits(c1, c2, d_cls)
            if self._raw_graph_data.get("nodes"):
                self.load_graph(self._raw_graph_data)

    def _on_pairwise_edits_batch(self, change):
        """
        Gestisce un blocco (tramite lista) di modifiche simultanee inviato in una sola volta.
        Si verifica quando il trascinamento di 1 nodo altera la distanza con più nodi ad esso collegati.
        Delegata al SidecarManager.
        """
        edits = change["new"]
        if edits and isinstance(edits, list):
            self.sidecar.save_pairwise_class_edits_batch(edits)
            if self._raw_graph_data.get("nodes"):
                self.load_graph(self._raw_graph_data)
