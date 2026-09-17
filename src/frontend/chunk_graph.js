import * as d3 from "https://esm.sh/d3@7";

export function render({ model, el }) {
  el.innerHTML = "";

  const container = d3.select(el)
    .append("div")
    .style("position", "relative")
    .style("width", "650px")
    .style("font-family", "sans-serif");

  const width = 650;
  const height = 420;

  let selectedNodeId = null;

  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height)
    .style("background", "#f8fafc")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "8px")
    .style("cursor", "grab");

  // Contenitore SVG scalabile e traslabile
  const g = svg.append("g");

  // Gestore Zoom e Pan
  const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on("zoom", (event) => {
      g.attr("transform", event.transform);
    });

  svg.call(zoom);

  const infoBox = container.append("div")
    .style("position", "absolute")
    .style("top", "12px")
    .style("right", "12px")
    .style("width", "230px")
    .style("padding", "10px")
    .style("background", "rgba(255, 255, 255, 0.95)")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "6px")
    .style("font-size", "12px")
    .style("color", "#334155")
    .style("pointer-events", "none")
    .style("box-shadow", "0 2px 4px rgba(0,0,0,0.05)")
    .html("<b>📌 Info Chunk</b><br><span style='color:#94a3b8;'>Clicca un nodo per vederne testo e penalità</span>");

  function draw() {
    const graph = model.get("graph_data");
    if (!graph || !graph.nodes || graph.nodes.length === 0) return;

    // Lettura dinamica della soglia inviata da Python (fallback a 2.5)
    const hardFilterThreshold = graph.hard_filter_threshold || 2.5;

    g.selectAll("*").remove();

    const nodes = graph.nodes.map(d => ({ ...d }));
    const links = graph.links ? graph.links.map(d => ({ ...d })) : [];

    // Trova la penalità massima tra tutti gli archi incidenti sul nodo
    function getMaxPenalty(nodeId) {
      let maxPenalty = 1.0;
      links.forEach(l => {
        const srcId = typeof l.source === 'object' ? l.source.id : l.source;
        const tgtId = typeof l.target === 'object' ? l.target.id : l.target;
        if (srcId === nodeId || tgtId === nodeId) {
          const factor = l.distance_factor || (l.target_distance && l.base_distance ? l.target_distance / l.base_distance : 1.0);
          if (factor > maxPenalty) maxPenalty = factor;
        }
      });
      return maxPenalty;
    }

    // Simulazione D3 basata su distanze da Qdrant * distance_factor
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links)
        .id(d => d.id)
        .distance(d => (d.base_distance || 120) * (d.distance_factor || 1.0))
      )
      .force("charge", d3.forceManyBody().strength(-180))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = g.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "#94a3b8")
      .attr("stroke-width", 2);

    const linkText = g.append("g")
      .selectAll("text")
      .data(links)
      .enter().append("text")
      .attr("font-size", "11px")
      .attr("font-weight", "bold")
      .attr("fill", "#0284c7")
      .attr("text-anchor", "middle");

    const node = g.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", 12)
      .attr("stroke-width", 2)
      .style("cursor", "pointer");

    const label = g.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .attr("font-size", "11px")
      .attr("dx", 15)
      .attr("dy", 4)
      .attr("fill", "#1e293b");

    function updateNodeStyles() {
      node
        .attr("fill", d => {
          if (d.id === selectedNodeId) return "#f59e0b"; // Giallo/Ambra = Selezionato
          const maxP = getMaxPenalty(d.id);
          if (maxP >= hardFilterThreshold) return "#ef4444"; // Rosso = Escluso
          if (maxP > 1.0) return "#f97316"; // Arancione = Penalizzato
          return "#6366f1"; // Indaco = Normale
        })
        .attr("stroke", d => (d.id === selectedNodeId ? "#1e293b" : "#ffffff"))
        .attr("stroke-width", d => (d.id === selectedNodeId ? 3 : 2));

      label.text(d => {
        const maxP = getMaxPenalty(d.id);
        return maxP > 1.0 ? `${d.id} (${maxP.toFixed(1)}x)` : d.id;
      });
    }

    simulation.on("end", () => {
      nodes.forEach(n => {
        n.fx = n.x;
        n.fy = n.y;
      });
    });

    const drag = d3.drag()
      .on("start", (event, d) => {
        nodes.forEach(n => {
          n.fx = n.x;
          n.fy = n.y;
        });
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
        d.x = event.x;
        d.y = event.y;
        updatePositions();
      })
      .on("end", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
        d.x = event.x;
        d.y = event.y;

        links.forEach(l => {
          const srcId = typeof l.source === 'object' ? l.source.id : l.source;
          const tgtId = typeof l.target === 'object' ? l.target.id : l.target;

          if (srcId === d.id || tgtId === d.id) {
            const dx = l.target.x - l.source.x;
            const dy = l.target.y - l.source.y;
            const currentDist = Math.sqrt(dx * dx + dy * dy);
            const baseDist = l.base_distance || 120;
            const factor = currentDist / baseDist;

            l.distance_factor = factor;

            model.set("pairwise_edit", {
              chunk_1: srcId,
              chunk_2: tgtId,
              distance_factor: factor
            });
            model.save_changes();
          }
        });

        updateNodeStyles();
        updatePositions();
      });

    node.call(drag);

    node.on("click", (event, d) => {
      selectedNodeId = d.id;
      updateNodeStyles();

      const maxP = getMaxPenalty(d.id);
      let statusHtml = "<span style='color:#10b981; font-weight:bold;'>✅ Attivo</span>";
      if (maxP >= hardFilterThreshold) {
        statusHtml = `<span style='color:#ef4444; font-weight:bold;'>❌ ESCLUSO (${maxP.toFixed(2)}x)</span>`;
      } else if (maxP > 1.0) {
        statusHtml = `<span style='color:#f97316; font-weight:bold;'>⚠️ Penalizzato (${maxP.toFixed(2)}x)</span>`;
      }

      infoBox.html(`
        <b>🆔 ${d.id}</b><br>
        <b>Stato:</b> ${statusHtml}<br>
        <hr style="border:0; border-top:1px solid #e2e8f0; margin:6px 0;">
        <span style="color:#334155; display:block; max-height:100px; overflow-y:auto;">📖 ${d.text || "Nessun testo disponibile"}</span>
      `);

      model.set("selected_tag", { id: d.id, text: d.text || "" });
      model.save_changes();
    });

    function updatePositions() {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      linkText
        .attr("x", d => (d.source.x + d.target.x) / 2)
        .attr("y", d => (d.source.y + d.target.y) / 2 - 5)
        .text(d => {
          const dx = d.target.x - d.source.x;
          const dy = d.target.y - d.source.y;
          const dist = Math.round(Math.sqrt(dx * dx + dy * dy));
          const baseDist = d.base_distance || 120;
          const factor = (dist / baseDist).toFixed(2);
          return `${dist}px (${factor}x)`;
        });

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    }

    updateNodeStyles();
    simulation.on("tick", updatePositions);
  }

  model.on("change:graph_data", draw);
  draw();
}

export default { render };
