import * as d3 from "https://esm.sh/d3@7";

export function render({ model, el }) {
  el.innerHTML = "";

  function str(val) {
    return String(val);
  }

  let currentSimulation = null;
  let renderPending = false;
  let selectedNodeId = null;
  let currentNodes = [];

  // --- LAYOUT PRINCIPALE FLEXBOX ---
  const mainContainer = d3.select(el)
    .append("div")
    .style("display", "flex")
    .style("flex-direction", "row")
    .style("gap", "15px")
    .style("font-family", "system-ui, -apple-system, sans-serif")
    .style("color", "#1e293b");

  // --- COLONNA SINISTRA: GRAFO D3.js ---
  const graphContainer = mainContainer.append("div")
    .style("position", "relative")
    .style("width", "580px")
    .style("height", "420px");

  const width = 580;
  const height = 420;

  const svg = graphContainer.append("svg")
    .attr("width", width)
    .attr("height", height)
    .style("background", "#f8fafc")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "8px")
    .style("cursor", "grab");

  const defs = svg.append("defs");
  const g = svg.append("g");

  const zoom = d3.zoom()
    .scaleExtent([0.2, 4])
    .on("zoom", (event) => {
      g.attr("transform", event.transform);
    });

  svg.call(zoom);

  // --- COLONNA DESTRA: PANNELLO DI CONTROLLO ---
  const panel = mainContainer.append("div")
    .style("width", "320px")
    .style("height", "420px")
    .style("box-sizing", "border-box")
    .style("padding", "12px")
    .style("background", "#ffffff")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "8px")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("gap", "12px")
    .style("overflow-y", "auto");

  // Legenda
  const legendSection = panel.append("div")
    .style("padding-bottom", "10px")
    .style("border-bottom", "1px solid #e2e8f0");

  legendSection.append("div")
    .style("font-weight", "bold")
    .style("font-size", "13px")
    .style("margin-bottom", "6px")
    .text("*>> Tag Globali Attivi");

  const legendContainer = legendSection.append("div")
    .style("display", "flex")
    .style("flex-wrap", "wrap")
    .style("gap", "6px");

  // Dettagli Chunk
  const detailSection = panel.append("div")
    .style("display", "flex")
    .style("flex-direction", "column")
    .style("gap", "8px");

  detailSection.append("div")
    .style("font-weight", "bold")
    .style("font-size", "13px")
    .text(">> Chunk Selezionato");

  const detailContent = detailSection.append("div")
    .style("font-size", "12px")
    .style("color", "#64748b");

  const detailHeader = detailContent.append("div").style("font-weight", "bold").style("color", "#0f172a").style("margin-bottom", "4px");
  const detailText = detailContent.append("div")
    .style("max-height", "110px")
    .style("overflow-y", "auto")
    .style("padding", "6px 8px")
    .style("background", "#f1f5f9")
    .style("border-radius", "4px")
    .style("font-size", "11px")
    .style("margin-bottom", "8px");

  detailContent.append("div").style("font-weight", "600").style("font-size", "11px").style("margin-bottom", "4px").text("Tag Associati:");
  const currentTagsContainer = detailContent.append("div")
    .style("display", "flex")
    .style("flex-wrap", "wrap")
    .style("gap", "4px")
    .style("margin-bottom", "10px");

  const addTagBox = detailContent.append("div")
    .style("display", "flex")
    .style("gap", "4px");

  const inputTag = addTagBox.append("input")
    .attr("type", "text")
    .attr("placeholder", "Nuovo Tag...")
    .style("flex", "1")
    .style("padding", "4px 6px")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "4px")
    .style("font-size", "11px");

  const submitBtn = addTagBox.append("button")
    .text("+ Aggiungi")
    .style("padding", "4px 8px")
    .style("background", "#2563eb")
    .style("color", "#ffffff")
    .style("border", "none")
    .style("border-radius", "4px")
    .style("cursor", "pointer")
    .style("font-size", "11px");

  // Aggiunta Tag con controllo duplicati
  function submitTag() {
    const val = inputTag.property("value").trim();
    if (!val || !selectedNodeId) return;

    const graph = model.get("graph_data");
    const nodeData = graph && graph.nodes ? graph.nodes.find(n => str(n.id) === str(selectedNodeId)) : null;
    const userTags = nodeData ? (nodeData.user_tags || nodeData.tags || []) : [];

    if (!userTags.includes(val)) {
      model.set("tag_action", {
        action: "add",
        chunk_id: selectedNodeId,
        tag: val,
        timestamp: Date.now()
      });
      model.save_changes();
      inputTag.property("value", "");
    }
  }

  inputTag.on("keydown", (event) => { if (event.key === "Enter") submitTag(); });
  submitBtn.on("click", submitTag);

  function updateLegend() {
    const globalTags = model.get("global_tags") || {};
    legendContainer.html("");

    const tagNames = Object.keys(globalTags);
    if (tagNames.length === 0) {
      legendContainer.append("span").style("color", "#94a3b8").style("font-size", "11px").text("Nessun tag assegnato.");
      return;
    }

    tagNames.forEach(tagName => {
      const info = globalTags[tagName];
      legendContainer.append("span")
        .style("display", "inline-flex")
        .style("align-items", "center")
        .style("padding", "2px 8px")
        .style("border-radius", "12px")
        .style("font-size", "11px")
        .style("color", "#ffffff")
        .style("font-weight", "500")
        .style("background", info.color || "#6366f1")
        .text(`${tagName} (${info.count})`);
    });
  }

  // Generazione colore o gradiente per multi-tag
  function getNodeFill(d, globalTags) {
    const userTags = d.user_tags || d.tags || [];
    if (userTags.length === 0) return "#94a3b8";

    if (userTags.length === 1) {
      const tag = userTags[0];
      return (globalTags[tag] && globalTags[tag].color) ? globalTags[tag].color : "#6366f1";
    }

    // Nodi Multi-Tag: Creazione Gradiente SVG
    const gradId = `grad-${str(d.id).replace(/[^a-zA-Z0-9_-]/g, '_')}`;
    defs.select(`#${gradId}`).remove();

    const grad = defs.append("linearGradient")
      .attr("id", gradId)
      .attr("x1", "0%")
      .attr("y1", "0%")
      .attr("x2", "100%")
      .attr("y2", "100%");

    const count = userTags.length;
    userTags.forEach((t, i) => {
      const color = (globalTags[t] && globalTags[t].color) ? globalTags[t].color : "#6366f1";
      const startPercent = (i / count) * 100;
      const endPercent = ((i + 1) / count) * 100;

      grad.append("stop").attr("offset", `${startPercent}%`).attr("stop-color", color);
      grad.append("stop").attr("offset", `${endPercent}%`).attr("stop-color", color);
    });

    return `url(#${gradId})`;
  }

  function renderDetailPanel() {
    const graph = model.get("graph_data");
    const globalTags = model.get("global_tags") || {};

    if (!selectedNodeId || !graph || !graph.nodes) {
      detailHeader.text("");
      detailText.text("Clicca un nodo nel grafo per vederne il testo e gestirne i Tag.");
      currentTagsContainer.html("");
      addTagBox.style("display", "none");
      return;
    }

    const nodeData = graph.nodes.find(n => str(n.id) === str(selectedNodeId));
    if (!nodeData) {
      detailHeader.text("");
      detailText.text("Nodo non trovato.");
      currentTagsContainer.html("");
      addTagBox.style("display", "none");
      return;
    }

    addTagBox.style("display", "flex");
    detailHeader.text(`ID ${nodeData.id}`);
    detailText.text(nodeData.text || "Nessun testo associato.");

    currentTagsContainer.html("");
    const userTags = nodeData.user_tags || nodeData.tags || [];

    if (userTags.length === 0) {
      currentTagsContainer.append("span").style("font-size", "11px").style("color", "#94a3b8").text("Nessun tag.");
    } else {
      userTags.forEach(t => {
        const tagColor = (globalTags[t] && globalTags[t].color) ? globalTags[t].color : "#6366f1";
        const tagPill = currentTagsContainer.append("span")
          .style("display", "inline-flex")
          .style("align-items", "center")
          .style("gap", "4px")
          .style("padding", "2px 6px")
          .style("border-radius", "4px")
          .style("background", tagColor)
          .style("color", "#ffffff")
          .style("font-size", "11px");

        tagPill.append("span").text(t);
        tagPill.append("span")
          .style("cursor", "pointer")
          .style("font-weight", "bold")
          .style("margin-left", "2px")
          .text("✕")
          .on("click", () => {
            model.set("tag_action", {
              action: "remove",
              chunk_id: nodeData.id,
              tag: t,
              timestamp: Date.now()
            });
            model.save_changes();
          });
      });
    }
  }

  function draw() {
    if (currentSimulation) {
      currentSimulation.stop();
    }

    g.selectAll("*").remove();
    defs.selectAll("*").remove();

    const graph = model.get("graph_data");
    const globalTags = model.get("global_tags") || {};

    updateLegend();
    renderDetailPanel();

    if (!graph || !graph.nodes || graph.nodes.length === 0) return;

    const posMap = {};
    if (currentNodes) {
      currentNodes.forEach(n => {
        posMap[str(n.id)] = { x: n.x, y: n.y, fx: n.fx, fy: n.fy };
      });
    }

    const nodes = graph.nodes.map(d => {
      const existing = posMap[str(d.id)];
      if (existing) {
        return { ...d, x: existing.x, y: existing.y, fx: existing.fx, fy: existing.fy };
      }
      return { ...d };
    });

    currentNodes = nodes;
    const links = graph.links ? graph.links.map(d => ({ ...d })) : [];
    const BASE_DISTANCE = graph.base_distance || 120;

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links)
        .id(d => d.id)
        .distance(d => BASE_DISTANCE * (d.distance_factor || 1.0))
      )
      .force("charge", d3.forceManyBody().strength(-180))
      .force("center", d3.forceCenter(width / 2, height / 2));

    currentSimulation = simulation;

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
      .attr("r", 13)
      .attr("fill", d => getNodeFill(d, globalTags))
      .attr("stroke", d => str(d.id) === str(selectedNodeId) ? "#000000" : "#ffffff")
      .attr("stroke-width", d => str(d.id) === str(selectedNodeId) ? 3 : 2)
      .style("cursor", "grab");

    const label = g.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.id)
      .attr("font-size", "11px")
      .attr("dx", 16)
      .attr("dy", 4)
      .attr("fill", "#1e293b");

    simulation.on("end", () => {
      nodes.forEach(n => {
        n.fx = n.x;
        n.fy = n.y;
      });
    });

    // Binding corretto del drag rispetto allo zoom
    const drag = d3.drag()
      .container(g.node())
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

        const batch = [];
        links.forEach(l => {
          const srcId = typeof l.source === 'object' ? l.source.id : l.source;
          const tgtId = typeof l.target === 'object' ? l.target.id : l.target;

          if (str(srcId) === str(d.id) || str(tgtId) === str(d.id)) {
            const dx = l.target.x - l.source.x;
            const dy = l.target.y - l.source.y;
            const currentDist = Math.hypot(dx, dy);
            const factor = currentDist / BASE_DISTANCE;

            l.distance_factor = factor;
            batch.push({
              chunk_1: srcId,
              chunk_2: tgtId,
              distance_factor: factor
            });
          }
        });

        if (batch.length > 0) {
          model.set("pairwise_edits_batch", batch);
          model.save_changes();
        }

        updatePositions();
      });

    node.call(drag);

    node.on("click", (event, d) => {
      if (str(selectedNodeId) !== str(d.id)) {
        inputTag.property("value", ""); // Reset input cambio nodo
      }

      selectedNodeId = d.id;
      renderDetailPanel();

      node.attr("stroke", n => str(n.id) === str(d.id) ? "#000000" : "#ffffff")
          .attr("stroke-width", n => str(n.id) === str(d.id) ? 3 : 2);

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
          const dist = Math.round(Math.hypot(dx, dy));
          const factor = (dist / BASE_DISTANCE).toFixed(2);
          return `${dist}px (${factor}x)`;
        });

      node.attr("cx", d => d.x).attr("cy", d => d.y);
      label.attr("x", d => d.x).attr("y", d => d.y);
    }

    simulation.on("tick", updatePositions);
    updatePositions(); // Posizionamento istantaneo
  }

  function scheduleDraw() {
    if (!renderPending) {
      renderPending = true;
      requestAnimationFrame(() => {
        renderPending = false;
        draw();
      });
    }
  }

  model.on("change:graph_data", scheduleDraw);
  model.on("change:global_tags", scheduleDraw);

  draw();

  return () => {
    if (currentSimulation) {
      currentSimulation.stop();
    }
    model.off("change:graph_data", scheduleDraw);
    model.off("change:global_tags", scheduleDraw);
  };
}
