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

  const svg = container.append("svg")
    .attr("width", width)
    .attr("height", height)
    .style("background", "#f8fafc")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "8px");

  const infoBox = container.append("div")
    .style("position", "absolute")
    .style("top", "12px")
    .style("right", "12px")
    .style("width", "220px")
    .style("padding", "10px")
    .style("background", "rgba(255, 255, 255, 0.95)")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "6px")
    .style("font-size", "12px")
    .style("color", "#334155")
    .style("pointer-events", "none")
    .html("<b>📌 Info Chunk</b><br><span style='color:#94a3b8;'>Clicca un nodo per vederne il testo</span>");

  function draw() {
    const graph = model.get("graph_data");
    if (!graph || !graph.nodes || graph.nodes.length === 0) return;

    svg.selectAll("*").remove();

    const nodes = graph.nodes.map(d => ({ ...d }));
    const links = graph.links ? graph.links.map(d => ({ ...d })) : [];

    const BASE_DISTANCE = 120;

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links)
        .id(d => d.id)
        .distance(d => BASE_DISTANCE * (d.distance_factor || 1.0))
      )
      .force("charge", d3.forceManyBody().strength(-180))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "#94a3b8")
      .attr("stroke-width", 2);

    const linkText = svg.append("g")
      .selectAll("text")
      .data(links)
      .enter().append("text")
      .attr("font-size", "11px")
      .attr("font-weight", "bold")
      .attr("fill", "#0284c7")
      .attr("text-anchor", "middle");

    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", 12)
      .attr("fill", "#6366f1")
      .attr("stroke", "#ffffff")
      .attr("stroke-width", 2)
      .style("cursor", "grab");

    const label = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.id)
      .attr("font-size", "11px")
      .attr("dx", 15)
      .attr("dy", 4)
      .attr("fill", "#1e293b");

    // Blocca permanentemente TUTTI i nodi appena il layout iniziale è pronto
    simulation.on("end", () => {
      nodes.forEach(n => {
        n.fx = n.x;
        n.fy = n.y;
      });
    });

    const drag = d3.drag()
      .on("start", (event, d) => {
        // Congela istantaneamente la posizione di tutti i nodi
        nodes.forEach(n => {
          n.fx = n.x;
          n.fy = n.y;
        });
      })
      .on("drag", (event, d) => {
        // Aggiorna solo il nodo trascinato senza svegliare la fisica di D3
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
            const factor = currentDist / BASE_DISTANCE;

            l.distance_factor = factor;

            model.set("pairwise_edit", {
              chunk_1: srcId,
              chunk_2: tgtId,
              distance_factor: factor
            });
            model.save_changes();
          }
        });

        updatePositions();
      });

    node.call(drag);

    node.on("click", (event, d) => {
      node.attr("fill", n => n.id === d.id ? "#ef4444" : "#6366f1");
      infoBox.html(`<b>🆔 ${d.id}</b><br><span style="color:#334155;">📖 ${d.text || "Nessun testo"}</span>`);
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
          const factor = (dist / BASE_DISTANCE).toFixed(2);
          return `${dist}px (${factor}x)`;
        });

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    }

    simulation.on("tick", updatePositions);
  }

  model.on("change:graph_data", draw);
  draw();
}
