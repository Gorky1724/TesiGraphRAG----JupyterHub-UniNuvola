import * as d3 from "https://esm.sh/d3@7";

function render({ model, el }) {
  el.innerHTML = "";

  const width = 600;
  const height = 400;

  const svg = d3.select(el).append("svg")
    .attr("width", width)
    .attr("height", height)
    .style("background", "#f8fafc")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "8px");

  function draw() {
    const graph = model.get("graph_data");
    if (!graph || !graph.nodes || graph.nodes.length === 0) return;

    svg.selectAll("*").remove();

    const nodes = graph.nodes.map(d => ({ ...d }));
    const rawLinks = graph.links || graph.edges || [];
    const links = rawLinks.map(d => ({ ...d }));

    const initialPositions = {};

    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id).distance(100))
      .force("charge", d3.forceManyBody().strength(-200))
      .force("center", d3.forceCenter(width / 2, height / 2));

    const link = svg.append("g")
      .selectAll("line")
      .data(links)
      .enter().append("line")
      .attr("stroke", "#94a3b8")
      .attr("stroke-width", 2);

    // Gestione Drag & Drop con D3
    const drag = d3.drag()
      .on("start", (event, d) => {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
        // Salva la posizione iniziale al momento del click
        initialPositions[d.id] = { x: d.x, y: d.y };
      })
      .on("drag", (event, d) => {
        d.fx = event.x;
        d.fy = event.y;
      })
      .on("end", (event, d) => {
        if (!event.active) simulation.alphaTarget(0);

        // Trova i collegamenti del nodo rilasciato
        const connectedLinks = links.filter(l => 
          (l.source.id === d.id || l.source === d.id) || 
          (l.target.id === d.id || l.target === d.id)
        );

        if (connectedLinks.length > 0) {
          const targetLink = connectedLinks[0];
          const sourceObj = typeof targetLink.source === 'object' ? targetLink.source : { id: targetLink.source };
          const targetObj = typeof targetLink.target === 'object' ? targetLink.target : { id: targetLink.target };

          const otherNode = sourceObj.id === d.id ? targetObj : sourceObj;

          const origPos = initialPositions[d.id] || { x: d.x, y: d.y };
          const origDist = Math.hypot(origPos.x - otherNode.x, origPos.y - otherNode.y) || 1;
          const newDist = Math.hypot(d.x - otherNode.x, d.y - otherNode.y);

          const distanceFactor = newDist / origDist;

          // Invia la modifica a Python
          model.set("pairwise_edit", {
            chunk_1: d.id,
            chunk_2: otherNode.id,
            distance_factor: distanceFactor
          });
          model.save_changes();
        }
      });

    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", 12)
      .attr("fill", "#4f46e5")
      .style("cursor", "grab")
      .call(drag)
      .on("click", (event, d) => {
        model.set("selected_tag", { id: d.id, text: d.text || "" });
        model.save_changes();
      });

    const label = svg.append("g")
      .selectAll("text")
      .data(nodes)
      .enter().append("text")
      .text(d => d.id)
      .attr("font-size", "12px")
      .attr("dx", 15)
      .attr("dy", 4);

    simulation.on("tick", () => {
      link
        .attr("x1", d => d.source.x)
        .attr("y1", d => d.source.y)
        .attr("x2", d => d.target.x)
        .attr("y2", d => d.target.y);

      node
        .attr("cx", d => d.x)
        .attr("cy", d => d.y);

      label
        .attr("x", d => d.x)
        .attr("y", d => d.y);
    });
  }

  model.on("change:graph_data", draw);
  draw();
}

export default { render };
