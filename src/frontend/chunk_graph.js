import * as d3 from "https://esm.sh/d3@7";

function render({ model, el }) {
  el.innerHTML = "";

  const width = 600;
  const height = 400;

  const svg = d3.select(el).append("svg")
    .attr("width", width)
    .attr("height", height)
    .style("background", "#f9f9f9")
    .style("border", "1px solid #cbd5e1")
    .style("border-radius", "8px");

  function draw() {
    const graph = model.get("graph_data");
    if (!graph || !graph.nodes || graph.nodes.length === 0) return;

    svg.selectAll("*").remove();

    const nodes = graph.nodes.map(d => ({ ...d }));
    const rawLinks = graph.links || graph.edges || [];
    const links = rawLinks.map(d => ({ ...d }));

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

    const node = svg.append("g")
      .selectAll("circle")
      .data(nodes)
      .enter().append("circle")
      .attr("r", 12)
      .attr("fill", "#4f46e5")
      .style("cursor", "pointer")
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
