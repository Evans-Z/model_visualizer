/* global cytoscape, dagre */

cytoscape.use(cytoscapeDagre);

const statusOutput = document.getElementById("status-output");
const nodeDetails = document.getElementById("node-details");
const form = document.getElementById("visualize-form");

let cy = null;

function setStatus(message) {
  statusOutput.textContent = message;
}

function summarizeShapes(shapeList, maxItems = 3) {
  if (!shapeList || shapeList.length === 0) {
    return "[]";
  }
  const shown = shapeList.slice(0, maxItems);
  const more = shapeList.length - shown.length;
  return more > 0 ? `${shown.join(" | ")} (+${more} more)` : shown.join(" | ");
}

function moduleColor(moduleType) {
  const type = moduleType.toLowerCase();
  if (type.includes("input")) return "#334155";
  if (type.includes("output")) return "#334155";
  if (type.includes("attention")) return "#8b5cf6";
  if (type.includes("linear")) return "#f59e0b";
  if (type.includes("embedding")) return "#14b8a6";
  if (type.includes("norm")) return "#22c55e";
  if (type.includes("dropout")) return "#64748b";
  return "#3b82f6";
}

function buildElements(payload) {
  const nodes = payload.nodes.map((node) => {
    const size = node.parameter_count > 0 ? 45 + Math.log10(node.parameter_count + 1) * 12 : 36;
    return {
      data: {
        id: node.id,
        label: node.id === "__input__" || node.id === "__output__" ? node.label : node.id,
        module_type: node.module_type,
        parameter_count: node.parameter_count,
        call_count: node.call_count,
        input_shapes: node.input_shapes,
        output_shapes: node.output_shapes,
        parameter_shapes: node.parameter_shapes,
        size,
        color: moduleColor(node.module_type),
      },
    };
  });

  const edges = payload.edges.map((edge, index) => ({
    data: {
      id: `${edge.source}->${edge.target}-${index}`,
      source: edge.source,
      target: edge.target,
      label: summarizeShapes(edge.shapes, 2),
    },
  }));

  return [...nodes, ...edges];
}

function renderDetails(data) {
  const lines = [];
  lines.push(`Node: ${data.id}`);
  lines.push(`Type: ${data.module_type}`);
  lines.push(`Call count: ${data.call_count}`);
  lines.push(`Parameter count: ${data.parameter_count.toLocaleString()}`);
  lines.push("");
  lines.push("Input shapes:");
  if (data.input_shapes.length === 0) {
    lines.push("  (none)");
  } else {
    data.input_shapes.forEach((s) => lines.push(`  - ${s}`));
  }
  lines.push("");
  lines.push("Output shapes:");
  if (data.output_shapes.length === 0) {
    lines.push("  (none)");
  } else {
    data.output_shapes.forEach((s) => lines.push(`  - ${s}`));
  }
  lines.push("");
  lines.push("Parameter dimensions:");
  if (!data.parameter_shapes || data.parameter_shapes.length === 0) {
    lines.push("  (none)");
  } else {
    const maxRows = 120;
    data.parameter_shapes.slice(0, maxRows).forEach((param) => {
      lines.push(`  - ${param.name}: [${param.shape.join(", ")}] (${param.numel.toLocaleString()})`);
    });
    if (data.parameter_shapes.length > maxRows) {
      lines.push(`  ... ${data.parameter_shapes.length - maxRows} more`);
    }
  }

  nodeDetails.textContent = lines.join("\n");
}

function renderGraph(payload) {
  const container = document.getElementById("graph");
  const elements = buildElements(payload);

  if (cy) {
    cy.destroy();
  }

  cy = cytoscape({
    container,
    elements,
    style: [
      {
        selector: "node",
        style: {
          label: "data(label)",
          color: "#e2e8f0",
          "font-size": 10,
          "text-wrap": "wrap",
          "text-max-width": 120,
          "background-color": "data(color)",
          width: "data(size)",
          height: "data(size)",
          "border-width": 1,
          "border-color": "#94a3b8",
        },
      },
      {
        selector: "edge",
        style: {
          width: 1.8,
          "line-color": "#475569",
          "target-arrow-color": "#475569",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          label: "data(label)",
          color: "#a5b4fc",
          "font-size": 8,
          "text-background-color": "#020617",
          "text-background-opacity": 1,
          "text-background-padding": 1,
        },
      },
    ],
    layout: {
      name: "dagre",
      rankDir: "LR",
      nodeSep: 30,
      rankSep: 70,
      edgeSep: 12,
      animate: false,
    },
    minZoom: 0.15,
    maxZoom: 2.2,
  });

  cy.on("tap", "node", (event) => renderDetails(event.target.data()));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Tracing model execution...");
  nodeDetails.textContent = "Run visualization and click a node for details.";

  const payload = {
    model_id_or_path: document.getElementById("model-id").value.trim(),
    sample_text: document.getElementById("sample-text").value,
    seq_len: Number(document.getElementById("seq-len").value),
    batch_size: Number(document.getElementById("batch-size").value),
    trust_remote_code: document.getElementById("trust-remote-code").checked,
    device: document.getElementById("device").value,
  };

  try {
    const response = await fetch("/api/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();

    if (!response.ok) {
      setStatus(`Error: ${result.detail || "Unknown failure"}`);
      return;
    }

    renderGraph(result);
    setStatus(
      [
        `Model: ${result.model.id_or_path} (${result.model.class})`,
        `Executed modules: ${result.totals.executed_modules.toLocaleString()}`,
        `Total params: ${result.totals.parameters.toLocaleString()}`,
        `Trainable params: ${result.totals.trainable_parameters.toLocaleString()}`,
        `Warnings: ${result.warnings.length > 0 ? result.warnings.join(" | ") : "none"}`,
      ].join("\n"),
    );
  } catch (error) {
    setStatus(`Request failed: ${error}`);
  }
});
