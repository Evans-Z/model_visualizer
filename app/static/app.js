/* global cytoscape, dagre, cytoscapeDagre */

cytoscape.use(cytoscapeDagre);

const statusOutput = document.getElementById("status-output");
const nodeDetails = document.getElementById("node-details");
const form = document.getElementById("visualize-form");
const releaseGpuButton = document.getElementById("release-gpu-button");

let cy = null;

function setStatus(message) {
  statusOutput.textContent = message;
}

function summarizeShapes(shapeList, maxItems = 3) {
  if (!shapeList || shapeList.length === 0) return "[]";
  const shown = shapeList.slice(0, maxItems);
  const more = shapeList.length - shown.length;
  return more > 0 ? `${shown.join(" | ")} (+${more} more)` : shown.join(" | ");
}

function moduleColor(moduleType) {
  const type = String(moduleType || "").toLowerCase();
  if (type.includes("input")) return "#334155";
  if (type.includes("output")) return "#334155";
  if (type.includes("functional")) return "#f97316";
  if (type.includes("attention")) return "#8b5cf6";
  if (type.includes("linear")) return "#f59e0b";
  if (type.includes("embedding")) return "#14b8a6";
  if (type.includes("norm")) return "#22c55e";
  if (type.includes("dropout")) return "#64748b";
  return "#3b82f6";
}

function nodeDisplayLabel(node) {
  if (node.id === "__input__" || node.id === "__output__" || node.id === "__functional__") {
    return node.label;
  }
  const moduleName = node.module_name || node.id;
  const shortPath = moduleName.split(".").slice(-2).join(".") || moduleName;
  const callIndex = node.call_index ? ` #${node.call_index}` : "";
  return `${shortPath}${callIndex}\n${node.module_type}`;
}

function buildElements(payload) {
  const nodes = payload.nodes.map((node) => {
    const size = node.parameter_count > 0 ? 40 + Math.log10(node.parameter_count + 1) * 10 : 30;
    return {
      data: {
        id: node.id,
        label: nodeDisplayLabel(node),
        module_name: node.module_name || node.id,
        call_index: node.call_index || null,
        module_type: node.module_type,
        parameter_count: node.parameter_count,
        call_count: node.call_count,
        input_shapes: node.input_shapes || [],
        output_shapes: node.output_shapes || [],
        parameter_shapes: node.parameter_shapes || [],
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
      kind: edge.kind || "observed",
      shape_summary: summarizeShapes(edge.shapes, 4),
      shapes: edge.shapes || [],
    },
    classes: edge.kind || "observed",
  }));

  return [...nodes, ...edges];
}

function renderNodeDetails(data) {
  const lines = [];
  lines.push(`Node: ${data.id}`);
  lines.push(`Module: ${data.module_name}`);
  lines.push(`Type: ${data.module_type}`);
  if (data.call_index) lines.push(`Invocation: #${data.call_index}`);
  lines.push(`Parameter count: ${Number(data.parameter_count || 0).toLocaleString()}`);
  lines.push("");
  lines.push("Input shapes:");
  if (data.input_shapes.length === 0) lines.push("  (none)");
  else data.input_shapes.forEach((s) => lines.push(`  - ${s}`));
  lines.push("");
  lines.push("Output shapes:");
  if (data.output_shapes.length === 0) lines.push("  (none)");
  else data.output_shapes.forEach((s) => lines.push(`  - ${s}`));
  lines.push("");
  lines.push("Parameter dimensions:");
  if (data.parameter_shapes.length === 0) {
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

function renderEdgeDetails(data) {
  nodeDetails.textContent = [
    `Edge: ${data.source} -> ${data.target}`,
    `Kind: ${data.kind}`,
    "Observed shapes:",
    ...(data.shapes.length ? data.shapes.map((s) => `  - ${s}`) : ["  (none)"]),
  ].join("\n");
}

function renderGraph(payload) {
  const container = document.getElementById("graph");
  const elements = buildElements(payload);

  if (cy) cy.destroy();

  cy = cytoscape({
    container,
    elements,
    style: [
      {
        selector: "node",
        style: {
          label: "data(label)",
          color: "#e2e8f0",
          "font-size": 9,
          "text-wrap": "wrap",
          "text-max-width": 118,
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
          width: 1.6,
          "line-color": "#94a3b8",
          "target-arrow-color": "#94a3b8",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
        },
      },
      {
        selector: "edge.input",
        style: {
          "line-color": "#38bdf8",
          "target-arrow-color": "#38bdf8",
        },
      },
      {
        selector: "edge.functional_inferred",
        style: {
          "line-color": "#fb923c",
          "target-arrow-color": "#fb923c",
          "line-style": "dashed",
        },
      },
      {
        selector: "edge.terminal",
        style: {
          "line-color": "#22c55e",
          "target-arrow-color": "#22c55e",
        },
      },
    ],
    layout: {
      name: "dagre",
      rankDir: "LR",
      nodeSep: 28,
      rankSep: 85,
      edgeSep: 10,
      animate: false,
    },
    minZoom: 0.1,
    maxZoom: 2.5,
  });

  cy.on("tap", "node", (event) => renderNodeDetails(event.target.data()));
  cy.on("tap", "edge", (event) => renderEdgeDetails(event.target.data()));
}

async function releaseGpuMemory(showMessage = true) {
  try {
    const response = await fetch("/api/release-gpu", {
      method: "POST",
      keepalive: true,
    });
    if (!response.ok) {
      const body = await response.json();
      if (showMessage) setStatus(`GPU release failed: ${body.detail || response.statusText}`);
      return;
    }
    if (showMessage) setStatus("GPU cache release requested.");
  } catch (error) {
    if (showMessage) setStatus(`GPU release request failed: ${error}`);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("Tracing model execution...");
  nodeDetails.textContent = "Run visualization and click a node/edge for details.";

  const payload = {
    model_id_or_path: document.getElementById("model-id").value.trim(),
    sample_text: document.getElementById("sample-text").value,
    seq_len: Number(document.getElementById("seq-len").value),
    batch_size: Number(document.getElementById("batch-size").value),
    trust_remote_code: document.getElementById("trust-remote-code").checked,
    device: document.getElementById("device").value.trim(),
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
        `Device: ${result.model.device}`,
        `Executed unique modules: ${result.totals.executed_modules.toLocaleString()}`,
        `Executed calls: ${result.totals.executed_calls.toLocaleString()}`,
        `Total params: ${result.totals.parameters.toLocaleString()}`,
        `Trainable params: ${result.totals.trainable_parameters.toLocaleString()}`,
        `Warnings: ${result.warnings.length > 0 ? result.warnings.join(" | ") : "none"}`,
      ].join("\n"),
    );
  } catch (error) {
    setStatus(`Request failed: ${error}`);
  }
});

releaseGpuButton.addEventListener("click", async () => {
  await releaseGpuMemory(true);
});

window.addEventListener("beforeunload", () => {
  try {
    navigator.sendBeacon("/api/release-gpu", "");
  } catch (error) {
    // Best-effort cleanup only.
  }
});
