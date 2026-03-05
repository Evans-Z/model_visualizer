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

function clearGraph() {
  if (cy) {
    cy.destroy();
    cy = null;
  }
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
        op_target: node.op_target || node.module_name || node.id,
        call_index: node.call_index || null,
        module_type: node.module_type,
        sequence_order: Number.isFinite(node.sequence_order) ? node.sequence_order : 0,
        stage_index: Number.isFinite(node.stage_index) ? node.stage_index : null,
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

function buildPresetPositions(elements) {
  const nodeElements = elements.filter((element) => !Object.prototype.hasOwnProperty.call(element.data, "source"));
  const nodesSorted = [...nodeElements].sort((a, b) => {
    const stageA = Number.isFinite(a.data.stage_index) ? a.data.stage_index : Number.MAX_SAFE_INTEGER;
    const stageB = Number.isFinite(b.data.stage_index) ? b.data.stage_index : Number.MAX_SAFE_INTEGER;
    if (stageA !== stageB) return stageA - stageB;
    return (a.data.sequence_order || 0) - (b.data.sequence_order || 0);
  });

  const stageValues = nodesSorted
    .map((node) => node.data.stage_index)
    .filter((stage) => Number.isFinite(stage) && stage > -2);
  const uniqueStageValues = [...new Set(stageValues)].sort((a, b) => a - b);
  const stageToRank = new Map(uniqueStageValues.map((stage, idx) => [stage, idx + 1]));
  let fallbackStage = uniqueStageValues.length + 2;
  const stageCounts = new Map();
  const positions = {};

  for (const node of nodesSorted) {
    let stage;
    if (node.data.id === "__input__") {
      stage = 0;
    } else if (node.data.id === "__output__") {
      stage = uniqueStageValues.length + 3;
    } else if (Number.isFinite(node.data.stage_index)) {
      stage = stageToRank.get(node.data.stage_index) ?? fallbackStage;
    } else {
      stage = fallbackStage;
    }
    fallbackStage = Math.max(fallbackStage, stage + 1);

    const row = stageCounts.get(stage) || 0;
    stageCounts.set(stage, row + 1);
    positions[node.data.id] = {
      x: 220 * stage,
      y: 80 + row * 90,
    };
  }

  return positions;
}

function renderNodeDetails(data) {
  const lines = [];
  lines.push(`Node: ${data.id}`);
  lines.push(`Module: ${data.module_name}`);
  lines.push(`Target: ${data.op_target}`);
  lines.push(`Type: ${data.module_type}`);
  if (data.call_index) lines.push(`Invocation: #${data.call_index}`);
  lines.push(`Stage index: ${data.stage_index}`);
  lines.push(`Sequence order: ${data.sequence_order}`);
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
  const positions = buildPresetPositions(elements);

  clearGraph();

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
      {
        selector: "edge.compressed_key",
        style: {
          "line-color": "#fbbf24",
          "target-arrow-color": "#fbbf24",
          "line-style": "dashed",
        },
      },
    ],
    layout: { name: "preset", positions, animate: false },
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
  clearGraph();
  setStatus("Tracing model execution...");
  nodeDetails.textContent = "Run visualization and click a node/edge for details.";

  const payload = {
    model_id_or_path: document.getElementById("model-id").value.trim(),
    sample_text: document.getElementById("sample-text").value,
    seq_len: Number(document.getElementById("seq-len").value),
    batch_size: Number(document.getElementById("batch-size").value),
    trust_remote_code: document.getElementById("trust-remote-code").checked,
    device: document.getElementById("device").value.trim(),
    graph_mode: document.getElementById("graph-mode").value,
    operation_detail: document.getElementById("operation-detail").value,
  };

  try {
    const response = await fetch("/api/visualize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      clearGraph();
      setStatus(`Error: ${result.detail || "Unknown failure"}`);
      return;
    }

    renderGraph(result);
    setStatus(
      [
        `Model: ${result.model.id_or_path} (${result.model.class})`,
        `Device: ${result.model.device}`,
        `Graph mode requested: ${result.graph_mode_requested}`,
        `Graph mode used: ${result.graph_mode_used}`,
        `Operation detail requested: ${result.operation_detail_requested}`,
        `Operation detail used: ${result.operation_detail_used}`,
        `Executed unique modules: ${result.totals.executed_modules.toLocaleString()}`,
        `Executed calls: ${result.totals.executed_calls.toLocaleString()}`,
        `Total params: ${result.totals.parameters.toLocaleString()}`,
        `Trainable params: ${result.totals.trainable_parameters.toLocaleString()}`,
        `Warnings: ${result.warnings.length > 0 ? result.warnings.join(" | ") : "none"}`,
      ].join("\n"),
    );
  } catch (error) {
    clearGraph();
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
