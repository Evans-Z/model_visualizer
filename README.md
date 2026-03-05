# Hugging Face Model Visualizer (MVP)

This project visualizes the **real forward execution path** of a Hugging Face model and shows:

- layer/module data flow,
- activation/input/output shapes,
- parameter dimensions per executed module,
- total/trainable parameter counts.

The app traces execution by running one forward pass with auto-generated sample inputs.

## Features

- **Backend**: FastAPI endpoint that loads a model and traces executed leaf modules.
- **Graph extraction**: tensor provenance tracking to build directed data-flow edges per module invocation.
- **Operation-level tracing**: optional Torch FX graph to include intermediate operation nodes (e.g. matmul/softmax/getitem) when traceable.
- **Frontend**: browser UI with an interactive architecture graph and layer detail panel.
- **Simple deployment**: one Python service that serves both API and static web page.
- **Device targeting**: supports `cpu`, `cuda`, and `cuda:<index>` (e.g. `cuda:0`, `cuda:1`).

## Quick start

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Run the app

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3) Open in browser

```text
http://localhost:8000
```

## API

### `POST /api/visualize`

Request example:

```json
{
  "model_id_or_path": "distilbert-base-uncased",
  "sample_text": "Hello architecture visualizer",
  "batch_size": 1,
  "seq_len": 16,
  "trust_remote_code": false,
  "device": "cuda:0",
  "graph_mode": "hybrid"
}
```

Response contains:

- `nodes`: module-invocation nodes (plus Input/Output/Functional pseudo nodes),
- `edges`: observed and inferred data flow edges (`input`, `observed`, `functional_inferred`, `terminal`),
- `input_shapes`, `warnings`,
- `totals.executed_modules`, `totals.executed_calls`,
- `totals.parameters`, `totals.trainable_parameters`.
- `graph_mode_requested`, `graph_mode_used`.

`graph_mode` values:

- `module`: module invocation graph from runtime hooks,
- `operations`: operation graph from Torch FX only (fails if model is not FX-traceable),
- `hybrid`: try operation graph first, fallback to module graph.

In `hybrid` mode the backend retries operation tracing with multiple strategies
(default FX, `transformers.utils.fx` compatibility tracer, a fallback tracer with
autowrapped Python builtins such as `len`/`range`, and runtime operator tracing
via `make_fx`) before falling back.

### `POST /api/release-gpu`

Best-effort GPU cache cleanup (`gc.collect()`, `torch.cuda.empty_cache()`, `torch.cuda.ipc_collect()`).
The UI also calls this automatically when you press **Release GPU memory** or close the tab.

## Notes and limitations

- The graph reflects execution for the **provided sample input**, which is usually what you want for "real" flow.
- This MVP traces **leaf modules** (e.g., Linear, LayerNorm, Attention internals implemented as modules). Pure functional ops not wrapped in modules may not appear as separate nodes.
- In `operations`/`hybrid` mode, the graph can include intermediate operation nodes and often exposes attention internals (for example q/k/v-related computation paths) when symbolic trace succeeds.
- When a tensor arrives from non-leaf/functional code, the graph routes it through a `Functional / Uncaptured Ops` pseudo node instead of incorrectly showing it as a direct model input edge.
- Input generation is automatic (tokenizer first, then random fallback). Some highly custom models may require extending input builders.
- `trust_remote_code=true` can execute remote model code; only enable in trusted environments.