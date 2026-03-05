# Hugging Face Model Visualizer (MVP)

This project visualizes the **real forward execution path** of a Hugging Face model and shows:

- layer/module data flow,
- activation/input/output shapes,
- parameter dimensions per executed module,
- total/trainable parameter counts.

The app traces execution by running one forward pass with auto-generated sample inputs.

## Features

- **Backend**: FastAPI endpoint that loads a model and traces executed leaf modules.
- **Graph extraction**: tensor provenance tracking to build directed data-flow edges.
- **Frontend**: browser UI with an interactive architecture graph and layer detail panel.
- **Simple deployment**: one Python service that serves both API and static web page.

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
  "device": "cpu"
}
```

Response contains:

- `nodes`: modules (plus Input/Output pseudo nodes),
- `edges`: observed data flow edges,
- `input_shapes`, `warnings`,
- `totals.parameters`, `totals.trainable_parameters`.

## Notes and limitations

- The graph reflects execution for the **provided sample input**, which is usually what you want for "real" flow.
- This MVP traces **leaf modules** (e.g., Linear, LayerNorm, Attention internals implemented as modules). Pure functional ops not wrapped in modules may not appear as separate nodes.
- Input generation is automatic (tokenizer first, then random fallback). Some highly custom models may require extending input builders.
- `trust_remote_code=true` can execute remote model code; only enable in trusted environments.