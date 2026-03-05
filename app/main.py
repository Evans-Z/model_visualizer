from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer


class VisualizeRequest(BaseModel):
    model_id_or_path: str = Field(
        ...,
        description="Hugging Face model id or local path",
        examples=["distilbert-base-uncased"],
    )
    sample_text: str = Field(
        default="Hello from model visualizer",
        description="Text used to create sample inputs for text models",
    )
    batch_size: int = Field(default=1, ge=1, le=8)
    seq_len: int = Field(default=16, ge=1, le=512)
    trust_remote_code: bool = Field(default=False)
    device: str = Field(default="cpu", description="cpu or cuda")


@dataclass
class NodeStats:
    name: str
    module_type: str
    depth: int
    order: int
    call_count: int
    input_shapes: set[str]
    output_shapes: set[str]
    parameter_shapes: list[dict[str, Any]]
    parameter_count: int


def iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_tensors(item)


def shape_of(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (list, tuple)):
        return [shape_of(item) for item in value]
    if isinstance(value, dict):
        return {key: shape_of(item) for key, item in value.items()}
    return type(value).__name__


def collect_tensor_ids(value: Any) -> list[int]:
    return [id(tensor) for tensor in iter_tensors(value)]


def format_shape(shape: Any) -> str:
    return json.dumps(shape, separators=(",", ":"))


def _accepts_kwargs(signature: inspect.Signature) -> bool:
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )


def filter_for_forward(
    signature: inspect.Signature,
    candidate_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if _accepts_kwargs(signature):
        return candidate_inputs
    valid = set(signature.parameters.keys())
    return {key: value for key, value in candidate_inputs.items() if key in valid}


def build_fallback_inputs(
    model: torch.nn.Module,
    signature: inspect.Signature,
    batch_size: int,
    seq_len: int,
) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {}
    config = model.config
    fields = set(signature.parameters.keys())

    if "pixel_values" in fields:
        channels = int(getattr(config, "num_channels", 3))
        image_size = getattr(config, "image_size", 224)
        if isinstance(image_size, (tuple, list)) and len(image_size) == 2:
            height, width = int(image_size[0]), int(image_size[1])
        else:
            height = width = int(image_size)
        inputs["pixel_values"] = torch.randn(batch_size, channels, height, width)

    if "input_features" in fields:
        # Typical mel bins for many speech models.
        feature_size = int(getattr(config, "num_mel_bins", 80))
        inputs["input_features"] = torch.randn(batch_size, feature_size, seq_len)

    if "input_ids" in fields and "input_ids" not in inputs:
        vocab_size = int(getattr(config, "vocab_size", 32000))
        vocab_size = max(vocab_size, 2)
        inputs["input_ids"] = torch.randint(0, vocab_size, (batch_size, seq_len))
        if "attention_mask" in fields:
            inputs["attention_mask"] = torch.ones(batch_size, seq_len, dtype=torch.long)

    if (
        getattr(config, "is_encoder_decoder", False)
        and "decoder_input_ids" in fields
        and "decoder_input_ids" not in inputs
    ):
        decoder_start_token_id = getattr(config, "decoder_start_token_id", None)
        bos_token_id = getattr(config, "bos_token_id", None)
        pad_token_id = getattr(config, "pad_token_id", None)
        token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else bos_token_id
            if bos_token_id is not None
            else pad_token_id
            if pad_token_id is not None
            else 0
        )
        inputs["decoder_input_ids"] = torch.full(
            (batch_size, 1), int(token_id), dtype=torch.long
        )

    if not inputs:
        hidden_size = int(getattr(config, "hidden_size", 768))
        for name, param in signature.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                inputs[name] = torch.randn(batch_size, seq_len, hidden_size)
                break

    return inputs


def build_inputs(
    model: torch.nn.Module,
    request: VisualizeRequest,
    signature: inspect.Signature,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    warnings: list[str] = []
    fields = set(signature.parameters.keys())
    tokenizable_fields = {"input_ids", "attention_mask", "token_type_ids"}
    can_use_tokenizer = bool(fields & tokenizable_fields)

    if can_use_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                request.model_id_or_path,
                trust_remote_code=request.trust_remote_code,
            )
            encoded = tokenizer(
                request.sample_text,
                return_tensors="pt",
                truncation=True,
                max_length=request.seq_len,
                padding="max_length",
            )
            tokenized = {
                key: value.repeat(request.batch_size, 1)
                if value.shape[0] == 1 and request.batch_size > 1
                else value
                for key, value in encoded.items()
            }
            if (
                getattr(model.config, "is_encoder_decoder", False)
                and "decoder_input_ids" in fields
                and "decoder_input_ids" not in tokenized
            ):
                start_id = (
                    getattr(model.config, "decoder_start_token_id", None)
                    or getattr(model.config, "bos_token_id", None)
                    or getattr(model.config, "pad_token_id", None)
                    or 0
                )
                tokenized["decoder_input_ids"] = torch.full(
                    (request.batch_size, 1), int(start_id), dtype=torch.long
                )

            tokenized = filter_for_forward(signature, tokenized)
            if tokenized:
                return tokenized, warnings
        except Exception as exc:  # pragma: no cover - behavior depends on model
            warnings.append(
                f"Tokenizer-based input generation failed; using random fallback. ({exc})"
            )

    fallback = build_fallback_inputs(
        model=model,
        signature=signature,
        batch_size=request.batch_size,
        seq_len=request.seq_len,
    )
    fallback = filter_for_forward(signature, fallback)
    if not fallback:
        raise ValueError(
            "Could not infer model inputs automatically. "
            "Try a different model or extend input generation logic."
        )
    return fallback, warnings


def trace_model_execution(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    model.eval()
    total_parameters = int(sum(param.numel() for param in model.parameters()))
    trainable_parameters = int(
        sum(param.numel() for param in model.parameters() if param.requires_grad)
    )

    stats: dict[str, NodeStats] = {}
    order_counter = 0
    edge_shapes: dict[tuple[str, str], set[str]] = {}
    all_produced_ids: set[int] = set()
    consumed_ids: set[int] = set()
    tensor_producer: dict[int, str] = {}
    tensor_shape: dict[int, str] = {}
    hooks = []

    for tensor in iter_tensors(inputs):
        tensor_shape[id(tensor)] = format_shape(list(tensor.shape))

    def add_edge(source: str, target: str, shape: str):
        key = (source, target)
        if key not in edge_shapes:
            edge_shapes[key] = set()
        edge_shapes[key].add(shape)

    for name, module in model.named_modules():
        if name == "":
            continue
        if any(True for _ in module.children()):
            continue

        direct_params = []
        param_count = 0
        for param_name, param in module.named_parameters(recurse=False):
            shape = list(param.shape)
            numel = int(param.numel())
            direct_params.append({"name": param_name, "shape": shape, "numel": numel})
            param_count += numel

        def hook_fn(
            current_module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
            *,
            module_name: str = name,
            module_type: str = module.__class__.__name__,
            direct_parameter_shapes: list[dict[str, Any]] = direct_params,
            direct_parameter_count: int = param_count,
        ):
            nonlocal order_counter

            if module_name not in stats:
                stats[module_name] = NodeStats(
                    name=module_name,
                    module_type=module_type,
                    depth=module_name.count("."),
                    order=order_counter,
                    call_count=0,
                    input_shapes=set(),
                    output_shapes=set(),
                    parameter_shapes=direct_parameter_shapes,
                    parameter_count=direct_parameter_count,
                )
                order_counter += 1

            node = stats[module_name]
            node.call_count += 1
            node.input_shapes.add(format_shape(shape_of({"args": args, "kwargs": kwargs})))
            node.output_shapes.add(format_shape(shape_of(output)))

            input_ids = collect_tensor_ids((args, kwargs))
            for tensor_id in input_ids:
                shape_repr = tensor_shape.get(tensor_id, "unknown")
                source = tensor_producer.get(tensor_id)
                if source is None:
                    add_edge("__input__", module_name, shape_repr)
                else:
                    consumed_ids.add(tensor_id)
                    if source != module_name:
                        add_edge(source, module_name, shape_repr)

            for tensor in iter_tensors(output):
                tid = id(tensor)
                shape_repr = format_shape(list(tensor.shape))
                tensor_producer[tid] = module_name
                tensor_shape[tid] = shape_repr
                all_produced_ids.add(tid)

        try:
            hooks.append(module.register_forward_hook(hook_fn, with_kwargs=True))
        except TypeError:
            # Compatibility path for older torch versions without with_kwargs support.
            def compat_hook(
                current_module: torch.nn.Module,
                args: tuple[Any, ...],
                output: Any,
                *,
                wrapped_hook=hook_fn,
            ):
                wrapped_hook(current_module, args, {}, output)

            hooks.append(module.register_forward_hook(compat_hook))

    with torch.no_grad():
        _ = model(**inputs)

    for hook in hooks:
        hook.remove()

    terminal_ids = all_produced_ids - consumed_ids
    terminal_modules = {tensor_producer[tid] for tid in terminal_ids if tid in tensor_producer}
    for module_name in terminal_modules:
        add_edge(module_name, "__output__", "terminal")

    nodes: list[dict[str, Any]] = [
        {
            "id": "__input__",
            "label": "Input",
            "module_type": "Input",
            "depth": 0,
            "call_count": 1,
            "input_shapes": [],
            "output_shapes": [format_shape(shape_of(inputs))],
            "parameter_count": 0,
            "parameter_shapes": [],
        }
    ]

    for node in sorted(stats.values(), key=lambda item: item.order):
        nodes.append(
            {
                "id": node.name,
                "label": f"{node.name}\n({node.module_type})",
                "module_type": node.module_type,
                "depth": node.depth + 1,
                "call_count": node.call_count,
                "input_shapes": sorted(node.input_shapes),
                "output_shapes": sorted(node.output_shapes),
                "parameter_count": node.parameter_count,
                "parameter_shapes": node.parameter_shapes,
            }
        )

    nodes.append(
        {
            "id": "__output__",
            "label": "Output",
            "module_type": "Output",
            "depth": 0,
            "call_count": 1,
            "input_shapes": [],
            "output_shapes": [],
            "parameter_count": 0,
            "parameter_shapes": [],
        }
    )

    edges: list[dict[str, Any]] = []
    for (source, target), shapes in sorted(edge_shapes.items()):
        edges.append(
            {
                "source": source,
                "target": target,
                "shapes": sorted(shapes),
                "count": len(shapes),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "totals": {
            "executed_modules": len(stats),
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
    }


app = FastAPI(title="Hugging Face Model Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def read_index():
    return FileResponse("app/static/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/visualize")
def visualize(request: VisualizeRequest):
    device = request.device.lower().strip()
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA requested but not available")

    try:
        model = AutoModel.from_pretrained(
            request.model_id_or_path,
            trust_remote_code=request.trust_remote_code,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Failed to load model '{request.model_id_or_path}'. "
                "If this model requires custom code, enable trust_remote_code. "
                f"Original error: {exc}"
            ),
        ) from exc

    model.to(device)
    signature = inspect.signature(model.forward)

    try:
        inputs, warnings = build_inputs(model, request, signature)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to build inputs: {exc}") from exc

    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        graph = trace_model_execution(model=model, inputs=inputs)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model trace failed: {exc}") from exc

    return {
        "model": {
            "id_or_path": request.model_id_or_path,
            "class": model.__class__.__name__,
            "device": device,
        },
        "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "warnings": warnings,
        **graph,
    }
