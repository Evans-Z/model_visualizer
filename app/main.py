from __future__ import annotations

import gc
import inspect
import json
import re
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from torch.fx import GraphModule, Tracer, symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from transformers import AutoModel, AutoTokenizer

try:  # pragma: no cover - depends on transformers version
    from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
except Exception:  # pragma: no cover - optional fallback
    hf_symbolic_trace = None


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
    device: str = Field(default="cpu", description="cpu, cuda, or cuda:<index>")
    graph_mode: str = Field(
        default="hybrid",
        description="module, operations, or hybrid (prefer operations fallback to module)",
    )

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


def normalize_device(device_value: str) -> tuple[torch.device, str]:
    normalized = device_value.lower().strip()
    if normalized == "cpu":
        return torch.device("cpu"), "cpu"

    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        index = torch.cuda.current_device()
        return torch.device(f"cuda:{index}"), f"cuda:{index}"

    if normalized.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        suffix = normalized.split(":", 1)[1]
        if not suffix.isdigit():
            raise ValueError("CUDA device must look like cuda:<index>")
        index = int(suffix)
        count = torch.cuda.device_count()
        if index < 0 or index >= count:
            raise ValueError(
                f"CUDA device index {index} is out of range. Available devices: 0..{count - 1}"
            )
        return torch.device(f"cuda:{index}"), f"cuda:{index}"

    raise ValueError("device must be 'cpu', 'cuda', or 'cuda:<index>'")


def release_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def infer_stage_index(name: str, sequence_order: int) -> int:
    patterns = [
        r"(?:^|\.)(?:layers?|blocks?|h)\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:encoder|decoder)\.layer\.(\d+)(?:\.|$)",
        r"(?:^|\.)(?:encoder|decoder)\.block\.(\d+)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    # Keep non-layered nodes deterministic but after indexed layers.
    return 10_000 + sequence_order


def node_shape_from_fx_meta(meta: Any) -> str:
    if meta is None:
        return "unknown"
    if isinstance(meta, (list, tuple)):
        return format_shape([node_shape_from_fx_meta(item) for item in meta])
    if hasattr(meta, "shape"):
        try:
            return format_shape(list(meta.shape))
        except Exception:
            return str(meta)
    return str(meta)


def split_inputs_for_signature(
    signature: inspect.Signature, inputs: dict[str, torch.Tensor]
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    args: list[torch.Tensor] = []
    kwargs: dict[str, torch.Tensor] = {}
    for name, param in signature.parameters.items():
        if name == "self" or name not in inputs:
            continue
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(inputs[name])
        else:
            kwargs[name] = inputs[name]
    return args, kwargs


def build_fx_graph_with_retries(
    model: torch.nn.Module,
    input_names: list[str],
) -> tuple[GraphModule, list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    try:
        return symbolic_trace(model), warnings
    except Exception as exc:
        errors.append(f"default symbolic_trace failed: {exc}")

    if hf_symbolic_trace is not None:
        try:
            traced = hf_symbolic_trace(
                model,
                input_names=input_names,
            )
            warnings.append("Using transformers.fx symbolic trace compatibility mode.")
            return traced, warnings
        except Exception as exc:
            errors.append(f"transformers.fx symbolic_trace failed: {exc}")

    # Fallback tracer: autowrap common Python builtins that appear in
    # model forward logic (for example len/range in cache handling).
    try:
        tracer = Tracer(
            autowrap_functions=(len, range, int, float, bool, max, min, abs),
        )
        graph = tracer.trace(model)
        warnings.append("Using fallback FX tracer with autowrapped builtins.")
        return GraphModule(model, graph), warnings
    except Exception as exc:
        errors.append(f"fallback tracer failed: {exc}")

    raise RuntimeError(" | ".join(errors))


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

    edge_shapes: dict[tuple[str, str, str], set[str]] = {}
    module_nodes: list[dict[str, Any]] = []
    module_call_count: dict[str, int] = {}
    functional_node_used = False
    sequence_order = 0
    tensor_producer: dict[int, str] = {}
    tensor_shape: dict[int, str] = {}
    input_tensor_ids = {id(tensor) for tensor in iter_tensors(inputs)}
    hooks = []

    for tensor in iter_tensors(inputs):
        tensor_shape[id(tensor)] = format_shape(list(tensor.shape))

    def add_edge(source: str, target: str, shape: str, kind: str):
        key = (source, target, kind)
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
            nonlocal functional_node_used
            nonlocal sequence_order

            current_call = module_call_count.get(module_name, 0) + 1
            module_call_count[module_name] = current_call
            call_node_id = f"{module_name}#{current_call}"
            node_order = sequence_order
            sequence_order += 1

            module_nodes.append(
                {
                    "id": call_node_id,
                    "label": f"{module_name}#{current_call}\n({module_type})",
                    "module_name": module_name,
                    "op_target": module_name,
                    "call_index": current_call,
                    "module_type": module_type,
                    "depth": module_name.count(".") + 1,
                    "call_count": 1,
                    "sequence_order": node_order,
                    "stage_index": infer_stage_index(module_name, node_order),
                    "input_shapes": [format_shape(shape_of({"args": args, "kwargs": kwargs}))],
                    "output_shapes": [format_shape(shape_of(output))],
                    "parameter_count": direct_parameter_count,
                    "parameter_shapes": direct_parameter_shapes,
                }
            )

            input_ids = collect_tensor_ids((args, kwargs))
            for tensor_id in input_ids:
                shape_repr = tensor_shape.get(tensor_id, "unknown")
                source = tensor_producer.get(tensor_id)
                if source is None:
                    if tensor_id in input_tensor_ids:
                        add_edge("__input__", call_node_id, shape_repr, "input")
                    else:
                        functional_node_used = True
                        add_edge(
                            "__functional__",
                            call_node_id,
                            shape_repr,
                            "functional_inferred",
                        )
                else:
                    if source != call_node_id:
                        add_edge(source, call_node_id, shape_repr, "observed")

            for tensor in iter_tensors(output):
                tid = id(tensor)
                shape_repr = format_shape(list(tensor.shape))
                tensor_producer[tid] = call_node_id
                tensor_shape[tid] = shape_repr

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

    try:
        with torch.no_grad():
            model_output = model(**inputs)
    finally:
        for hook in hooks:
            hook.remove()

    output_tensor_ids = set(collect_tensor_ids(model_output))
    for tensor_id in output_tensor_ids:
        shape_repr = tensor_shape.get(tensor_id, "unknown")
        source = tensor_producer.get(tensor_id)
        if source is not None:
            add_edge(source, "__output__", shape_repr, "terminal")
            continue
        if tensor_id in input_tensor_ids:
            add_edge("__input__", "__output__", shape_repr, "terminal")
            continue
        functional_node_used = True
        add_edge("__functional__", "__output__", shape_repr, "terminal")

    nodes: list[dict[str, Any]] = [
        {
            "id": "__input__",
            "label": "Input",
            "module_type": "Input",
            "depth": 0,
            "call_count": 1,
            "sequence_order": -2,
            "stage_index": -2,
            "input_shapes": [],
            "output_shapes": [format_shape(shape_of(inputs))],
            "parameter_count": 0,
            "parameter_shapes": [],
        }
    ]

    if functional_node_used:
        nodes.append(
            {
                "id": "__functional__",
                "label": "Functional / Uncaptured Ops",
                "module_type": "Functional",
                "depth": 0,
                "call_count": 1,
                "sequence_order": -1,
                "stage_index": -1,
                "input_shapes": [],
                "output_shapes": [],
                "parameter_count": 0,
                "parameter_shapes": [],
            }
        )

    nodes.extend(module_nodes)

    nodes.append(
        {
            "id": "__output__",
            "label": "Output",
            "module_type": "Output",
            "depth": 0,
            "call_count": 1,
            "sequence_order": sequence_order + 1,
            "stage_index": infer_stage_index("output", sequence_order + 1),
            "input_shapes": [],
            "output_shapes": [],
            "parameter_count": 0,
            "parameter_shapes": [],
        }
    )

    edges: list[dict[str, Any]] = []
    for (source, target, kind), shapes in sorted(edge_shapes.items()):
        edges.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "shapes": sorted(shapes),
                "count": len(shapes),
            }
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "totals": {
            "executed_modules": len({node["module_name"] for node in module_nodes}),
            "executed_calls": len(module_nodes),
            "parameters": total_parameters,
            "trainable_parameters": trainable_parameters,
        },
    }


def trace_operations_execution(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    signature: inspect.Signature,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    traced, trace_warnings = build_fx_graph_with_retries(model, list(inputs.keys()))
    warnings.extend(trace_warnings)

    args, kwargs = split_inputs_for_signature(signature, inputs)
    try:
        ShapeProp(traced).propagate(*args, **kwargs)
    except Exception as exc:
        warnings.append(f"Shape propagation unavailable for operation graph: {exc}")

    total_parameters = int(sum(param.numel() for param in model.parameters()))
    trainable_parameters = int(
        sum(param.numel() for param in model.parameters() if param.requires_grad)
    )

    nodes: list[dict[str, Any]] = [
        {
            "id": "__input__",
            "label": "Input",
            "module_name": "__input__",
            "op_target": "__input__",
            "call_index": 1,
            "module_type": "Input",
            "depth": 0,
            "call_count": 1,
            "sequence_order": -2,
            "stage_index": -2,
            "input_shapes": [],
            "output_shapes": [format_shape(shape_of(inputs))],
            "parameter_count": 0,
            "parameter_shapes": [],
        }
    ]
    edges_map: dict[tuple[str, str, str], set[str]] = {}

    def add_edge(source: str, target: str, shape: str, kind: str):
        key = (source, target, kind)
        if key not in edges_map:
            edges_map[key] = set()
        edges_map[key].add(shape)

    node_id_by_name: dict[str, str] = {}
    output_sources: list[str] = []
    executed_module_names: set[str] = set()
    sequence_order = 0

    for fx_node in traced.graph.nodes:
        if fx_node.op == "placeholder":
            node_id = f"fx:input:{fx_node.name}"
            node_id_by_name[fx_node.name] = node_id
            input_tensor = inputs.get(fx_node.name)
            output_shape = (
                format_shape(list(input_tensor.shape))
                if input_tensor is not None
                else node_shape_from_fx_meta(fx_node.meta.get("tensor_meta"))
            )
            nodes.append(
                {
                    "id": node_id,
                    "label": fx_node.name,
                    "module_name": fx_node.name,
                    "op_target": fx_node.name,
                    "call_index": 1,
                    "module_type": "InputArg",
                    "depth": 1,
                    "call_count": 1,
                    "sequence_order": sequence_order,
                    "stage_index": -1,
                    "input_shapes": [],
                    "output_shapes": [output_shape],
                    "parameter_count": 0,
                    "parameter_shapes": [],
                }
            )
            add_edge("__input__", node_id, output_shape, "input")
            sequence_order += 1
            continue

        if fx_node.op == "output":
            output_sources = [node_id_by_name[src.name] for src in fx_node.all_input_nodes]
            continue

        node_id = f"fx:{fx_node.name}"
        node_id_by_name[fx_node.name] = node_id

        parameter_shapes: list[dict[str, Any]] = []
        parameter_count = 0
        module_type = fx_node.op
        op_target = str(fx_node.target)
        module_name = str(fx_node.target)

        if fx_node.op == "call_module":
            submodule = traced.get_submodule(str(fx_node.target))
            module_type = submodule.__class__.__name__
            module_name = str(fx_node.target)
            op_target = module_name
            executed_module_names.add(module_name)
            for param_name, param in submodule.named_parameters(recurse=False):
                parameter_shapes.append(
                    {"name": param_name, "shape": list(param.shape), "numel": int(param.numel())}
                )
                parameter_count += int(param.numel())
        elif fx_node.op == "call_function":
            target_name = getattr(fx_node.target, "__name__", str(fx_node.target))
            module_type = "FunctionOp"
            op_target = target_name
            module_name = fx_node.name
        elif fx_node.op == "call_method":
            module_type = "MethodOp"
            op_target = str(fx_node.target)
            module_name = fx_node.name
        elif fx_node.op == "get_attr":
            module_type = "GetAttr"
            op_target = str(fx_node.target)
            module_name = str(fx_node.target)

        output_shape = node_shape_from_fx_meta(fx_node.meta.get("tensor_meta"))
        nodes.append(
            {
                "id": node_id,
                "label": f"{fx_node.name}\n({module_type})",
                "module_name": module_name,
                "op_target": op_target,
                "call_index": 1,
                "module_type": module_type,
                "depth": module_name.count(".") + 1,
                "call_count": 1,
                "sequence_order": sequence_order,
                "stage_index": infer_stage_index(module_name, sequence_order),
                "input_shapes": [],
                "output_shapes": [output_shape],
                "parameter_count": parameter_count,
                "parameter_shapes": parameter_shapes,
            }
        )

        for source_node in fx_node.all_input_nodes:
            source_id = node_id_by_name.get(source_node.name)
            if source_id is None:
                continue
            src_shape = node_shape_from_fx_meta(source_node.meta.get("tensor_meta"))
            add_edge(source_id, node_id, src_shape, "observed")

        sequence_order += 1

    max_stage = max((node.get("stage_index", 0) for node in nodes), default=0)
    nodes.append(
        {
            "id": "__output__",
            "label": "Output",
            "module_name": "__output__",
            "op_target": "__output__",
            "call_index": 1,
            "module_type": "Output",
            "depth": 0,
            "call_count": 1,
            "sequence_order": sequence_order + 1,
            "stage_index": max_stage + 1,
            "input_shapes": [],
            "output_shapes": [],
            "parameter_count": 0,
            "parameter_shapes": [],
        }
    )

    node_ids = {node["id"] for node in nodes}
    for source_id in output_sources:
        if source_id in node_ids:
            add_edge(source_id, "__output__", "terminal", "terminal")

    edges: list[dict[str, Any]] = []
    for (source, target, kind), shapes in sorted(edges_map.items()):
        edges.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "shapes": sorted(shapes),
                "count": len(shapes),
            }
        )

    return (
        {
            "nodes": nodes,
            "edges": edges,
            "totals": {
                "executed_modules": len(executed_module_names),
                "executed_calls": max(len(nodes) - 2, 0),
                "parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            },
        },
        warnings,
    )


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


@app.post("/api/release-gpu")
def release_gpu():
    release_gpu_memory()
    return {"status": "released"}


@app.post("/api/visualize")
def visualize(request: VisualizeRequest):
    model = None
    inputs: dict[str, torch.Tensor] = {}
    device_obj = torch.device("cpu")
    warnings: list[str] = []
    graph: dict[str, Any] | None = None
    normalized_device = "cpu"
    graph_mode_used = "module"

    try:
        try:
            device_obj, normalized_device = normalize_device(request.device)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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

        model.to(device_obj)
        signature = inspect.signature(model.forward)

        try:
            inputs, warnings = build_inputs(model, request, signature)
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to build inputs: {exc}"
            ) from exc

        inputs = {key: value.to(device_obj) for key, value in inputs.items()}

        graph_mode_requested = request.graph_mode.lower().strip()
        if graph_mode_requested not in {"module", "operations", "hybrid"}:
            raise HTTPException(
                status_code=400,
                detail="graph_mode must be 'module', 'operations', or 'hybrid'",
            )

        if graph_mode_requested in {"operations", "hybrid"}:
            try:
                graph, op_warnings = trace_operations_execution(
                    model=model,
                    inputs=inputs,
                    signature=signature,
                )
                warnings.extend(op_warnings)
                graph_mode_used = "operations"
            except Exception as exc:
                warnings.append(f"Operation-level graph failed; falling back to module graph. ({exc})")
                if graph_mode_requested == "operations":
                    raise HTTPException(
                        status_code=500,
                        detail=f"Operation graph generation failed: {exc}",
                    ) from exc

        if graph is None:
            try:
                graph = trace_model_execution(model=model, inputs=inputs)
                graph_mode_used = "module"
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Model trace failed: {exc}") from exc

        return {
            "model": {
                "id_or_path": request.model_id_or_path,
                "class": model.__class__.__name__,
                "device": normalized_device,
            },
            "graph_mode_requested": graph_mode_requested,
            "graph_mode_used": graph_mode_used,
            "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
            "warnings": warnings,
            **graph,
        }
    finally:
        if graph is not None:
            del graph
        if inputs:
            del inputs
        if model is not None:
            del model
        if device_obj.type == "cuda":
            release_gpu_memory()
