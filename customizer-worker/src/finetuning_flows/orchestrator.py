"""Recipe orchestration — compose public module APIs only."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from src.finetuning_modules.compatibility.recipes import get_recipe_spec
from src.finetuning_modules.inference_testers import BatchInferenceTester
from src.utils.logcarrier import LogCarrier
from src.utils.progress import make_train_progress_callback, progress_notify


def _hf_repo_url(repo_id: str) -> str:
    return f"https://huggingface.co/{repo_id.strip().strip('/')}"


def _resolve_hf_token(export_cfg: dict[str, Any]) -> Optional[str]:
    token = export_cfg.get("hf_token") or export_cfg.get("token")
    if token:
        return str(token)
    from src.core.config import get_settings

    return get_settings().hf_token or None


def _resolve_hf_repo(payload: dict[str, Any], job_uuid: str) -> str:
    """Explicit export.hf_repo wins; otherwise {HF_USERNAME}/{job_uuid} when token+username set."""
    export_cfg = dict(payload.get("export") or {})
    explicit = str(export_cfg.get("hf_repo") or "").strip()
    if explicit:
        return explicit

    from src.core.config import get_settings

    settings = get_settings()
    username = str(getattr(settings, "hf_username", "") or "").strip().strip("/")
    if not username or not _resolve_hf_token(export_cfg):
        return ""
    return f"{username}/{job_uuid}"


def _ensure_hf_repo_on_payload(payload: dict[str, Any], job_uuid: str) -> str:
    """Resolve and write hf_repo onto payload.export so preflight/export stay consistent."""
    repo_id = _resolve_hf_repo(payload, job_uuid)
    if not repo_id:
        return ""
    export_cfg = dict(payload.get("export") or {})
    export_cfg["hf_repo"] = repo_id
    payload["export"] = export_cfg
    return repo_id


def _notify_hf_push_started(
    job_uuid: str, repo_id: str, export_dest: str, log: LogCarrier
) -> None:
    """Best-effort progress webhook emitted immediately before the Hub upload."""
    progress_notify(
        job_uuid,
        "RUNNING",
        phase=f"Publishing model to Hugging Face ({repo_id})",
        patch={
            "export": {
                "export_dest": export_dest,
                "hf_push": {
                    "status": "pushing",
                    "repo_id": repo_id,
                    "url": None,
                },
            }
        },
        force=True,
    )


def _preflight_hf_push(
    payload: dict[str, Any], *, dry_run: bool, log: LogCarrier, job_uuid: str = ""
) -> Optional[dict[str, Any]]:
    """Validate credentials before expensive training when an HF push is requested."""
    uuid = job_uuid or str(payload.get("job_uuid") or "")
    repo_id = _ensure_hf_repo_on_payload(payload, uuid)
    if not repo_id:
        return None
    if dry_run:
        return {"status": "requested", "repo_id": repo_id, "url": None}

    export_cfg = dict(payload.get("export") or {})
    token = _resolve_hf_token(export_cfg)
    if not token:
        return {
            "status": "failed",
            "repo_id": repo_id,
            "url": None,
            "error": "HF_TOKEN is required when a Hugging Face export repository is requested",
        }

    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.whoami(token=token)
        api.create_repo(
            repo_id=repo_id,
            token=token,
            private=bool(export_cfg.get("private", True)),
            exist_ok=True,
            repo_type="model",
        )
    except Exception as exc:
        log.warning("hf_token_validation_failed", repo_id=repo_id, error=str(exc))
        return {
            "status": "failed",
            "repo_id": repo_id,
            "url": None,
            "error": "HF_TOKEN validation failed; check that the token is valid and can write to the requested model repository",
        }
    return {"status": "pushing", "repo_id": repo_id, "url": None}



def _write_json_best_effort(path: str, data: Any, log: LogCarrier) -> Optional[str]:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return path
    except OSError as exc:
        log.warning("json_report_write_failed", path=path, error=str(exc))
        return None

def _job_minio_prefix(job_uuid: str) -> str:
    return f"minio://jobs/{job_uuid}"


def _load_dataset(payload: dict[str, Any]):
    from src.finetuning_modules.dataset_loaders import ExternalDatasetLoader, MinioDatasetLoader

    minio_uri = payload.get("dataset_minio_uri") or payload.get("datasetMinioUri")
    hf_dataset = payload.get("hf_dataset") or payload.get("hfDataset")
    if minio_uri:
        return MinioDatasetLoader.load(minio_uri)
    if hf_dataset:
        path = hf_dataset.get("path") if isinstance(hf_dataset, dict) else hf_dataset
        split = hf_dataset.get("split") if isinstance(hf_dataset, dict) else "train"
        name = hf_dataset.get("name") if isinstance(hf_dataset, dict) else None
        return ExternalDatasetLoader.load(path=path, split=split, name=name)
    raise ValueError("Job requires dataset_minio_uri or hf_dataset")


_SFT_TEXT_COLUMNS = (
    "review",
    "content",
    "body",
    "comment",
    "description",
    "document",
    "doc",
    "sentence",
    "passage",
    "article",
)
_SFT_LABEL_COLUMNS = (
    "star",
    "stars",
    "rating",
    "label",
    "score",
    "sentiment",
    "class",
    "category",
)
_SFT_CONTEXT_COLUMNS = (
    "package_name",
    "app_name",
    "product",
    "product_name",
    "title",
    "name",
    "item",
)
_SFT_QA_PAIRS = (
    ("question", "answer"),
    ("query", "response"),
    ("query", "answer"),
    ("input", "output"),
)


def _pick_column(columns: set[str], candidates: tuple[str, ...]) -> Optional[str]:
    lower_map = {str(name).lower(): name for name in columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _pick_default_chat_template(model_id: str = "") -> str:
    mid = (model_id or "").lower()
    if "gemma-3" in mid or "gemma3" in mid:
        return "gemma-3"
    if "gemma" in mid:
        return "gemma"
    if "llama-3.1" in mid or "llama3.1" in mid:
        return "llama-3.1"
    if "llama-3" in mid or "llama3" in mid:
        return "llama-3"
    if "qwen" in mid:
        return "qwen-2.5"
    if "mistral" in mid:
        return "mistral"
    if "phi-4" in mid or "phi4" in mid:
        return "phi-4"
    if "phi-3" in mid or "phi3" in mid:
        return "phi-3"
    return "chatml"


def _ensure_chat_template(tokenizer: Any, model_id: str = "") -> Any:
    """Attach a chat template when the tokenizer shipped without one (common on base models)."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer

    template_name = _pick_default_chat_template(model_id)
    try:
        from unsloth.chat_templates import get_chat_template

        tokenizer = get_chat_template(tokenizer, chat_template=template_name)
        if getattr(tokenizer, "chat_template", None):
            return tokenizer
    except Exception:
        pass

    # Minimal ChatML jinja so apply_chat_template works without Unsloth.
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] "
        "+ '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )
    return tokenizer


def _format_messages_fallback(messages: list[dict[str, str]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", ""))
        parts.append(f"### {role.capitalize()}:\n{content}")
    return "\n\n".join(parts).rstrip() + "\n"


def _apply_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass
    return _format_messages_fallback(messages)


def _prepare_sft_dataset(dataset: Any, tokenizer: Any) -> Any:
    """Normalize common SFT layouts (chat, Alpaca, reviews+labels, …) to a text column."""
    columns = set(getattr(dataset, "column_names", []) or [])
    if "text" in columns:
        return dataset

    if {"instruction", "output"}.issubset(columns):
        def format_alpaca(row: dict[str, Any]) -> dict[str, str]:
            prompt = str(row.get("instruction", ""))
            if row.get("input"):
                prompt += "\n\n" + str(row["input"])
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": str(row.get("output", ""))},
            ]
            return {"text": _apply_chat(tokenizer, messages)}
        return dataset.map(format_alpaca)

    if {"prompt", "completion"}.issubset(columns):
        return dataset.map(lambda row: {"text": f"{row['prompt']}{row['completion']}"})

    for left, right in _SFT_QA_PAIRS:
        if {left, right}.issubset({c.lower() for c in columns}):
            q_col = _pick_column(columns, (left,))
            a_col = _pick_column(columns, (right,))

            def format_qa(row: dict[str, Any], q=q_col, a=a_col) -> dict[str, str]:
                messages = [
                    {"role": "user", "content": str(row.get(q, ""))},
                    {"role": "assistant", "content": str(row.get(a, ""))},
                ]
                return {"text": _apply_chat(tokenizer, messages)}

            return dataset.map(format_qa)

    text_col = _pick_column(columns, _SFT_TEXT_COLUMNS)
    label_col = _pick_column(columns, _SFT_LABEL_COLUMNS)
    context_col = _pick_column(columns, _SFT_CONTEXT_COLUMNS)
    if text_col and label_col:
        def format_labeled(row: dict[str, Any]) -> dict[str, str]:
            body = str(row.get(text_col, "") or "").strip()
            label = str(row.get(label_col, "") or "").strip()
            context = str(row.get(context_col, "") or "").strip() if context_col else ""
            label_key = str(label_col).lower()
            if label_key in {"star", "stars", "rating"}:
                user = "What star rating (1-5) does this review deserve"
                if context:
                    user += f" for {context}"
                user += f"?\n\n{body}"
            else:
                user = "Classify the following text"
                if context:
                    user += f" about {context}"
                user += f":\n\n{body}"
            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": label},
            ]
            return {"text": _apply_chat(tokenizer, messages)}

        return dataset.map(format_labeled)

    if text_col:
        return dataset.map(lambda row, col=text_col: {"text": str(row.get(col, "") or "")})

    source = "messages" if "messages" in columns else "conversations" if "conversations" in columns else None
    if source is None:
        raise ValueError(
            "SFT dataset requires a text, messages, conversations, instruction/output, "
            "prompt/completion, question/answer, or text+label (e.g. review/star) column; "
            f"received {sorted(columns)}"
        )

    def format_row(row: dict[str, Any]) -> dict[str, str]:
        messages = row[source]
        if source == "conversations":
            role_map = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
            messages = [
                {
                    "role": role_map.get(str(item.get("from", "user")).lower(), "user"),
                    "content": str(item.get("value", "")),
                }
                for item in messages
            ]
        return {"text": _apply_chat(tokenizer, messages)}

    return dataset.map(format_row)


def _prepare_embedding_dataset(dataset: Any) -> Any:
    columns = set(getattr(dataset, "column_names", []) or [])
    if {"anchor", "positive"}.issubset(columns):
        return dataset
    if {"text1", "text2"}.issubset(columns):
        return dataset.rename_columns({"text1": "anchor", "text2": "positive"})
    if {"query", "pos"}.issubset(columns):
        mapping = {"query": "anchor", "pos": "positive"}
        if "neg" in columns:
            mapping["neg"] = "negative"
        return dataset.rename_columns(mapping)
    if {"sentence1", "sentence2"}.issubset(columns):
        return dataset.rename_columns({"sentence1": "anchor", "sentence2": "positive"})
    raise ValueError(
        "Embedding dataset requires anchor/positive[/negative], text1/text2, query/pos[/neg], "
        f"or sentence1/sentence2 columns; received {sorted(columns)}"
    )


def _export_and_maybe_push(
    model, tokenizer, payload: dict[str, Any], scratch_dir: str, log: LogCarrier, job_uuid: str,
    *, default_format: str = "lora",
) -> dict[str, Any]:
    from src.finetuning_modules.model_exporters import UnslothModelExporter

    export_cfg = dict(payload.get("export") or {})
    export_format = export_cfg.get("format") or payload.get("export_format") or default_format
    local_dir = export_cfg.get("local_dir") or os.path.join(scratch_dir, "export")
    minio_dest = export_cfg.get("minio_destination") or f"{_job_minio_prefix(job_uuid)}/exports"

    dest = UnslothModelExporter.export_model(
        model=model,
        tokenizer=tokenizer,
        export_format=export_format,
        local_output_dir=local_dir,
        minio_destination_link=minio_dest,
    )

    hf_repo = _ensure_hf_repo_on_payload(payload, job_uuid)
    export_cfg = dict(payload.get("export") or {})
    token = _resolve_hf_token(export_cfg)
    hf_push = None
    if hf_repo:
        _notify_hf_push_started(job_uuid, hf_repo, dest, log)
        try:
            pushed = UnslothModelExporter.push_to_hub(
                model=model,
                tokenizer=tokenizer,
                repo_id=hf_repo,
                token=token,
                private=bool(export_cfg.get("private", True)),
            )
            hf_push = {
                "status": "published",
                "repo_id": pushed,
                "url": _hf_repo_url(pushed),
            }
            log.info("hf_push_complete", repo_id=hf_repo)
        except Exception as exc:
            log.error("hf_push_failed", repo_id=hf_repo, error=str(exc))
            hf_push = {
                "status": "failed",
                "repo_id": hf_repo,
                "url": None,
                "error": str(exc),
            }

    return {"export_dest": dest, "hf_push": hf_push}


def _run_eval(
    *,
    job_uuid: str,
    payload: dict[str, Any],
    model: Any,
    tokenizer: Any,
    base_model_id: str,
    max_seq: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    log: LogCarrier,
    dry_run: bool,
    baseline: Optional[dict[str, dict[str, Any]]] = None,
) -> Optional[dict[str, Any]]:
    eval_cfg = payload.get("eval") or {}
    prompts = eval_cfg.get("prompts") or []
    if not prompts:
        return None

    tester = BatchInferenceTester(max_prompts=int(eval_cfg.get("max_prompts") or 1000))

    if dry_run:

        def gen_base(p: str) -> str:
            return f"[base]{p}"

        def gen_ft(p: str) -> str:
            return p

        report = tester.run(
            job_uuid=job_uuid,
            prompts=prompts,
            generate_base=gen_base,
            generate_ft=gen_ft,
            reference_loss_base=lambda p, r: 2.0,
            reference_loss_ft=lambda p, r: 1.0,
        )
        log.info("eval_complete_dry_run", n_prompts=report.get("n_prompts"))
        return report

    from src.finetuning_modules.inference_testers.unsloth_eval import (
        make_generate_fn,
        make_reference_loss_fn,
    )
    from src.finetuning_modules.model_loaders import UnslothModelLoader

    try:
        from unsloth import FastLanguageModel

        enable = FastLanguageModel.for_inference
    except Exception:
        enable = None

    gen_ft = make_generate_fn(model, tokenizer, enable_inference=enable)
    loss_ft = make_reference_loss_fn(model, tokenizer)

    base_model = None
    base_tok = None
    if baseline is not None:
        gen_base = lambda p: baseline.get(p, {}).get("generation", "")
        loss_base = lambda p, r: baseline.get(p, {}).get("loss", {}).get(str(r), 0.0)
    else:
        try:
            base_model, base_tok = UnslothModelLoader.load_model(
                base_model_id,
                max_seq_length=max_seq,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
            )
            gen_base = make_generate_fn(base_model, base_tok, enable_inference=enable)
            loss_base = make_reference_loss_fn(base_model, base_tok)
        except Exception as exc:
            log.warning("eval_skipped_no_baseline", error=str(exc))
            return None

    report = tester.run(
        job_uuid=job_uuid,
        prompts=prompts,
        generate_base=gen_base,
        generate_ft=gen_ft,
        reference_loss_base=loss_base,
        reference_loss_ft=loss_ft,
    )
    log.info("eval_complete", n_prompts=report.get("n_prompts"))
    del base_model, base_tok
    return report


def _stabilize_fullparams_dtypes(model: Any, log: LogCarrier) -> Any:
    """
    Full-parameter Gemma runs often end up with float32 weights and fp16 activations
    (especially with Unsloth FORCE_FLOAT32 + compile disabled). Align every Linear so
    matmul dtypes match.
    """
    import torch
    from torch import nn

    try:
        model = model.float()
    except Exception as exc:
        log.warning("fullparams_model_float_failed", error=str(exc))

    if getattr(model, "_customizer_dtype_hooks", False):
        log.info("fullparams_dtype_stabilized", linear_hooks="already_attached")
        return model

    def _pre_hook(module: nn.Module, args: tuple[Any, ...]):
        if not args:
            return None
        hidden = args[0]
        weight = getattr(module, "weight", None)
        if (
            weight is None
            or not torch.is_tensor(hidden)
            or not torch.is_tensor(weight)
            or hidden.dtype == weight.dtype
        ):
            return None
        return (hidden.to(dtype=weight.dtype),) + tuple(args[1:])

    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.register_forward_pre_hook(_pre_hook)

    if hasattr(model, "config"):
        try:
            model.config.torch_dtype = torch.float32
        except Exception:
            pass

    try:
        setattr(model, "_customizer_dtype_hooks", True)
    except Exception:
        pass

    log.info("fullparams_dtype_stabilized", linear_hooks=True)
    return model


def _capture_eval_baseline(payload: dict[str, Any], model: Any, tokenizer: Any, log: LogCarrier) -> Optional[dict[str, dict[str, Any]]]:
    prompts = (payload.get("eval") or {}).get("prompts") or []
    if not prompts:
        return None
    from src.finetuning_modules.inference_testers.unsloth_eval import make_generate_fn, make_reference_loss_fn

    enable_inference = None
    enable_training = None
    try:
        from unsloth import FastLanguageModel
        enable_inference = FastLanguageModel.for_inference
        enable_training = FastLanguageModel.for_training
    except (ImportError, AttributeError):
        pass
    generate = make_generate_fn(model, tokenizer, enable_inference=enable_inference)
    reference_loss = make_reference_loss_fn(model, tokenizer)
    baseline: dict[str, dict[str, Any]] = {}
    try:
        for item in prompts[:1000]:
            prompt = str(item.get("prompt", ""))
            record = {"generation": generate(prompt), "loss": {}}
            if item.get("reference") is not None:
                ref = str(item["reference"])
                record["loss"][ref] = reference_loss(prompt, ref)
            baseline[prompt] = record
    finally:
        if enable_training is not None:
            enable_training(model)
    log.info("baseline_eval_complete", n_prompts=len(baseline))
    return baseline


def run_sft_style(
    payload: dict[str, Any],
    *,
    scratch_dir: str,
    log: LogCarrier,
    load_in_4bit: bool,
    load_in_8bit: bool,
    use_peft: bool,
) -> dict[str, Any]:
    job_uuid = str(payload.get("job_uuid") or payload.get("jobId"))
    base_model_id = payload.get("base_model_id") or payload.get("baseModelId")
    if not base_model_id:
        raise ValueError("base_model_id is required")

    dry_run = bool(payload.get("dry_run") or payload.get("dryRun") or os.getenv("CUSTOMIZER_DRY_RUN") == "1")
    max_seq = int(payload.get("max_seq_length") or 2048)
    hf_push = _preflight_hf_push(payload, dry_run=dry_run, log=log)

    if hf_push and hf_push["status"] == "failed":
        return {
            "status": "FAILED",
            "job_uuid": job_uuid,
            "export": {"export_dest": None, "hf_push": hf_push},
            "eval": None,
        }

    if dry_run:
        log.info("dry_run_pipeline")
        eval_report = _run_eval(
            job_uuid=job_uuid,
            payload=payload,
            model=None,
            tokenizer=None,
            base_model_id=base_model_id,
            max_seq=max_seq,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            log=log,
            dry_run=True,
        )
        _write_json_best_effort(os.path.join(scratch_dir, "eval_report.json"), eval_report or {}, log)
        return {
            "status": "COMPLETED",
            "job_uuid": job_uuid,
            "export": {
                "export_dest": f"{_job_minio_prefix(job_uuid)}/exports",
                "hf_push": hf_push,
                "dry_run": True,
            },
            "eval": eval_report,
        }

    from src.finetuning_modules.model_loaders import UnslothModelLoader
    from src.finetuning_modules.model_trainers import UnslothModelTrainer

    dataset = _load_dataset(payload)
    log.info("dataset_loaded")

    log.info("model_loading", base_model_id=base_model_id)
    load_kwargs: dict[str, Any] = {}
    if not use_peft:
        # Prefer float32 full-FT. Unsloth may still mix dtypes on Gemma; hooks below fix that.
        import torch

        load_kwargs["dtype"] = torch.float32
        os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
    model, tokenizer = UnslothModelLoader.load_model(
        base_model_id,
        max_seq_length=max_seq,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=not use_peft,
        **load_kwargs,
    )
    log.info("model_loaded", base_model_id=base_model_id)
    tokenizer = _ensure_chat_template(tokenizer, base_model_id)
    if use_peft:
        peft_cfg = payload.get("peft") or {}
        model = UnslothModelLoader.get_peft_model(
            model, **{k: v for k, v in peft_cfg.items() if v is not None}
        )
    else:
        model = _stabilize_fullparams_dtypes(model, log)

    baseline = _capture_eval_baseline(payload, model, tokenizer, log)
    if not use_peft:
        # Baseline inference mode can leave modules in a mixed state; re-assert float32.
        model = _stabilize_fullparams_dtypes(model, log)
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_training(model)
        except Exception:
            model.train()

    dataset = _prepare_sft_dataset(dataset, tokenizer)
    train_cfg = dict(payload.get("train") or {})
    if str(payload.get("objective") or "").lower() == "cpt":
        # Raw text is causal continued pretraining: do not fabricate an
        # instruction target or mask the prompt portion.
        train_cfg.setdefault("assistant_only_loss", False)
        train_cfg.setdefault("completion_only_loss", False)
    if "batch_size" not in train_cfg:
        payload_bs = payload.get("batch_size") or payload.get("batchSize")
        if payload_bs is not None:
            train_cfg["batch_size"] = int(payload_bs)
        else:
            # Full-FT float32 needs a tiny micro-batch on consumer GPUs.
            train_cfg["batch_size"] = 1 if not use_peft else 2
    if not use_peft:
        train_cfg.setdefault("learning_rate", 2e-5)
        train_cfg.setdefault("optimizer", "adamw_torch")
        train_cfg.setdefault("force_float32", True)
        train_cfg.setdefault("grad_accum_steps", 4)
    output_dir = train_cfg.pop("output_dir", os.path.join(scratch_dir, "checkpoints"))
    UnslothModelTrainer.train(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=output_dir,
        max_seq_length=max_seq,
        on_progress=make_train_progress_callback(log),
        resume_from_checkpoint=payload.get("resume_from_checkpoint"),
        **train_cfg,
    )
    log.info("train_complete")

    export_info = _export_and_maybe_push(
        model, tokenizer, payload, scratch_dir, log, job_uuid,
        default_format="lora" if use_peft else "merged_16bit",
    )

    eval_report = _run_eval(
        job_uuid=job_uuid,
        payload=payload,
        model=model,
        tokenizer=tokenizer,
        base_model_id=base_model_id,
        max_seq=max_seq,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        log=log,
        dry_run=False,
        baseline=baseline,
    )
    if eval_report is not None:
        _write_json_best_effort(os.path.join(scratch_dir, "eval_report.json"), eval_report, log)

    return {
        "status": "COMPLETED",
        "job_uuid": job_uuid,
        "export": export_info,
        "eval": eval_report,
    }


def run_fullparams(payload: dict[str, Any], scratch_dir: str, log: LogCarrier) -> dict[str, Any]:
    spec = get_recipe_spec("fullparams")
    return run_sft_style(
        payload,
        scratch_dir=scratch_dir,
        log=log,
        load_in_4bit=spec.load_in_4bit,
        load_in_8bit=spec.load_in_8bit,
        use_peft=spec.use_peft,
    )


def run_lora(payload: dict[str, Any], scratch_dir: str, log: LogCarrier) -> dict[str, Any]:
    spec = get_recipe_spec("lora")
    return run_sft_style(
        payload,
        scratch_dir=scratch_dir,
        log=log,
        load_in_4bit=spec.load_in_4bit,
        load_in_8bit=spec.load_in_8bit,
        use_peft=spec.use_peft,
    )


def run_qlora(payload: dict[str, Any], scratch_dir: str, log: LogCarrier, bit_width: int = 4) -> dict[str, Any]:
    recipe = "qlora_4bit" if int(bit_width) == 4 else "qlora_8bit"
    spec = get_recipe_spec(recipe)
    payload = {**payload, "recipe": recipe}
    return run_sft_style(
        payload,
        scratch_dir=scratch_dir,
        log=log,
        load_in_4bit=spec.load_in_4bit,
        load_in_8bit=spec.load_in_8bit,
        use_peft=spec.use_peft,
    )


def run_embedding(payload: dict[str, Any], scratch_dir: str, log: LogCarrier) -> dict[str, Any]:
    job_uuid = str(payload.get("job_uuid") or payload.get("jobId"))
    base_model_id = payload.get("base_model_id") or payload.get("baseModelId")
    if not base_model_id:
        raise ValueError("base_model_id is required")
    dry_run = bool(payload.get("dry_run") or payload.get("dryRun") or os.getenv("CUSTOMIZER_DRY_RUN") == "1")
    hf_push = _preflight_hf_push(payload, dry_run=dry_run, log=log)
    if hf_push and hf_push["status"] == "failed":
        return {
            "status": "FAILED",
            "job_uuid": job_uuid,
            "export": {"export_dest": None, "hf_push": hf_push},
            "eval": None,
        }
    if dry_run:
        return {
            "status": "COMPLETED",
            "job_uuid": job_uuid,
            "export": {
                "export_dest": f"{_job_minio_prefix(job_uuid)}/exports",
                "hf_push": hf_push,
                "dry_run": True,
            },
            "eval": None,
        }

    from unsloth import FastSentenceTransformer
    from src.finetuning_modules.model_trainers import UnslothEmbeddingTrainer

    max_seq = int(payload.get("max_seq_length") or payload.get("maxSeqLength") or 512)
    model = FastSentenceTransformer.from_pretrained(
        model_name=base_model_id,
        max_seq_length=max_seq,
        full_finetuning=False,
    )
    peft_cfg = dict(payload.get("peft") or {})
    if "target_modules" not in peft_cfg:
        lowered = base_model_id.lower()
        peft_cfg["target_modules"] = (
            ["value", "key", "dense", "query"]
            if any(name in lowered for name in ("minilm", "modernbert", "mpnet", "bge"))
            else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
    peft_cfg.setdefault("r", 32)
    peft_cfg.setdefault("lora_alpha", peft_cfg["r"])
    peft_cfg.setdefault("lora_dropout", 0)
    peft_cfg.setdefault("bias", "none")
    peft_cfg.setdefault("use_gradient_checkpointing", False)
    peft_cfg.setdefault("random_state", 3407)
    peft_cfg.setdefault("use_rslora", False)
    peft_cfg.setdefault("loftq_config", None)
    peft_cfg.setdefault("task_type", "FEATURE_EXTRACTION")
    model = FastSentenceTransformer.get_peft_model(model, **peft_cfg)

    dataset = _prepare_embedding_dataset(_load_dataset(payload))
    train_cfg = dict(payload.get("train") or {})
    train_cfg.setdefault("batch_size", int(payload.get("batch_size") or payload.get("batchSize") or 32))
    output_dir = train_cfg.pop("output_dir", os.path.join(scratch_dir, "checkpoints"))
    UnslothEmbeddingTrainer.train(model=model, dataset=dataset, output_dir=output_dir, **train_cfg)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Embedding model did not expose its tokenizer")
    export_info = _export_and_maybe_push(
        model, tokenizer, payload, scratch_dir, log, job_uuid, default_format="lora"
    )
    return {"status": "COMPLETED", "job_uuid": job_uuid, "export": export_info, "eval": None}


def run_recipe(
    payload: dict[str, Any],
    *,
    scratch_dir: str,
    log: Optional[LogCarrier] = None,
) -> dict[str, Any]:
    recipe = (payload.get("recipe") or "").strip().lower()
    objective = str((payload.get("plan") or {}).get("objective") or payload.get("objective") or "").strip().lower()
    job_uuid = str(payload.get("job_uuid") or payload.get("jobId") or "unknown")
    carrier = log or LogCarrier(job_uuid)

    if objective in {"pretrain", "dpo", "kto", "reward_model", "ppo", "grpo"}:
        from src.finetuning_flows.objectives import RUNNERS
        return RUNNERS[objective](payload, scratch_dir, carrier)

    dispatch = {
        "fullparams": run_fullparams,
        "lora": run_lora,
        "qlora_4bit": lambda p, s, l: run_qlora(p, s, l, bit_width=4),
        "qlora_8bit": lambda p, s, l: run_qlora(p, s, l, bit_width=8),
        "embedding": run_embedding,
    }
    # The current certified backend supports causal SFT through the existing
    # recipes. Objective names are accepted at the boundary so the manager can
    # persist a real plan without inventing a second SFT implementation.
    if objective in {"sft", "cpt"} and recipe in {"", "auto"}:
        recipe = "qlora_4bit"
    if recipe not in dispatch:
        raise ValueError(f"Unsupported recipe '{recipe}'")
    return dispatch[recipe](payload, scratch_dir, carrier)
