"""Official Trainer-backed objectives beyond SFT/CPT.

Each function imports its trainer lazily so the capability probe can report a
missing runtime without preventing dataset preparation or worker startup.
"""

from __future__ import annotations

import inspect
import os
from typing import Any


def _config(config_type: Any, output_dir: str, train: dict[str, Any]) -> Any:
    values = {
        "output_dir": output_dir, "per_device_train_batch_size": int(train.get("batch_size", 1)),
        "gradient_accumulation_steps": int(train.get("grad_accum_steps", 1)),
        "learning_rate": float(train.get("learning_rate", 1e-5)),
        "num_train_epochs": float(train.get("epochs", 1)), "max_steps": int(train.get("max_steps", -1)),
        "logging_steps": int(train.get("logging_steps", 1)), "report_to": "none",
        "save_strategy": train.get("save_strategy", "steps"), "save_steps": int(train.get("save_steps", 50)),
        "seed": int(train.get("seed", 3407)),
    }
    accepted = inspect.signature(config_type).parameters
    values.update({k: v for k, v in train.items() if k in accepted})
    return config_type(**{k: v for k, v in values.items() if k in accepted})


def _dataset(payload: dict[str, Any]) -> Any:
    from src.finetuning_flows.orchestrator import _load_dataset
    return _load_dataset(payload)


def _policy(payload: dict[str, Any]) -> tuple[Any, Any]:
    from src.finetuning_modules.model_loaders import UnslothModelLoader
    from src.finetuning_modules.compatibility.recipes import get_recipe_spec
    spec = get_recipe_spec(str(payload.get("recipe") or "qlora_4bit"))
    model, tokenizer = UnslothModelLoader.load_model(
        str(payload["base_model_id"]), max_seq_length=int(payload.get("max_seq_length", 2048)),
        load_in_4bit=spec.load_in_4bit, load_in_8bit=spec.load_in_8bit,
    )
    if spec.use_peft:
        model = UnslothModelLoader.get_peft_model(model, **dict(payload.get("peft") or {}))
    return model, tokenizer


def _finish(model: Any, tokenizer: Any, payload: dict[str, Any], scratch_dir: str, log: Any) -> dict[str, Any]:
    from src.finetuning_flows.orchestrator import _export_and_maybe_push
    job_uuid = str(payload["job_uuid"])
    return {"status": "COMPLETED", "job_uuid": job_uuid,
            "export": _export_and_maybe_push(model, tokenizer, payload, scratch_dir, log, job_uuid), "eval": None}


def run_dpo(payload: dict[str, Any], scratch_dir: str, log: Any) -> dict[str, Any]:
    from trl import DPOConfig, DPOTrainer
    model, tokenizer = _policy(payload)
    trainer = DPOTrainer(model=model, processing_class=tokenizer, train_dataset=_dataset(payload),
                         args=_config(DPOConfig, os.path.join(scratch_dir, "checkpoints"), dict(payload.get("train") or {})))
    trainer.train(resume_from_checkpoint=payload.get("resume_from_checkpoint"))
    return _finish(model, tokenizer, payload, scratch_dir, log)


def run_kto(payload: dict[str, Any], scratch_dir: str, log: Any) -> dict[str, Any]:
    from trl import KTOConfig, KTOTrainer
    model, tokenizer = _policy(payload)
    trainer = KTOTrainer(model=model, processing_class=tokenizer, train_dataset=_dataset(payload),
                         args=_config(KTOConfig, os.path.join(scratch_dir, "checkpoints"), dict(payload.get("train") or {})))
    trainer.train(resume_from_checkpoint=payload.get("resume_from_checkpoint"))
    return _finish(model, tokenizer, payload, scratch_dir, log)


def run_reward_model(payload: dict[str, Any], scratch_dir: str, log: Any) -> dict[str, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from trl import RewardConfig, RewardTrainer
    model_id = str(payload["base_model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1)
    trainer = RewardTrainer(model=model, processing_class=tokenizer, train_dataset=_dataset(payload),
                            args=_config(RewardConfig, os.path.join(scratch_dir, "checkpoints"), dict(payload.get("train") or {})))
    trainer.train(resume_from_checkpoint=payload.get("resume_from_checkpoint"))
    return _finish(model, tokenizer, payload, scratch_dir, log)


def run_pretrain(payload: dict[str, Any], scratch_dir: str, log: Any) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
    train = dict(payload.get("train") or {})
    config_id = str(payload.get("scratch_config_model_id") or payload["base_model_id"])
    tokenizer_id = str(payload.get("tokenizer_id") or config_id)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(config_id))
    source = _dataset(payload)
    data = source.map(lambda x: tokenizer(x["text"], truncation=True, max_length=int(payload.get("max_seq_length", 512))),
                      batched=True, remove_columns=source.column_names)
    output = os.path.join(scratch_dir, "checkpoints")
    args = TrainingArguments(output_dir=output, per_device_train_batch_size=int(train.get("batch_size", 1)),
                             gradient_accumulation_steps=int(train.get("grad_accum_steps", 1)), num_train_epochs=float(train.get("epochs", 1)),
                             max_steps=int(train.get("max_steps", -1)), save_steps=int(train.get("save_steps", 50)), report_to="none")
    Trainer(model=model, args=args, train_dataset=data, processing_class=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)).train(
                resume_from_checkpoint=payload.get("resume_from_checkpoint"))
    return _finish(model, tokenizer, payload, scratch_dir, log)


def run_grpo(payload: dict[str, Any], scratch_dir: str, log: Any) -> dict[str, Any]:
    from trl import GRPOConfig, GRPOTrainer
    model, tokenizer = _policy(payload)
    def exact_match(completions: list[Any], answer: list[Any], **_: Any) -> list[float]:
        return [float(str(c).strip() == str(a).strip()) for c, a in zip(completions, answer)]
    trainer = GRPOTrainer(model=model, processing_class=tokenizer, train_dataset=_dataset(payload), reward_funcs=exact_match,
                          args=_config(GRPOConfig, os.path.join(scratch_dir, "checkpoints"), dict(payload.get("train") or {})))
    trainer.train(resume_from_checkpoint=payload.get("resume_from_checkpoint"))
    return _finish(model, tokenizer, payload, scratch_dir, log)


def run_ppo(payload: dict[str, Any], scratch_dir: str, log: Any) -> dict[str, Any]:
    """Run the installed TRL PPO implementation with explicit policy/reward/value inputs."""
    if not payload.get("reward_model_id"):
        raise ValueError("PPO requires reward_model_id from a trained reward-model job")
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    from trl import PPOConfig, PPOTrainer
    model_id, reward_id = str(payload["base_model_id"]), str(payload["reward_model_id"])
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(model_id)
    ref_policy = AutoModelForCausalLM.from_pretrained(model_id)
    reward = AutoModelForSequenceClassification.from_pretrained(reward_id, num_labels=1)
    value = AutoModelForSequenceClassification.from_pretrained(reward_id, num_labels=1)
    trainer = PPOTrainer(args=_config(PPOConfig, os.path.join(scratch_dir, "checkpoints"), dict(payload.get("train") or {})),
                         processing_class=tokenizer, model=policy, ref_model=ref_policy, reward_model=reward,
                         value_model=value, train_dataset=_dataset(payload))
    trainer.train(resume_from_checkpoint=payload.get("resume_from_checkpoint"))
    return _finish(policy, tokenizer, payload, scratch_dir, log)


RUNNERS = {"pretrain": run_pretrain, "dpo": run_dpo, "kto": run_kto, "reward_model": run_reward_model, "ppo": run_ppo, "grpo": run_grpo}
