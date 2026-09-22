from typing import Any, Callable, Optional
import os
import sys

from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from unsloth import is_bfloat16_supported


def _default_dataset_num_proc() -> Optional[int]:
    # datasets>=3/4 treats num_proc>=1 as "use a process Pool", including num_proc=1.
    # Only None disables multiprocessing. On Windows (spawn), those pool workers often
    # die while pickling CUDA/tokenizer state — the exact "subprocess abruptly died" error.
    if sys.platform.startswith("win"):
        return None
    return 2


class LossTargetCallback(TrainerCallback):
    """Stop when loss hits an absolute floor or has fallen by a relative fraction."""

    def __init__(
        self,
        *,
        target_loss: Optional[float] = None,
        min_loss_drop_ratio: Optional[float] = None,
        on_progress: Optional[Callable[..., None]] = None,
    ):
        self.target_loss = target_loss
        self.min_loss_drop_ratio = min_loss_drop_ratio
        self.on_progress = on_progress
        self.initial_loss: Optional[float] = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not logs or "loss" not in logs:
            return
        loss = float(logs["loss"])
        if self.initial_loss is None or self.initial_loss <= 0:
            self.initial_loss = loss

        drop_ratio = 0.0
        if self.initial_loss and self.initial_loss > 0:
            drop_ratio = (self.initial_loss - loss) / self.initial_loss

        hit_target = self.target_loss is not None and loss <= float(self.target_loss)
        hit_drop = (
            self.min_loss_drop_ratio is not None
            and drop_ratio >= float(self.min_loss_drop_ratio)
        )
        if not (hit_target or hit_drop):
            return

        reason = "target_loss" if hit_target else "loss_drop"
        print(
            f"Stopping early ({reason}): loss={loss:.4f} "
            f"initial={self.initial_loss:.4f} drop={drop_ratio:.1%}"
        )
        if self.on_progress is not None:
            try:
                self.on_progress(
                    step=state.global_step,
                    total=state.max_steps if state.max_steps and state.max_steps > 0 else None,
                    loss=loss,
                    stop_reason=reason,
                    initial_loss=self.initial_loss,
                    drop_ratio=round(drop_ratio, 4),
                )
            except Exception:
                pass
        control.should_training_stop = True


class UnslothModelTrainer:
    """
    A utility class to train models using Unsloth and TRL's SFTTrainer.
    """

    @staticmethod
    def _build_progress_callback(on_progress: Optional[Callable[..., None]]) -> Optional[TrainerCallback]:
        if on_progress is None:
            return None

        class _ProgressCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if not logs:
                    return
                fields: dict[str, Any] = {"step": state.global_step}
                if state.max_steps and state.max_steps > 0:
                    fields["total"] = state.max_steps
                elif state.num_train_epochs:
                    fields["epoch"] = round(state.epoch or 0, 3)
                if "loss" in logs:
                    fields["loss"] = logs["loss"]
                if "learning_rate" in logs:
                    fields["learning_rate"] = logs["learning_rate"]
                on_progress(**fields)

        return _ProgressCallback()

    @staticmethod
    def train(
        model: Any,
        tokenizer: Any,
        dataset: Any,
        output_dir: str = "outputs",
        dataset_text_field: str = "text",
        max_seq_length: int = 2048,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
        grad_accum_steps: int = 4,
        epochs: float = 1.0,
        max_steps: int = -1,
        logging_steps: int = 1,
        seed: int = 3407,
        dataset_num_proc: Optional[int] = None,
        packing: bool = False,
        optimizer: str = "adamw_8bit",
        force_float32: bool = False,
        warmup_ratio: float = 0.03,
        target_loss: Optional[float] = None,
        min_loss_drop_ratio: Optional[float] = None,
        on_progress: Optional[Callable[..., None]] = None,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        print(f"Initializing trainer for model output: '{output_dir}'...")

        is_bf16 = is_bfloat16_supported()
        use_bf16 = bool(is_bf16) and not force_float32
        use_fp16 = (not is_bf16) and not force_float32
        print(f"Bfloat16 supported: {is_bf16}; force_float32={force_float32}; fp16={use_fp16} bf16={use_bf16}")
        if force_float32:
            os.environ["ACCELERATE_MIXED_PRECISION"] = "no"
            os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
            try:
                model = model.float()
            except Exception as exc:
                print(f"Warning: could not cast model to float32 before train: {exc}")
        if dataset_num_proc is None and "dataset_num_proc" not in (kwargs.get("training_args_kwargs") or {}):
            dataset_num_proc = _default_dataset_num_proc()
        training_args_kwargs = dict(kwargs.get("training_args_kwargs") or {})
        if "dataset_num_proc" in training_args_kwargs:
            dataset_num_proc = training_args_kwargs.pop("dataset_num_proc")
        print(f"dataset_num_proc: {dataset_num_proc!r} (None disables map multiprocessing)")
        if target_loss is not None or min_loss_drop_ratio is not None:
            print(
                f"Loss early-stop enabled: target_loss={target_loss!r} "
                f"min_loss_drop_ratio={min_loss_drop_ratio!r} "
                f"(ceiling max_steps={max_steps}, epochs={epochs})"
            )

        training_args = SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            warmup_ratio=float(warmup_ratio),
            num_train_epochs=epochs if max_steps <= 0 else 1.0,
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=logging_steps,
            optim=optimizer,
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=output_dir,
            save_strategy=training_args_kwargs.pop("save_strategy", "steps"),
            save_steps=int(training_args_kwargs.pop("save_steps", 50)),
            report_to="none",
            max_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            dataset_text_field=dataset_text_field,
            **training_args_kwargs,
        )

        trainer_kwargs = dict(kwargs.get("trainer_kwargs", {}))
        callbacks = list(trainer_kwargs.get("callbacks") or [])
        progress_callback = UnslothModelTrainer._build_progress_callback(on_progress)
        if progress_callback is not None:
            callbacks.append(progress_callback)
        if target_loss is not None or min_loss_drop_ratio is not None:
            callbacks.append(
                LossTargetCallback(
                    target_loss=target_loss,
                    min_loss_drop_ratio=min_loss_drop_ratio,
                    on_progress=on_progress,
                )
            )
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=training_args,
            **trainer_kwargs,
        )

        print("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        print("Training completed successfully!")

        return train_result
