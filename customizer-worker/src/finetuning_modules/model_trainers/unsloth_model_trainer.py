from typing import Any, Callable, Optional
import os
import sys

from trl import SFTConfig, SFTTrainer
from transformers import TrainerCallback
from unsloth import is_bfloat16_supported


def _default_dataset_num_proc() -> Optional[int]:
    # datasets>=3/4 treats num_proc>=1 as "use a process Pool", including num_proc=1.
    # Only None disables multiprocessing. On Windows (spawn), those pool workers often
    # die while pickling CUDA/tokenizer state — the exact "subprocess abruptly died" error.
    if sys.platform.startswith("win"):
        return None
    return 2


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
        on_progress: Optional[Callable[..., None]] = None,
        resume_from_checkpoint: Optional[str] = None,
        **kwargs: Any
    ) -> Any:
        """
        Trains the given model on the provided dataset.
        
        Args:
            model: PEFT-wrapped model to train.
            tokenizer: Tokenizer corresponding to the model.
            dataset: HuggingFace Dataset for training.
            output_dir (str): Local directory to write checkpoints.
            dataset_text_field (str): The column in the dataset containing the text.
            max_seq_length (int): Maximum sequence length.
            learning_rate (float): Initial learning rate for AdamW.
            batch_size (int): Training batch size per device.
            grad_accum_steps (int): Gradient accumulation steps.
            epochs (float): Number of training epochs.
            max_steps (int): Maximum training steps. If positive, overrides epochs.
            logging_steps (int): Logging frequency.
            seed (int): Random seed for training.
            dataset_num_proc (int): Number of workers for dataset tokenization.
                Use None to disable multiprocessing (required on Windows with datasets 4.x).
            packing (bool): Whether to pack multiple short sequences into a single block.
            force_float32 (bool): Disable fp16/bf16 autocast. Needed for Gemma full-FT when
                Unsloth patches expect matching activation/weight dtypes.
            **kwargs: Additional arguments for TrainingArguments or SFTTrainer.
            
        Returns:
            Any: The training results or trainer output.
        """
        print(f"Initializing trainer for model output: '{output_dir}'...")
        
        # Auto-detect float type support
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
        
        # TRL 0.24+ keeps SFT-specific preprocessing options in SFTConfig.
        training_args = SFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            warmup_ratio=0.03,
            num_train_epochs=epochs if max_steps <= 0 else 1.0, # SFTTrainer uses num_train_epochs only if max_steps is negative
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
            report_to="none", # Disable reporting (wandb/tensorboard) to avoid worker environment crashes
            max_length=max_seq_length,
            dataset_num_proc=dataset_num_proc,
            packing=packing,
            dataset_text_field=dataset_text_field,
            **training_args_kwargs,
        )
        
        # Configure SFTTrainer
        trainer_kwargs = dict(kwargs.get("trainer_kwargs", {}))
        progress_callback = UnslothModelTrainer._build_progress_callback(on_progress)
        if progress_callback is not None:
            callbacks = list(trainer_kwargs.get("callbacks") or [])
            callbacks.append(progress_callback)
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
