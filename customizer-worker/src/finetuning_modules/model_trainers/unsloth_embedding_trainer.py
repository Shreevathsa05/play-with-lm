"""Unsloth sentence-embedding training built on Sentence Transformers."""

from __future__ import annotations

from typing import Any


class UnslothEmbeddingTrainer:
    """Train pair or triplet embedding datasets with an appropriate contrastive loss."""

    @staticmethod
    def train(
        *,
        model: Any,
        dataset: Any,
        output_dir: str,
        batch_size: int = 32,
        grad_accum_steps: int = 1,
        epochs: float = 1.0,
        learning_rate: float = 2e-4,
        logging_steps: int = 10,
        seed: int = 3407,
        **kwargs: Any,
    ) -> Any:
        from sentence_transformers import (
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from sentence_transformers.training_args import BatchSamplers
        from unsloth import is_bfloat16_supported

        columns = set(dataset.column_names)
        if {"anchor", "positive", "negative"}.issubset(columns):
            train_dataset = dataset.select_columns(["anchor", "positive", "negative"])
            loss = losses.TripletLoss(model)
            batch_sampler = BatchSamplers.BATCH_SAMPLER
        elif {"anchor", "positive"}.issubset(columns):
            train_dataset = dataset.select_columns(["anchor", "positive"])
            loss = losses.MultipleNegativesRankingLoss(model)
            batch_sampler = BatchSamplers.NO_DUPLICATES
        else:
            raise ValueError(
                "Embedding dataset requires anchor/positive[/negative] columns after normalization"
            )

        use_bf16 = is_bfloat16_supported()
        args = SentenceTransformerTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            learning_rate=learning_rate,
            fp16=not use_bf16,
            bf16=use_bf16,
            logging_steps=logging_steps,
            warmup_ratio=0.03,
            report_to="none",
            lr_scheduler_type="linear",
            batch_sampler=batch_sampler,
            seed=seed,
            **kwargs.get("training_args_kwargs", {}),
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=loss,
            args=args,
            **kwargs.get("trainer_kwargs", {}),
        )
        return trainer.train()


__all__ = ["UnslothEmbeddingTrainer"]
