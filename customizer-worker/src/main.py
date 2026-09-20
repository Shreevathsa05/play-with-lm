"""Customizer worker entrypoint for dataset preparation and training."""

from __future__ import annotations

import logging
import os


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    mode = os.getenv("CUSTOMIZER_MODE", "consumer").lower()
    if mode == "demo":
        from src.finetuning_modules.dataset_loaders import ExternalDatasetLoader

        dataset = ExternalDatasetLoader.load(path="imdb", split="train[:1%]")
        print(f"Loaded {len(dataset)} examples from IMDB.")
        return

    # Dataset work has its own RabbitMQ connection/thread so CPU auditing does
    # not block the training consumer's GPU job loop.
    from src.dataset_processor import start_dataset_consumer_thread

    start_dataset_consumer_thread()
    from src.consumer import start_consumer
    start_consumer()


if __name__ == "__main__":
    main()
