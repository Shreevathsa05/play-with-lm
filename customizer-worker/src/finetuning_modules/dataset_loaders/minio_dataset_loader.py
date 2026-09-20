import os
import tempfile
from typing import Optional, Union, Dict, Any

from datasets import load_dataset, Dataset

from src.storage.minio_data_manipulation import MinioCRUD
from src.utils.minio_links import parse_minio_link


class MinioDatasetLoader:
    """Load datasets directly from a MinIO bucket."""

    @staticmethod
    def load(
        minio_link: str,
        split: Optional[str] = "train",
        **kwargs: Any,
    ) -> Union[Dataset, Dict[str, Dataset]]:
        bucket, obj_name = parse_minio_link(minio_link)

        _, ext = os.path.splitext(obj_name)
        ext = ext.lower()

        if ext == ".txt":
            fmt = "text"
        elif ext in [".json", ".jsonl"]:
            fmt = "json"
        elif ext == ".csv":
            fmt = "csv"
        else:
            raise ValueError(f"Unsupported file extension for MinIO dataset loading: {ext}")

        fd, local_input = tempfile.mkstemp(suffix=ext)
        os.close(fd)

        try:
            print(f"Downloading from MinIO: {minio_link}")
            MinioCRUD.download_file(bucket, obj_name, local_input)

            print(f"Loading dataset with format '{fmt}'...")
            dataset = load_dataset(
                path=fmt,
                data_files=local_input,
                split=split,
                **kwargs,
            )
            print("Dataset successfully loaded from MinIO!")
            return dataset
        finally:
            if os.path.exists(local_input):
                try:
                    os.remove(local_input)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary file {local_input}: {e}")
