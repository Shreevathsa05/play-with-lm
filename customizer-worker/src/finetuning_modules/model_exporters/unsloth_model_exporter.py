import os
import tempfile
import shutil
from typing import Any, Optional

from src.storage.minio_data_manipulation import MinioCRUD
from src.utils.minio_links import parse_minio_link


class UnslothModelExporter:
    """Save / export trained models locally, to MinIO, or to Hugging Face Hub."""

    @staticmethod
    def export_model(
        model: Any,
        tokenizer: Any,
        export_format: str = "lora",
        local_output_dir: Optional[str] = None,
        minio_destination_link: Optional[str] = None,
        quantization_method: str = "q4_k_m",
        **kwargs: Any,
    ) -> str:
        export_format = export_format.lower()
        valid_formats = ["lora", "merged_16bit", "merged_4bit", "gguf", "gguf_quantized"]
        if export_format not in valid_formats:
            raise ValueError(
                f"Unsupported export format '{export_format}'. Supported: {valid_formats}"
            )

        cleanup_local = False
        if local_output_dir is None:
            if minio_destination_link is None:
                raise ValueError("Must specify either local_output_dir or minio_destination_link.")
            local_output_dir = tempfile.mkdtemp(prefix=f"unsloth_export_{export_format}_")
            cleanup_local = True

        print(f"Exporting model to format '{export_format}' in '{local_output_dir}'...")
        os.makedirs(local_output_dir, exist_ok=True)

        try:
            if export_format == "lora":
                model.save_pretrained(local_output_dir, **kwargs)
                tokenizer.save_pretrained(local_output_dir, **kwargs)
            elif export_format in ["merged_16bit", "merged_4bit"]:
                model.save_pretrained_merged(
                    local_output_dir,
                    tokenizer,
                    save_method=export_format,
                    **kwargs,
                )
            elif export_format == "gguf":
                model.save_pretrained_merged(
                    local_output_dir,
                    tokenizer,
                    save_method="gguf",
                    **kwargs,
                )
            elif export_format == "gguf_quantized":
                model.save_pretrained_merged(
                    local_output_dir,
                    tokenizer,
                    save_method="gguf_quantized",
                    quantization_method=quantization_method,
                    **kwargs,
                )

            print(f"Model successfully saved/exported locally to {local_output_dir}!")

            if minio_destination_link:
                bucket, prefix = parse_minio_link(minio_destination_link)
                print(f"Uploading exported files to MinIO: '{minio_destination_link}'...")

                uploaded_any = False
                for root, _, files in os.walk(local_output_dir):
                    for file in files:
                        local_file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(local_file_path, local_output_dir)
                        if prefix:
                            object_name = f"{prefix.rstrip('/')}/{rel_path.replace(os.sep, '/')}"
                        else:
                            object_name = rel_path.replace(os.sep, "/")

                        MinioCRUD.upload_file(bucket, object_name, local_file_path)
                        uploaded_any = True

                if not uploaded_any:
                    raise FileNotFoundError("No exported files found to upload to MinIO.")
                print(f"Successfully uploaded model to MinIO under: {minio_destination_link}")
                return minio_destination_link

            return local_output_dir

        finally:
            if cleanup_local and os.path.exists(local_output_dir):
                try:
                    shutil.rmtree(local_output_dir)
                except Exception as e:
                    print(f"Warning: Failed to remove temporary export directory: {e}")

    @staticmethod
    def push_to_hub(
        model: Any,
        tokenizer: Any,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Push model + tokenizer to the user's Hugging Face repo.

        Token must never be logged by callers (LogCarrier redacts token fields).
        """
        if not repo_id or not str(repo_id).strip():
            raise ValueError("repo_id is required for Hugging Face push")

        push_kwargs = dict(kwargs)
        if token:
            push_kwargs["token"] = token
        push_kwargs.setdefault("private", private)

        if hasattr(model, "push_to_hub"):
            model.push_to_hub(repo_id, **push_kwargs)
        else:
            raise AttributeError("model does not support push_to_hub")

        tok_kwargs = {k: v for k, v in push_kwargs.items() if k != "private"}
        if hasattr(tokenizer, "push_to_hub"):
            tokenizer.push_to_hub(repo_id, **tok_kwargs)

        return repo_id
