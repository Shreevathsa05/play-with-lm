import os
import tempfile
from typing import Optional, Tuple, Any

from unsloth import FastLanguageModel

from src.storage.minio_connection import minio_client
from src.storage.minio_data_manipulation import MinioCRUD
from src.utils.minio_links import parse_minio_link


class UnslothModelLoader:
    """Load models via Unsloth from HF Hub, local path, or MinIO."""

    @staticmethod
    def load_model(
        model_name_or_link: str,
        max_seq_length: int = 2048,
        dtype: Optional[Any] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device_map: str = "auto",
        full_finetuning: bool = False,
        **kwargs: Any,
    ) -> Tuple[Any, Any]:
        if load_in_4bit and load_in_8bit:
            raise ValueError("load_in_4bit and load_in_8bit are mutually exclusive")

        print(f"Loading model from: '{model_name_or_link}'...")

        if model_name_or_link.startswith("minio://") or model_name_or_link.startswith("s3://"):
            bucket, prefix = parse_minio_link(model_name_or_link)
            temp_dir = tempfile.mkdtemp(prefix="unsloth_model_")
            print(f"Detected MinIO link. Downloading model files to: {temp_dir}")

            try:
                objects = minio_client.list_objects(bucket, prefix=prefix, recursive=True)
                downloaded_any = False
                for obj in objects:
                    object_name = obj.object_name
                    if object_name.endswith("/"):
                        continue

                    if prefix and object_name.startswith(prefix):
                        rel_path = object_name[len(prefix) :].lstrip("/")
                    else:
                        rel_path = os.path.basename(object_name)

                    local_file_path = os.path.join(temp_dir, rel_path)
                    os.makedirs(os.path.dirname(local_file_path) or temp_dir, exist_ok=True)
                    MinioCRUD.download_file(bucket, object_name, local_file_path)
                    downloaded_any = True

                if not downloaded_any:
                    raise FileNotFoundError(
                        f"No model files found in MinIO under link: {model_name_or_link}"
                    )

                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=temp_dir,
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,
                    full_finetuning=full_finetuning,
                    **kwargs,
                )
                print("Model and tokenizer successfully loaded from MinIO!")
                return model, tokenizer
            except Exception as e:
                print(f"Error downloading/loading model from MinIO: {e}")
                raise

        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name_or_link,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
                full_finetuning=full_finetuning,
                **kwargs,
            )
            print("Model and tokenizer successfully loaded!")
            return model, tokenizer
        except Exception as e:
            print(f"Error loading model '{model_name_or_link}': {e}")
            raise

    @staticmethod
    def get_peft_model(
        model: Any,
        r: int = 16,
        target_modules: Optional[Any] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Any = "unsloth",
        random_state: int = 3407,
        use_rslora: bool = False,
        loftq_config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        if target_modules is None:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

        print(f"Configuring PEFT model (LoRA r={r}, alpha={lora_alpha})...")
        try:
            peft_model = FastLanguageModel.get_peft_model(
                model=model,
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=random_state,
                use_rslora=use_rslora,
                loftq_config=loftq_config,
                **kwargs,
            )
            print("PEFT model successfully configured!")
            return peft_model
        except Exception as e:
            print(f"Error configuring PEFT model: {e}")
            raise
