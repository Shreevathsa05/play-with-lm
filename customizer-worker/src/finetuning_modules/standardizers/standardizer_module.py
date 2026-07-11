import os
import tempfile
from datasets import load_dataset

from unsloth.chat_templates import standardize_data_formats

from src.storage.minio_data_manipulation import MinioCRUD

def parse_minio_link(link: str):
    """
    Parses a minio link into bucket_name and object_name.
    Expected formats: 
    - minio://bucket_name/object_name
    - s3://bucket_name/object_name
    - bucket_name/object_name
    """
    if link.startswith("minio://"):
        link = link[8:]
    elif link.startswith("s3://"):
        link = link[5:]
    
    parts = link.split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "default-bucket", link

class DataStandardizationModule:
    @staticmethod
    def process(minio_link: str) -> tuple:
        """
        Process the dataset and standardize it dynamically based on its extension.
        Args:
            minio_link (str): Input Minio link (e.g. minio://bucket/object.json)
            
        Returns:
            tuple: (standardized minio link, local path to standardized file)
        """
        bucket, obj_name = parse_minio_link(minio_link)
        
        # Download from MinIO
        _, ext = os.path.splitext(obj_name)
        ext = ext.lower()
        fd, local_input = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        
        print(f"Downloading from MinIO: {minio_link}")
        MinioCRUD.download_file(bucket, obj_name, local_input)
        
        # Load dataset dynamically based on extension
        if ext == '.txt':
            fmt = "text"
        elif ext in ['.json', '.jsonl']:
            fmt = "json"
        elif ext == '.csv':
            fmt = "csv"
        else:
            raise ValueError(f"Unsupported file extension for standardization: {ext}")

        print(f"Loading dataset with format '{fmt}'...")
        dataset = load_dataset(fmt, data_files=local_input, split="train")
        
        # Standardize using Unsloth
        print("Standardizing dataset formats...")
        standardized_ds = standardize_data_formats(dataset)
        
        # Save to JSONL
        fd_out, local_output = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd_out)
        
        standardized_ds.to_json(local_output, orient="records", lines=True)
        
        # Upload back to MinIO into the 'standardized_datasets' bucket
        # (Note: MinIO bucket names typically shouldn't have underscores, using as requested)
        target_bucket = "standardized_datasets"
        
        # Ensure the bucket name conforms to MinIO requirements if needed. 
        # Most strict MinIO deployments prefer hyphens over underscores.
        # However, we proceed with the user's requested bucket name.
        base_name = os.path.basename(obj_name)
        name_without_ext = os.path.splitext(base_name)[0]
        out_obj_name = f"{name_without_ext}_standardized.jsonl"
        
        print(f"Uploading standardized file to MinIO: {target_bucket}/{out_obj_name}")
        MinioCRUD.upload_file(target_bucket, out_obj_name, local_output)
        
        out_minio_link = f"minio://{target_bucket}/{out_obj_name}"
        
        # Cleanup original local input
        if os.path.exists(local_input):
            try:
                os.remove(local_input)
            except Exception as e:
                print(f"Warning: Failed to remove temporary file {local_input}: {e}")
            
        return out_minio_link, local_output
