import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Mock dependencies to bypass CUDA/GPU/environment crashes during imports
sys.modules['unsloth'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finetuning_modules.model_exporters.unsloth_model_exporter import UnslothModelExporter

class TestModelExporters(unittest.TestCase):
    def test_export_lora_local(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Test lora export locally
        res = UnslothModelExporter.export_model(
            model=mock_model,
            tokenizer=mock_tokenizer,
            export_format="lora",
            local_output_dir="test_export_lora"
        )
        
        mock_model.save_pretrained.assert_called_once_with("test_export_lora")
        mock_tokenizer.save_pretrained.assert_called_once_with("test_export_lora")
        self.assertEqual(res, "test_export_lora")
        
        # Clean up created directory
        if os.path.exists("test_export_lora"):
            try:
                os.rmdir("test_export_lora")
            except Exception:
                pass

    @patch('src.finetuning_modules.model_exporters.unsloth_model_exporter.MinioCRUD')
    def test_export_merged_16bit_minio(self, mock_minio_crud):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Setup files inside temp directory to simulate output
        original_mkdtemp = tempfile.mkdtemp
        temp_dirs = []
        
        def mock_mkdtemp(*args, **kwargs):
            d = original_mkdtemp(*args, **kwargs)
            temp_dirs.append(d)
            # Create a mock file inside it
            with open(os.path.join(d, "config.json"), "w") as f:
                f.write("{}")
            return d
            
        with patch('tempfile.mkdtemp', side_effect=mock_mkdtemp):
            res = UnslothModelExporter.export_model(
                model=mock_model,
                tokenizer=mock_tokenizer,
                export_format="merged_16bit",
                minio_destination_link="minio://models-bucket/output/llama-merged"
            )
            
        mock_model.save_pretrained_merged.assert_called_once()
        call_args = mock_model.save_pretrained_merged.call_args
        self.assertEqual(call_args[0][1], mock_tokenizer)
        self.assertEqual(call_args[1]["save_method"], "merged_16bit")
        
        # Verify minio upload was called
        mock_minio_crud.upload_file.assert_called_once_with(
            "models-bucket",
            "output/llama-merged/config.json",
            os.path.join(temp_dirs[0], "config.json")
        )
        self.assertEqual(res, "minio://models-bucket/output/llama-merged")

if __name__ == '__main__':
    unittest.main()
