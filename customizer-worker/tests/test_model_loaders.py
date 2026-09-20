import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Mock dependencies to bypass CUDA/GPU/environment crashes during imports
sys.modules['unsloth'] = MagicMock()
sys.modules['trl'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.finetuning_modules.model_loaders.unsloth_model_loader import UnslothModelLoader

class TestModelLoaders(unittest.TestCase):
    @patch('src.finetuning_modules.model_loaders.unsloth_model_loader.FastLanguageModel')
    def test_load_from_hf(self, mock_flm):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        
        model, tokenizer = UnslothModelLoader.load_model("unsloth/llama-3-8b-Instruct-bnb-4bit", max_seq_length=1024)
        
        mock_flm.from_pretrained.assert_called_once_with(
            model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
            load_in_8bit=False,
            device_map="auto",
            full_finetuning=False,
        )
        self.assertEqual(model, mock_model)
        self.assertEqual(tokenizer, mock_tokenizer)

    @patch('src.finetuning_modules.model_loaders.unsloth_model_loader.FastLanguageModel')
    def test_full_finetuning_is_enabled_at_load(self, mock_flm):
        mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())
        UnslothModelLoader.load_model("model/id", load_in_4bit=False, full_finetuning=True)
        self.assertTrue(mock_flm.from_pretrained.call_args.kwargs["full_finetuning"])

    @patch('src.finetuning_modules.model_loaders.unsloth_model_loader.minio_client')
    @patch('src.finetuning_modules.model_loaders.unsloth_model_loader.MinioCRUD')
    @patch('src.finetuning_modules.model_loaders.unsloth_model_loader.FastLanguageModel')
    def test_load_from_minio(self, mock_flm, mock_minio_crud, mock_client):
        # Setup mock files inside bucket
        mock_obj1 = MagicMock()
        mock_obj1.object_name = "models/llama/config.json"
        mock_obj2 = MagicMock()
        mock_obj2.object_name = "models/llama/model.safetensors"
        mock_client.list_objects.return_value = [mock_obj1, mock_obj2]
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_flm.from_pretrained.return_value = (mock_model, mock_tokenizer)
        
        model, tokenizer = UnslothModelLoader.load_model("minio://test-bucket/models/llama")
        
        # Verify downloading objects from MinIO
        mock_client.list_objects.assert_called_once_with("test-bucket", prefix="models/llama", recursive=True)
        self.assertEqual(mock_minio_crud.download_file.call_count, 2)
        
        # Verify from_pretrained was called
        self.assertTrue(mock_flm.from_pretrained.called)
        call_kwargs = mock_flm.from_pretrained.call_args[1]
        self.assertTrue("unsloth_model_" in call_kwargs["model_name"])

    @patch('src.finetuning_modules.model_loaders.unsloth_model_loader.FastLanguageModel')
    def test_get_peft_model(self, mock_flm):
        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_flm.get_peft_model.return_value = mock_peft_model
        
        peft_model = UnslothModelLoader.get_peft_model(mock_model, r=8, lora_alpha=32)
        
        mock_flm.get_peft_model.assert_called_once_with(
            model=mock_model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None
        )
        self.assertEqual(peft_model, mock_peft_model)

if __name__ == '__main__':
    unittest.main()
