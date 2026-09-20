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

from src.finetuning_modules.model_trainers.unsloth_model_trainer import UnslothModelTrainer

class TestModelTrainers(unittest.TestCase):
    @patch('src.finetuning_modules.model_trainers.unsloth_model_trainer.is_bfloat16_supported')
    @patch('src.finetuning_modules.model_trainers.unsloth_model_trainer.SFTConfig')
    @patch('src.finetuning_modules.model_trainers.unsloth_model_trainer.SFTTrainer')
    def test_train_model(self, mock_sft_trainer, mock_training_args, mock_bf16):
        # Mocks
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_dataset = MagicMock()
        mock_bf16.return_value = True
        
        mock_trainer_inst = MagicMock()
        mock_sft_trainer.return_value = mock_trainer_inst
        mock_trainer_inst.train.return_value = "train_output_details"
        
        result = UnslothModelTrainer.train(
            model=mock_model,
            tokenizer=mock_tokenizer,
            dataset=mock_dataset,
            output_dir="test_outputs",
            dataset_text_field="text_field",
            max_seq_length=512,
            learning_rate=1e-4,
            batch_size=4,
            grad_accum_steps=2,
            epochs=2.0
        )
        
        # Verify training arguments creation
        mock_training_args.assert_called_once_with(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_ratio=0.03,
            num_train_epochs=2.0,
            max_steps=-1,
            learning_rate=1e-4,
            fp16=False,
            bf16=True,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="test_outputs",
            report_to="none",
            max_length=512,
            dataset_num_proc=None if __import__("sys").platform.startswith("win") else 2,
            packing=False,
            dataset_text_field="text_field",
        )
        
        # Verify SFTTrainer creation
        mock_sft_trainer.assert_called_once()
        self.assertEqual(mock_sft_trainer.call_args.kwargs["processing_class"], mock_tokenizer)
        self.assertEqual(result, "train_output_details")
        mock_trainer_inst.train.assert_called_once()

    @patch('src.finetuning_modules.model_trainers.unsloth_model_trainer.is_bfloat16_supported')
    @patch('src.finetuning_modules.model_trainers.unsloth_model_trainer.SFTConfig')
    @patch('src.finetuning_modules.model_trainers.unsloth_model_trainer.SFTTrainer')
    def test_force_float32_disables_mixed_precision(self, mock_sft_trainer, mock_training_args, mock_bf16):
        mock_bf16.return_value = False
        mock_sft_trainer.return_value.train.return_value = "ok"
        UnslothModelTrainer.train(
            model=MagicMock(),
            tokenizer=MagicMock(),
            dataset=MagicMock(),
            force_float32=True,
        )
        kwargs = mock_training_args.call_args.kwargs
        self.assertFalse(kwargs["fp16"])
        self.assertFalse(kwargs["bf16"])


if __name__ == '__main__':
    unittest.main()
