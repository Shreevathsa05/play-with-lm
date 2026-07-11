import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import shutil
import json

# Mock unsloth completely to bypass Windows/Torch 2.6 environment crashes during import
sys.modules['unsloth'] = MagicMock()
sys.modules['unsloth.chat_templates'] = MagicMock()
sys.modules['unsloth.chat_templates'].standardize_data_formats = lambda x: x

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.finetuning_modules.standardizers.standardizer_module import DataStandardizationModule

class TestJSONLStandardizer(unittest.TestCase):
    @patch('src.finetuning_modules.standardizers.standardizer_module.MinioCRUD')
    def test_jsonl_standardization(self, mock_minio):
        test_file = os.path.join(os.path.dirname(__file__), 'data', 'sample.jsonl')
        
        def mock_download(bucket, obj_name, local_input):
            shutil.copy(test_file, local_input)
            
        mock_minio.download_file.side_effect = mock_download
        mock_minio.upload_file.return_value = None
        
        link, out_file = DataStandardizationModule.process("minio://test-bucket/sample.jsonl")
        
        self.assertTrue(link.startswith("minio://standardized_datasets/sample_standardized.jsonl"))
        self.assertTrue(os.path.exists(out_file))
        
        # Verify the output is valid JSONL
        with open(out_file, 'r') as f:
            lines = f.readlines()
            self.assertTrue(len(lines) > 0)
            for line in lines:
                obj = json.loads(line)
                self.assertIsNotNone(obj)
                
        # Cleanup
        if os.path.exists(out_file):
            os.remove(out_file)

if __name__ == '__main__':
    unittest.main()
