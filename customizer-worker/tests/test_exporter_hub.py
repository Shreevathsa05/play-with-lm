import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.modules["unsloth"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.finetuning_modules.model_exporters.unsloth_model_exporter import UnslothModelExporter


class TestExporterHubPush(unittest.TestCase):
    def test_push_to_hub(self):
        model = MagicMock()
        tokenizer = MagicMock()
        repo = UnslothModelExporter.push_to_hub(
            model, tokenizer, repo_id="user/my-model", token="secret", private=True
        )
        self.assertEqual(repo, "user/my-model")
        model.push_to_hub.assert_called_once()
        tokenizer.push_to_hub.assert_called_once()

    def test_push_requires_repo(self):
        with self.assertRaises(ValueError):
            UnslothModelExporter.push_to_hub(MagicMock(), MagicMock(), repo_id="")


if __name__ == "__main__":
    unittest.main()
