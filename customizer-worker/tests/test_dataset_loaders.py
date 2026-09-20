import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.finetuning_modules.dataset_loaders.dataset_loader_module import ExternalDatasetLoader
from src.finetuning_modules.dataset_loaders.minio_dataset_loader import MinioDatasetLoader


class TestDatasetLoaders(unittest.TestCase):
    @patch("src.finetuning_modules.dataset_loaders.dataset_loader_module.load_dataset")
    def test_external_loader(self, mock_load):
        mock_load.return_value = {"ok": True}
        ds = ExternalDatasetLoader.load(path="imdb", split="train[:1%]")
        mock_load.assert_called_once()
        self.assertEqual(ds, {"ok": True})

    @patch("src.finetuning_modules.dataset_loaders.minio_dataset_loader.load_dataset")
    @patch("src.finetuning_modules.dataset_loaders.minio_dataset_loader.MinioCRUD")
    def test_minio_loader(self, mock_crud, mock_load):
        mock_load.return_value = MagicMock()
        MinioDatasetLoader.load("minio://bucket/data.jsonl")
        mock_crud.download_file.assert_called_once()
        self.assertEqual(mock_load.call_args.kwargs["path"], "json")

    def test_minio_unsupported_ext(self):
        with self.assertRaises(ValueError):
            MinioDatasetLoader.load("minio://bucket/data.parquet")


if __name__ == "__main__":
    unittest.main()
