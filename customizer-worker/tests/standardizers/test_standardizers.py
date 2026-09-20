import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import shutil
import json

sys.modules["unsloth"] = MagicMock()
sys.modules["unsloth.chat_templates"] = MagicMock()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.finetuning_modules.standardizers.standardizer_module import DataStandardizationModule


class _FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def to_json(self, path, orient="records", lines=True):
        with open(path, "w", encoding="utf-8") as f:
            for row in self.rows:
                f.write(json.dumps(row) + "\n")


def _run_standardizer(sample_name: str, link: str):
    test_file = os.path.join(os.path.dirname(__file__), "data", sample_name)

    def mock_download(bucket, obj_name, local_input):
        shutil.copy(test_file, local_input)

    with patch("src.finetuning_modules.standardizers.standardizer_module.MinioCRUD") as mock_minio, patch(
        "src.finetuning_modules.standardizers.standardizer_module.load_dataset"
    ) as mock_load, patch(
        "src.finetuning_modules.standardizers.standardizer_module.standardize_data_formats",
        side_effect=lambda ds: ds,
    ):
        mock_minio.download_file.side_effect = mock_download
        mock_minio.upload_file.return_value = None
        mock_load.return_value = _FakeDataset([{"text": "hello"}, {"text": "world"}])

        link_out, out_file = DataStandardizationModule.process(link)
        return link_out, out_file, mock_minio, mock_load


class TestJSONStandardizer(unittest.TestCase):
    def test_json_standardization(self):
        link, out_file, mock_minio, mock_load = _run_standardizer(
            "sample.json", "minio://test-bucket/sample.json"
        )
        self.assertTrue(link.startswith("minio://standardized_datasets/sample_standardized.jsonl"))
        self.assertTrue(os.path.exists(out_file))
        mock_load.assert_called_once()
        self.assertEqual(mock_load.call_args.args[0], "json")
        mock_minio.upload_file.assert_called_once()
        with open(out_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertGreater(len(lines), 0)
            for line in lines:
                json.loads(line)
        os.remove(out_file)


class TestJSONLStandardizer(unittest.TestCase):
    def test_jsonl_standardization(self):
        link, out_file, _, mock_load = _run_standardizer(
            "sample.jsonl", "minio://test-bucket/sample.jsonl"
        )
        self.assertTrue(link.endswith("sample_standardized.jsonl"))
        self.assertEqual(mock_load.call_args.args[0], "json")
        os.remove(out_file)


class TestCSVStandardizer(unittest.TestCase):
    def test_csv_standardization(self):
        link, out_file, _, mock_load = _run_standardizer(
            "sample.csv", "minio://test-bucket/sample.csv"
        )
        self.assertTrue(link.endswith("sample_standardized.jsonl"))
        self.assertEqual(mock_load.call_args.args[0], "csv")
        os.remove(out_file)


class TestTXTStandardizer(unittest.TestCase):
    def test_txt_standardization(self):
        link, out_file, _, mock_load = _run_standardizer(
            "sample.txt", "minio://test-bucket/sample.txt"
        )
        self.assertTrue(link.endswith("sample_standardized.jsonl"))
        self.assertEqual(mock_load.call_args.args[0], "text")
        os.remove(out_file)


if __name__ == "__main__":
    unittest.main()
