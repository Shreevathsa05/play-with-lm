import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.job_runner import load_payload, main


class TestJobRunner(unittest.TestCase):
    def test_load_payload(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"job_uuid": "x", "recipe": "lora"}, f)
            path = f.name
        try:
            data = load_payload(path)
            self.assertEqual(data["job_uuid"], "x")
        finally:
            os.remove(path)

    @patch("src.job_runner.run_pipeline")
    def test_main_success(self, mock_run):
        mock_run.return_value = {"status": "COMPLETED", "job_uuid": "j"}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"job_uuid": "j", "recipe": "lora"}, f)
            path = f.name
        try:
            code = main(["--payload", path])
            self.assertEqual(code, 0)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
