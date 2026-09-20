import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock heavy ML deps before imports
sys.modules["unsloth"] = MagicMock()
sys.modules["unsloth.chat_templates"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils.minio_links import parse_minio_link
from src.utils.logcarrier import LogCarrier


class TestParseMinioLink(unittest.TestCase):
    def test_minio_scheme(self):
        self.assertEqual(parse_minio_link("minio://bucket/path/file.json"), ("bucket", "path/file.json"))

    def test_s3_scheme(self):
        self.assertEqual(parse_minio_link("s3://b/obj"), ("b", "obj"))

    def test_bare(self):
        self.assertEqual(parse_minio_link("datasets/raw.jsonl"), ("datasets", "raw.jsonl"))


class TestLogCarrier(unittest.TestCase):
    def test_redacts_secrets_and_emails(self):
        logger = MagicMock()
        carrier = LogCarrier("job-abc", logger=logger)
        record = carrier.info("started", hf_token="secret-token", note="user@example.com ok")
        self.assertEqual(record["job_uuid"], "job-abc")
        self.assertEqual(record["hf_token"], "[REDACTED]")
        self.assertIn("[REDACTED_EMAIL]", record["note"])
        logger.log.assert_called_once()

    def test_requires_job_uuid(self):
        with self.assertRaises(ValueError):
            LogCarrier("")


if __name__ == "__main__":
    unittest.main()
