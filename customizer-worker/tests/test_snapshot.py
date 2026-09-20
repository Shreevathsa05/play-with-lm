import json
import os
import tempfile
import unittest

from src.snapshot import write_jsonl


class SnapshotTest(unittest.TestCase):
    def test_snapshot_is_canonical_and_repeatable(self):
        rows = [{"b": 2, "a": "x"}, {"text": "hello"}]
        with tempfile.TemporaryDirectory() as directory:
            first, first_hash = write_jsonl(rows, directory)
            second, second_hash = write_jsonl(rows, directory)
            with open(first, encoding="utf-8") as first_stream, open(second, encoding="utf-8") as second_stream:
                first_text = first_stream.read()
                second_text = second_stream.read()
            with open(first, encoding="utf-8") as first_stream:
                first_row = json.loads(first_stream.readline())
            self.assertEqual(first_hash, second_hash)
            self.assertEqual(first_text, second_text)
            self.assertEqual(first_row, {"a": "x", "b": 2})
            self.assertTrue(os.path.exists(first))


if __name__ == "__main__":
    unittest.main()
