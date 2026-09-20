import unittest

from src.snapshot import parse_dataset_bytes


class DatasetInputFormatsTest(unittest.TestCase):
    def test_jsonl_csv_and_text(self):
        self.assertEqual(parse_dataset_bytes(b'{"text":"a"}\n{"text":"b"}\n', "x.jsonl"),
                         [{"text": "a"}, {"text": "b"}])
        self.assertEqual(parse_dataset_bytes(b"text,label\na,1\n", "x.csv"),
                         [{"text": "a", "label": "1"}])
        self.assertEqual(parse_dataset_bytes("a\n\nb\n".encode(), "x.txt"),
                         [{"text": "a"}, {"text": "b"}])


if __name__ == "__main__":
    unittest.main()
