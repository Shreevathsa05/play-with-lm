"""Unit tests for HuggingFace multi-config loading helpers."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.hf_loader import (
    load_hf_dataset,
    parse_config_and_split,
    parse_hf_repo_and_config,
    pick_hf_config,
    pick_hf_split,
)


class TestHfConfigHelpers(unittest.TestCase):
    def test_parse_repo_config(self):
        repo, cfg = parse_hf_repo_and_config("HuggingFaceTB/smoltalk2:SFT")
        self.assertEqual(repo, "HuggingFaceTB/smoltalk2")
        self.assertEqual(cfg, "SFT")

    def test_explicit_config_wins(self):
        repo, cfg = parse_hf_repo_and_config("HuggingFaceTB/smoltalk2:Mid", "SFT")
        self.assertEqual(repo, "HuggingFaceTB/smoltalk2")
        self.assertEqual(cfg, "SFT")

    def test_parse_config_and_split(self):
        self.assertEqual(
            parse_config_and_split("SFT/OpenHermes_2.5_no_think"),
            ("SFT", "OpenHermes_2.5_no_think"),
        )
        self.assertEqual(parse_config_and_split("SFT"), ("SFT", None))

    def test_pick_prefers_sft(self):
        self.assertEqual(pick_hf_config(["Mid", "Preference", "SFT"]), "SFT")

    def test_pick_split_avoids_openthoughts(self):
        available = [
            "OpenThoughts3_1.2M_think",
            "Mixture_of_Thoughts_science_no_think",
            "LongAlign_64k_Qwen3_32B_yarn_131k_think",
        ]
        self.assertEqual(
            pick_hf_split(available), "Mixture_of_Thoughts_science_no_think"
        )

    @patch("src.hf_loader.get_dataset_split_names", return_value=["OpenThoughts3_1.2M_think", "OpenHermes_2.5_no_think"])
    @patch("src.hf_loader.get_dataset_config_names", return_value=["Mid", "Preference", "SFT"])
    @patch("src.hf_loader.load_dataset")
    def test_load_streams_compact_split(self, mock_load, _configs, _splits):
        mock_load.return_value = iter(
            [{"messages": [{"role": "user", "content": "hi"}]} for _ in range(5)]
        )
        rows = load_hf_dataset("HuggingFaceTB/smoltalk2", max_rows=3)
        self.assertEqual(len(rows), 3)
        kwargs = mock_load.call_args.kwargs
        self.assertEqual(kwargs.get("name"), "SFT")
        self.assertEqual(kwargs.get("split"), "OpenHermes_2.5_no_think")
        self.assertTrue(kwargs.get("streaming"))

    @patch("src.hf_loader.get_dataset_split_names", return_value=["train"])
    @patch("src.hf_loader.get_dataset_config_names", return_value=["Mid", "Preference", "SFT"])
    @patch("src.hf_loader.load_dataset")
    def test_load_retries_with_sft_when_config_missing(self, mock_load, _configs, _splits):
        def side_effect(*args, **kwargs):
            if "name" not in kwargs:
                raise ValueError(
                    "Config name is missing. Please pick one among the available configs: "
                    "['Mid', 'Preference', 'SFT'] Example of usage: `load_dataset('smoltalk2', 'Mid')`"
                )
            return iter([{"a": 1}, {"b": 2}])

        mock_load.side_effect = side_effect
        # Force the missing-config path by pretending catalog lookup failed.
        with patch("src.hf_loader.get_dataset_config_names", side_effect=RuntimeError("offline")):
            rows = load_hf_dataset("HuggingFaceTB/smoltalk2")
        self.assertEqual(len(rows), 2)
        self.assertTrue(any(call.kwargs.get("name") == "SFT" for call in mock_load.call_args_list))
        self.assertTrue(all(call.kwargs.get("streaming") for call in mock_load.call_args_list))


if __name__ == "__main__":
    unittest.main()
